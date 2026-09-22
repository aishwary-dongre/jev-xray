"""Cost accounting and rate limiting.

Attribution is request-heavy by construction: one explanation is tens to
hundreds of evaluations of the same question against ablated states. The money
is negligible at $0.042 per million input tokens with output free, but the
request rate is not, and neither is the risk of a runaway loop. So a budget is
a required argument rather than an optional safety net, and every explanation
reports what it actually spent.

Pre-flight checks use an estimated token count because the vendor tokenizer is
not public. Accounting afterwards uses the ``usage.input_tokens`` the service
reports, which is authoritative. The two are labelled separately and never
mixed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

__all__ = [
    "PRICE_USD_PER_INPUT_TOKEN",
    "DEFAULT_REQUESTS_PER_MINUTE",
    "DEFAULT_TOKENS_PER_SECOND",
    "Budget",
    "BudgetExceeded",
    "Ledger",
    "RateLimiter",
    "TokenBucket",
    "estimate_tokens",
]

# $0.042 per million input tokens; output tokens are free.
PRICE_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000

# Published limits for jev-1.13. A fresh account may be provisioned lower, so
# these are defaults to override, not facts to rely on.
DEFAULT_REQUESTS_PER_MINUTE = 1_200
DEFAULT_TOKENS_PER_SECOND = 250_000

# Rough characters-per-token ratio for English prose. Only ever used for
# pre-flight estimates, never for reporting spend.
_CHARS_PER_TOKEN = 4


def estimate_tokens(payload: object) -> int:
    """Estimate the input tokens a payload will bill.

    Deliberately crude. Its only job is to stop an explanation before it starts
    if the plan obviously blows the budget.
    """
    if payload is None:
        return 0
    if isinstance(payload, str):
        text = payload
    else:
        import json

        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(payload)
    return max(1, len(text) // _CHARS_PER_TOKEN)


class BudgetExceeded(RuntimeError):
    """A probe stopped because it would have exceeded its declared budget."""


@dataclass(frozen=True, slots=True)
class Budget:
    """A ceiling on what one probe may spend.

    ``None`` on any field means unlimited for that dimension, which is rarely
    what you want outside a test.
    """

    max_requests: int | None = 200
    max_usd: float | None = 0.05
    max_input_tokens: int | None = None

    @classmethod
    def unlimited(cls) -> "Budget":
        return cls(max_requests=None, max_usd=None, max_input_tokens=None)


@dataclass(slots=True)
class Ledger:
    """Running record of what a probe has actually spent."""

    requests: int = 0
    cached_requests: int = 0
    coalesced_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_requests: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _finished_at: float | None = None

    @property
    def usd(self) -> float:
        return self.input_tokens * PRICE_USD_PER_INPUT_TOKEN

    @property
    def wall_seconds(self) -> float:
        end = self._finished_at if self._finished_at is not None else time.monotonic()
        return end - self.started_at

    def finish(self) -> None:
        self._finished_at = time.monotonic()

    def record(
        self, *, input_tokens: int, output_tokens: int = 0, estimated: bool = False
    ) -> None:
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if estimated:
            self.estimated_requests += 1

    @property
    def tokens_are_estimated(self) -> bool:
        """True when any request's token count came from our own estimate.

        Not every host reports usage. When it does not, spend is a projection at
        the configured price rather than a measurement, and a report that blurs
        the two is worse than one that admits the gap.
        """
        return self.estimated_requests > 0

    def record_cache_hit(self) -> None:
        self.cached_requests += 1

    def record_coalesced(self) -> None:
        """An identical request was already in flight; this one rode along."""
        self.coalesced_requests += 1

    @property
    def avoided_requests(self) -> int:
        """Requests the cache and coalescing saved."""
        return self.cached_requests + self.coalesced_requests

    def check(self, budget: Budget, *, about_to_spend_tokens: int = 0) -> None:
        """Raise if issuing one more request would breach ``budget``.

        ``about_to_spend_tokens`` is an estimate, so this is a guard rail rather
        than an exact accountant.
        """
        if budget.max_requests is not None and self.requests + 1 > budget.max_requests:
            raise BudgetExceeded(
                f"request budget exhausted: {self.requests} already spent, "
                f"limit is {budget.max_requests}"
            )
        if budget.max_input_tokens is not None:
            projected = self.input_tokens + about_to_spend_tokens
            if projected > budget.max_input_tokens:
                raise BudgetExceeded(
                    f"token budget exhausted: ~{projected} projected, "
                    f"limit is {budget.max_input_tokens}"
                )
        if budget.max_usd is not None:
            projected_usd = (
                self.input_tokens + about_to_spend_tokens
            ) * PRICE_USD_PER_INPUT_TOKEN
            if projected_usd > budget.max_usd:
                raise BudgetExceeded(
                    f"cost budget exhausted: ~${projected_usd:.6f} projected, "
                    f"limit is ${budget.max_usd:.6f}"
                )

    def summary(self) -> str:
        bits = [f"{self.requests} requests"]
        if self.avoided_requests:
            bits.append(f"{self.avoided_requests} avoided")
        suffix = " (estimated)" if self.tokens_are_estimated else ""
        bits.append(f"{self.input_tokens:,} input tokens{suffix}")
        bits.append(f"${self.usd:.6f}{suffix}")
        bits.append(f"{self.wall_seconds:.2f}s wall")
        return ", ".join(bits)


class TokenBucket:
    """Asyncio token bucket. Refills lazily from a monotonic clock."""

    __slots__ = ("_rate", "_capacity", "_tokens", "_updated", "_lock")

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)
        self._capacity = float(capacity if capacity is not None else rate)
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        # A single request larger than the bucket would never fill; clamp it so
        # it waits one full bucket instead of deadlocking.
        amount = min(float(amount), self._capacity)
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = amount - self._tokens
                wait = deficit / self._rate
            await asyncio.sleep(wait)


class RateLimiter:
    """Holds a probe inside the account's requests-per-minute and tokens-per-second limits."""

    __slots__ = ("_requests", "_tokens")

    def __init__(
        self,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        tokens_per_second: int = DEFAULT_TOKENS_PER_SECOND,
        *,
        burst_seconds: float = 1.0,
    ) -> None:
        rps = requests_per_minute / 60.0
        self._requests = TokenBucket(rps, capacity=max(1.0, rps * burst_seconds))
        self._tokens = TokenBucket(
            tokens_per_second, capacity=max(1.0, tokens_per_second * burst_seconds)
        )

    async def acquire(self, estimated_tokens: int) -> None:
        await self._requests.acquire(1.0)
        await self._tokens.acquire(max(1, estimated_tokens))

    @classmethod
    def unlimited(cls) -> "RateLimiter":
        return cls(requests_per_minute=10**9, tokens_per_second=10**12)
