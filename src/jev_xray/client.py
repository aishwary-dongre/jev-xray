"""The metered client every probe goes through.

Wraps a transport with the four things a probe must not be allowed to skip:
cache lookup, budget check, rate limiting, and usage accounting. Probes call
``ask`` and get a parsed response; they never touch a transport directly, so
there is exactly one place where a runaway fan-out can be stopped.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Mapping

from .budget import Budget, Ledger, RateLimiter, estimate_tokens
from .cache import MemoryCache, ResponseCache, cache_key
from .transport.base import Transport
from .types import Question, State, SystemOneRequest, SystemOneResponse, is_alias, parse_response

__all__ = ["Client", "DEFAULT_MODEL"]

# Pinned on purpose. jev-latest and jev-preview move when a release ships, and
# an explanation or a tuned threshold that cannot name the version that produced
# it is not reproducible.
DEFAULT_MODEL = "jev-1.13.0"


class Client:
    def __init__(
        self,
        transport: Transport,
        *,
        model: str = DEFAULT_MODEL,
        cache: ResponseCache | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        if is_alias(model):
            warnings.warn(
                f"{model!r} is a moving alias: the version behind it can change "
                "without notice, which makes an explanation unreproducible and "
                "silently invalidates any threshold tuned against it. Pin a "
                "version id such as 'jev-1.13.0'.",
                UserWarning,
                stacklevel=2,
            )
        self.transport = transport
        self.model = model
        self.cache: ResponseCache = cache if cache is not None else MemoryCache()
        self.limiter = limiter if limiter is not None else RateLimiter()
        # Identical requests issued concurrently would each miss the cache and
        # hit the network. Ablations collide more often than you would guess —
        # two segments carrying the same evidence produce byte-identical states —
        # so in-flight requests are coalesced onto one call.
        self._inflight: dict[str, asyncio.Future[Mapping[str, Any]]] = {}

    async def ask(
        self,
        state: State,
        questions: Mapping[str, Question],
        *,
        ledger: Ledger,
        budget: Budget,
    ) -> SystemOneResponse:
        request = SystemOneRequest(model=self.model, state=state, questions=questions)

        key = cache_key(request)

        cached = self.cache.get(key)
        if cached is not None:
            ledger.record_cache_hit()
            return parse_response(request, cached)

        pending = self._inflight.get(key)
        if pending is not None:
            ledger.record_coalesced()
            return parse_response(request, await asyncio.shield(pending))

        wire = request.wire()
        estimated = estimate_tokens(wire)
        ledger.check(budget, about_to_spend_tokens=estimated)

        future: asyncio.Future[Mapping[str, Any]] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            await self.limiter.acquire(estimated)
            body = await self.transport.send(request)
        except BaseException as exc:
            future.set_exception(exc)
            # Followers observe the same failure; nobody is left awaiting a
            # future that will never resolve.
            raise
        else:
            future.set_result(body)
        finally:
            self._inflight.pop(key, None)
            if future.done() and not future.cancelled() and future.exception() is not None:
                future.exception()  # mark retrieved so asyncio stays quiet

        parsed = parse_response(request, body)
        reported = parsed.usage.input_tokens
        ledger.record(
            input_tokens=reported or estimated,
            output_tokens=parsed.usage.output_tokens,
            estimated=not reported,
        )
        self.cache.put(key, dict(body))
        return parsed

    async def aclose(self) -> None:
        await self.transport.aclose()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
