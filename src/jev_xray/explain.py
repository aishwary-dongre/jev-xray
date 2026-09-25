"""The front door.

``XRay`` is the one object most callers need. It assembles a transport, a cache,
a rate limiter and a budget into something you can ask a single question of, and
it defaults to a pinned model version rather than a moving alias.

Two ways in. ``explain`` is synchronous and fine for a script, a notebook cell or
a CLI. ``aexplain`` is the real interface: attribution is a fan-out of
independent requests and the async path is what makes a forty-segment
explanation finish in about the time of a single call.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .attribution import Attribution, leave_one_out
from .budget import Budget, RateLimiter
from .cache import MemoryCache, ResponseCache
from .client import DEFAULT_MODEL, Client  # noqa: F401  (Client used in aprobe)
from .segment import AblationMode, Segmenter
from .transport.base import Transport
from .transport.fake import FakeTransport, Signal
from .types import Question, State, Target

__all__ = ["XRay"]


class XRay:
    """Explains one Jev decision at a time."""

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        model: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        provider: str | None = None,
        cache: ResponseCache | None = None,
        limiter: RateLimiter | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.model = model or os.environ.get("JEV_XRAY_MODEL") or DEFAULT_MODEL
        self.budget = budget if budget is not None else Budget()
        self._cache = cache if cache is not None else MemoryCache()
        self._limiter = limiter if limiter is not None else RateLimiter()

        self._owns_transport = transport is None
        if transport is None:
            from .transport.http import HttpTransport

            transport = HttpTransport(
                api_key=api_key, endpoint=endpoint, provider=provider
            )
        self._transport = transport

    # -- constructors -------------------------------------------------------

    @classmethod
    def fake(cls, signals: Any = (), *, bias: float = 0.0, **kwargs: Any) -> "XRay":
        """An offline instance backed by the deterministic fake.

        For development and tests, and for demonstrating the tool while hosted
        access is closed. The judgments are fixtures, not predictions; only the
        machinery around them is real.
        """
        transport = FakeTransport(signals=tuple(signals), bias=bias)
        instance = cls(transport=transport, **kwargs)
        instance._owns_transport = True
        return instance

    # -- the probe ----------------------------------------------------------

    async def aexplain(
        self,
        state: State,
        question: Question,
        *,
        question_id: str = "q",
        segmenter: str | Segmenter = "auto",
        target: Target | None = None,
        mode: AblationMode = "delete",
        budget: Budget | None = None,
        concurrency: int = 8,
        include_empty: bool = True,
    ) -> Attribution:
        """Attribute one answer across its state. See :func:`leave_one_out`."""
        client = Client(
            self._transport,
            model=self.model,
            cache=self._cache,
            limiter=self._limiter,
        )
        return await leave_one_out(
            client,
            state,
            question,
            question_id=question_id,
            segmenter=segmenter,
            target=target,
            mode=mode,
            budget=budget if budget is not None else self.budget,
            concurrency=concurrency,
            include_empty=include_empty,
        )

    async def aprobe(
        self,
        state: State,
        question: Question,
        *,
        method: str = "deep",
        **kwargs: Any,
    ) -> Any:
        """Run one of the estimators.

        ``loo``
            Leave-one-out. One request per segment. Cheapest, and blind to
            evidence that only counts in combination.
        ``shapley``
            Average marginal contribution over subsets. Exact when affordable,
            sampled otherwise. Correct for interacting and redundant evidence.
        ``deep``
            Shapley plus the smallest sufficient evidence and the smallest change
            that flips the decision, all sharing one coalition cache.
        """
        if method == "loo":
            return await self.aexplain(state, question, **kwargs)

        client = Client(
            self._transport,
            model=self.model,
            cache=self._cache,
            limiter=self._limiter,
        )
        budget = kwargs.pop("budget", None) or self.budget

        if method == "shapley":
            from .shapley import shapley

            return await shapley(client, state, question, budget=budget, **kwargs)

        if method == "deep":
            from .probe import deep_explain

            return await deep_explain(client, state, question, budget=budget, **kwargs)

        raise ValueError(f"unknown method {method!r}; expected loo, shapley or deep")

    def probe(
        self, state: State, question: Question, *, method: str = "deep", **kwargs: Any
    ) -> Any:
        """Synchronous :meth:`aprobe`."""
        return self._run(self.aprobe(state, question, method=method, **kwargs))

    async def astability(
        self, state: State, question: Question, **kwargs: Any
    ) -> Any:
        """Run the stability suite: is this question safe to threshold on?

        Distinct from attribution. Attribution explains one answer; this asks
        whether the question itself is sound, by perturbing things that should not
        change the answer and measuring whether they did.
        """
        from .stability import stability as _stability

        client = Client(
            self._transport,
            model=self.model,
            cache=self._cache,
            limiter=self._limiter,
        )
        budget = kwargs.pop("budget", None) or self.budget
        return await _stability(client, state, question, budget=budget, **kwargs)

    def stability(self, state: State, question: Question, **kwargs: Any) -> Any:
        """Synchronous :meth:`astability`."""
        return self._run(self.astability(state, question, **kwargs))

    def explain(self, state: State, question: Question, **kwargs: Any) -> Attribution:
        """Synchronous :meth:`aexplain`."""
        return self._run(self.aexplain(state, question, **kwargs))

    def _run(self, coro: Any) -> Any:
        """Drive a coroutine to completion, refusing to nest inside a live loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise RuntimeError(
                "the synchronous API cannot run inside an active event loop; "
                "await the async form instead"
            )

        async def run() -> Any:
            try:
                return await coro
            finally:
                if self._owns_transport:
                    # HttpTransport rebuilds its connection pool lazily, so
                    # closing here keeps repeated sync calls loop-safe.
                    await self._transport.aclose()

        return asyncio.run(run())

    # -- lifecycle ----------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_transport:
            await self._transport.aclose()

    async def __aenter__(self) -> "XRay":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


# Re-exported for convenience: declaring signals is how you drive the fake.
__all__ += ["Signal"]
