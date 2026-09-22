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
from .client import DEFAULT_MODEL, Client
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

    def explain(self, state: State, question: Question, **kwargs: Any) -> Attribution:
        """Synchronous :meth:`aexplain`.

        Raises inside a running event loop rather than deadlocking; use
        ``aexplain`` there.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "explain() cannot run inside an active event loop; await aexplain() instead"
            )

        async def run() -> Attribution:
            try:
                return await self.aexplain(state, question, **kwargs)
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
