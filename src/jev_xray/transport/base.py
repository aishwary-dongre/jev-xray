"""The transport seam.

Everything above this line — segmenters, attribution, the searches, the report —
only needs the ``/v1/systemone`` contract. It does not care whether the answer
came from TypeSafe directly, a gateway reselling Jev, a local open reproduction
serving the same schema, or a recorded fixture. Keeping that seam narrow is what
lets the toolkit be built and tested while hosted access is closed.

A transport returns the *raw* response body. Parsing and validation happen above
it, so the cache stores bodies rather than typed objects and a schema correction
does not invalidate recorded traffic.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ..types import SystemOneRequest

__all__ = ["Transport", "TransportError"]


class TransportError(RuntimeError):
    """A request did not produce a response body.

    Carries a status code where one exists and nothing else. Request state,
    credentials and remote error bodies are deliberately excluded so that an
    exception trace cannot become the place your production data leaks.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@runtime_checkable
class Transport(Protocol):
    async def send(self, request: SystemOneRequest) -> Mapping[str, Any]:
        """Evaluate one request and return the raw response body."""
        ...

    async def aclose(self) -> None:
        """Release any held connections."""
        ...
