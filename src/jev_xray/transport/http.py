"""HTTP transport for TypeSafe directly, or any gateway fronting the same contract.

``POST <endpoint>`` with bearer auth and a JSON body of ``model``, ``state`` and
``questions``. That is the whole protocol.

Endpoint and model id are explicit rather than baked in per provider. Gateway
URLs and their model naming are theirs to change, and guessing them here would
produce confident 404s; take both from the provider's own docs and pass them in.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Mapping

from ..types import SystemOneRequest
from .base import TransportError

__all__ = [
    "HttpTransport",
    "TYPESAFE_ENDPOINT",
    "VERCEL_TYPESAFE_ENDPOINT",
    "LOCAL_ENDPOINT",
    "PROVIDERS",
    "resolve_provider",
]

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

# Vercel AI Gateway exposes two different things:
#
#   /typesafe        a TypeSafe-compatible surface. Same systemOne calls, same
#                    question types, same response shapes; only the base URL and
#                    key change. This is the one we use, because our wire types
#                    already are the TypeSafe contract.
#   /v1/evaluate     Vercel's own naming, where a Noul is called `boolean` and
#                    the answer field is `probability`. A different contract that
#                    would need a translation layer, so we deliberately skip it.
VERCEL_TYPESAFE_ENDPOINT = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"

# name -> (endpoint, default model, env var holding the key)
# A local open reproduction. LitJev, openjev-sglang and others serve the same
# /v1/systemone schema on open weights, which is the only route that needs
# neither an account nor a payment method.
LOCAL_ENDPOINT = "http://localhost:8000/v1/systemone"

PROVIDERS: dict[str, tuple[str, str, str]] = {
    "typesafe": (TYPESAFE_ENDPOINT, "jev-1.13.0", "TYPESAFE_API_KEY"),
    "vercel": (VERCEL_TYPESAFE_ENDPOINT, "jev-1.13.0", "AI_GATEWAY_API_KEY"),
    "local": (LOCAL_ENDPOINT, "jev-1.13.0", "TYPESAFE_API_KEY"),
}

_KEY_ENV_ORDER = ("TYPESAFE_API_KEY", "AI_GATEWAY_API_KEY")

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def resolve_provider(name: str) -> tuple[str, str, str]:
    try:
        return PROVIDERS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}"
        ) from None


class HttpTransport:
    """Async client for the System One endpoint.

    Retries only on rate limiting and transient server errors, honours
    ``Retry-After`` when the response carries one, and adds jitter so a fan-out
    of a hundred ablations does not resynchronise into a thundering herd after
    the first 429.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        provider: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        include_error_detail: bool = False,
        client: Any | None = None,
    ) -> None:
        preferred_env: tuple[str, ...] = _KEY_ENV_ORDER
        default_endpoint = TYPESAFE_ENDPOINT
        if provider:
            default_endpoint, _model, key_env = resolve_provider(provider)
            # A named provider looks at its own key variable first, so having
            # both a TypeSafe key and a gateway key in .env is unambiguous.
            preferred_env = (key_env, *(e for e in _KEY_ENV_ORDER if e != key_env))

        self._endpoint = (
            endpoint or os.environ.get("JEV_XRAY_BASE_URL") or default_endpoint
        )

        key = api_key
        if key is None:
            for name in preferred_env:
                key = os.environ.get(name)
                if key:
                    break
        if not key and provider and provider.lower() == "local":
            # A local model has nothing to authenticate against, and requiring a
            # placeholder here would be friction for no security benefit.
            key = "local"
        if not key:
            raise ValueError(
                "no API key: pass api_key=, or set TYPESAFE_API_KEY (TypeSafe "
                "direct) or AI_GATEWAY_API_KEY (Vercel AI Gateway). For a local "
                "open reproduction any non-empty placeholder will do."
            )
        if not self._endpoint.lower().startswith(("https://", "http://localhost", "http://127.")):
            raise ValueError("endpoint must be https, or a localhost address for local models")

        self._key = key
        self._timeout = timeout
        self._max_retries = max_retries
        # Off by default: an error body is provider-generated but can echo parts
        # of the request, and request state is exactly what should not end up in
        # an exception trace or a log. `jev-xray check` turns it on deliberately,
        # because the provider's own message is usually the whole answer. A 403
        # saying "add a credit card" is far more useful than "403".
        self._include_error_detail = include_error_detail
        self._owns_client = client is None
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def send(self, request: SystemOneRequest) -> Mapping[str, Any]:
        import httpx

        client = self._ensure_client()
        body = request.wire()
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(self._endpoint, json=body, headers=headers)
            except httpx.TimeoutException as exc:
                if attempt == self._max_retries:
                    raise TransportError("request timed out") from exc
                await self._backoff(attempt, None)
                continue
            except httpx.HTTPError as exc:
                # Deliberately drops the underlying cause's detail, which can
                # echo the URL and headers.
                if attempt == self._max_retries:
                    raise TransportError("transport failure") from None
                await self._backoff(attempt, None)
                continue

            status = response.status_code
            if status < 300:
                try:
                    return response.json()
                except ValueError as exc:
                    raise TransportError("response body was not JSON", status=status) from exc

            last_status = status
            if status in _RETRY_STATUSES and attempt < self._max_retries:
                await self._backoff(attempt, response.headers.get("retry-after"))
                continue

            raise TransportError(
                f"request failed with status {status}{self._detail(response)}",
                status=status,
            )

        raise TransportError("retries exhausted", status=last_status)

    def _detail(self, response: Any) -> str:
        """The provider's own explanation, when detail is explicitly enabled."""
        if not self._include_error_detail:
            return ""
        try:
            body = response.json()
        except ValueError:
            text = (response.text or "").strip()
            return f": {text[:400]}" if text else ""

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("type")
                if message:
                    return f": {str(message)[:400]}"
            if isinstance(error, str):
                return f": {error[:400]}"
            if body.get("message"):
                return f": {str(body['message'])[:400]}"
        return ""

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 60.0))
                return
            except ValueError:
                pass  # a date-formatted Retry-After: fall through to backoff
        delay = min(2.0**attempt, 16.0) * (0.5 + random.random())
        await asyncio.sleep(delay)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
