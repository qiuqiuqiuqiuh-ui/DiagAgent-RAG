from collections import OrderedDict
from collections.abc import Callable, Coroutine
from hmac import compare_digest
from math import ceil
from threading import Lock
from time import monotonic
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.routing import APIRoute
from pydantic import SecretStr

from src.core.settings import ApiSettings

_MAX_BUCKETS = 4096


class TokenBuckets:
    def __init__(self, requests_per_minute: int) -> None:
        self._capacity = requests_per_minute
        self._refill = requests_per_minute / 60
        self._buckets: OrderedDict[tuple[str, bool], tuple[float, float]] = OrderedDict()
        self._lock = Lock()

    def retry_after(self, scope: tuple[str, bool]) -> int | None:
        with self._lock:
            now = monotonic()
            while self._buckets and next(iter(self._buckets.values()))[1] <= now - 60:
                self._buckets.popitem(last=False)
            previous, updated = self._buckets.pop(scope, (float(self._capacity), now))
            available = min(self._capacity, previous + (now - updated) * self._refill)
            retry = None if available >= 1 else max(1, ceil((1 - available) / self._refill))
            self._buckets[scope] = (available - 1 if retry is None else available, now)
            if len(self._buckets) > _MAX_BUCKETS:
                self._buckets.popitem(last=False)
            return retry


def _key_bytes(secret: SecretStr | None) -> bytes | None:
    return secret.get_secret_value().strip().encode("utf-8") if secret is not None else None


def _provided_key(request: Request) -> bytes:
    supplied = request.headers.get("X-API-Key")
    if supplied is None:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        supplied = token if scheme.casefold() == "bearer" else ""
    return supplied.strip().encode("utf-8")


class AccessControl:
    def __init__(self, settings: ApiSettings) -> None:
        self._api_key = _key_bytes(settings.api_key)
        self._admin_key = _key_bytes(settings.admin_api_key) or self._api_key
        self._buckets = TokenBuckets(settings.rate_limit_rpm) if settings.rate_limit_rpm else None

    def check(self, request: Request, *, admin: bool, rate_limit: bool = True) -> None:
        expected = self._admin_key if admin else self._api_key
        if expected is not None:
            provided = _provided_key(request)
            if not compare_digest(provided, expected):
                if admin and self._api_key is not None and compare_digest(provided, self._api_key):
                    raise HTTPException(403, "Administrator credentials are required.")
                raise HTTPException(
                    401,
                    "Valid API credentials are required.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        if rate_limit and self._buckets is not None:
            peer = request.client.host if request.client else "unknown"
            retry = self._buckets.retry_after((peer, admin))
            if retry is not None:
                raise HTTPException(
                    429, "Request rate limit exceeded.", headers={"Retry-After": str(retry)}
                )


class ProtectedRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()
        if not self.path.startswith("/v1/") and self.path != "/metrics":
            return handler
        is_metrics = self.path == "/metrics"
        is_admin = is_metrics or self.path == "/v1/ingest"

        async def authorize(request: Request) -> Response:
            request.app.state.access_control.check(
                request, admin=is_admin, rate_limit=not is_metrics
            )
            return await handler(request)

        return authorize
