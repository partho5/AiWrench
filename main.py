"""ARWrench AI Enrichment API — FastAPI entry point."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import defaultdict
from typing import Callable

from dotenv import load_dotenv

# Load .env before any router imports so os.getenv() picks up the keys
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Import new logging system
from logger_config import logger, LogContext
from log_utils import log_result

from routers import enrich, vision, classify

app = FastAPI(
    title="ARWrench AI Enrichment API",
    description=(
        "FastAPI enrichment layer for the ARWrench iOS app. "
        "Owns all Grok + Gemini calls; Convex passes data and saves results."
    ),
    version="1.0.0",
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing only; restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Logging Middleware

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing, status, and context."""

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        LogContext.set(request_id=request_id)

        start_time = time.time()
        method = request.method
        path = request.url.path

        # Log incoming request
        logger.info(
            f"→ {method} {path}",
            extra={"request_id": request_id, "method": method, "path": path},
        )

        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            status = response.status_code
            icon = "✓" if response.status_code < 400 else "✗"
            logger.info(
                f"{icon} {method} {path} | {status} | {duration_ms:.0f}ms",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status,
                    "duration_ms": duration_ms,
                },
            )
            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.exception(
                f"✗ {method} {path} | ERROR | {duration_ms:.0f}ms | {type(e).__name__}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "duration_ms": duration_ms,
                    "error": str(e),
                },
            )
            raise
        finally:
            LogContext.clear()


app.add_middleware(RequestLoggingMiddleware)


# Auth Middleware — shared secret between Convex and this API

_UNPROTECTED_PATHS = {"/health"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Require X-API-Token header on all routes except /health."""

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        if request.url.path in _UNPROTECTED_PATHS:
            return await call_next(request)

        expected = os.getenv("API_SECRET_TOKEN", "")
        if not expected:
            # Token not configured — allow through but warn (dev environment)
            logger.warning("API_SECRET_TOKEN not set — auth check skipped")
            return await call_next(request)

        token = request.headers.get("X-API-Token", "")
        if token != expected:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        return await call_next(request)


app.add_middleware(AuthMiddleware)


# IP Rate Limit Middleware — abuse prevention at the server boundary

class IPRateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window IP rate limiter. Limits are a server-level abuse guard only.
    Per-user quota enforcement is handled by the Convex/app layer before requests reach here."""

    def __init__(self, app, max_requests: int = 200, window_seconds: int = 60):
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        if request.url.path in _UNPROTECTED_PATHS:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self._window

        async with self._lock:
            self._hits[ip] = [t for t in self._hits[ip] if t > cutoff]
            if len(self._hits[ip]) >= self._max:
                oldest = min(self._hits[ip])
                retry_after = max(1, int(oldest + self._window - now) + 1)
                logger.warning(f"Rate limit exceeded for IP {ip}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Too many requests. Retry after {retry_after}s."},
                    headers={"Retry-After": str(retry_after)},
                )
            self._hits[ip].append(now)

        return await call_next(request)


_rate_limit_rpm = int(os.getenv("RATE_LIMIT_RPM", "200"))
app.add_middleware(IPRateLimitMiddleware, max_requests=_rate_limit_rpm, window_seconds=60)


# Global error handler

@app.exception_handler(Exception)
async def _global_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        f"Unhandled error on {request.url.path}: {type(exc).__name__}",
        extra={"path": request.url.path, "error_type": type(exc).__name__},
    )
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# Health check

@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "service": "ARWrench AI Enrichment API", "version": "1.0.0"}


# Routers
app.include_router(enrich.router, tags=["Enrich"])
app.include_router(vision.router, tags=["Vision"])
app.include_router(classify.router, tags=["Classify"])
