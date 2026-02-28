"""ARWrench AI Enrichment API — FastAPI entry point."""
from __future__ import annotations

import time
import uuid
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
