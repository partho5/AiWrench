"""
Structured logging utilities for tracking operation breakdowns.
Provides decorators and context managers to log what worked, how, and timing.

Example:
    @log_operation("enrich_symptom")
    async def enrich(symptom: str):
        log_step("symptom_received", symptom=symptom)
        # ... do work ...
        log_result("success", result_count=5)
"""

import functools
import json
import logging
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Optional

from logger_config import logger


def log_step(
    name: str,
    status: str = "running",
    **metadata,
) -> None:
    """Log a single step in an operation.

    Args:
        name: Step identifier (e.g., "pdf_ingestion_started")
        status: "running", "success", "warning", "error"
        **metadata: Additional context (asset_id, chunk_count, etc.)
    """
    msg = f"[{status.upper()}] {name}"

    if status == "error":
        logger.error(msg, extra={"metadata": metadata})
    elif status == "warning":
        logger.warning(msg, extra={"metadata": metadata})
    else:
        logger.info(msg, extra={"metadata": metadata})


def log_result(
    name: str,
    success: bool = True,
    duration_ms: float = 0,
    **result_data,
) -> None:
    """Log the final result of an operation.

    Args:
        name: Operation name
        success: Whether operation succeeded
        duration_ms: How long it took
        **result_data: What was produced (e.g., result_count=5, confidence=0.95)
    """
    status = "✓ SUCCESS" if success else "✗ FAILED"
    msg = f"{status} | {name} | {duration_ms:.0f}ms"

    # Always log with extra data for structured analysis
    level = logging.INFO if success else logging.ERROR
    logger.log(
        level,
        msg,
        extra={
            "operation": name,
            "success": success,
            "duration_ms": duration_ms,
            "result": result_data,
        },
    )


def log_operation(operation_name: str) -> Callable:
    """Decorator to log async operation breakdown.

    Automatically logs:
      - Operation start
      - Duration
      - Success/failure
      - Any exception

    Example:
        @log_operation("classify_fault")
        async def classify(symptom: str) -> dict:
            return {"confidence": 0.95}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            operation_id = f"{operation_name}_{int(start_time * 1000) % 1000000}"

            logger.info(
                f"▶ START | {operation_name}",
                extra={"operation_id": operation_id, "operation": operation_name},
            )

            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"✓ DONE | {operation_name} | {duration_ms:.0f}ms",
                    extra={
                        "operation_id": operation_id,
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "success": True,
                    },
                )
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.exception(
                    f"✗ ERROR | {operation_name} | {duration_ms:.0f}ms | {type(e).__name__}",
                    extra={
                        "operation_id": operation_id,
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise

        return wrapper

    return decorator


@contextmanager
def log_context(name: str, **metadata):
    """Context manager for logging nested operations.

    Example:
        with log_context("rag_retrieval", asset_id="motor_1"):
            # ... retrieval logic ...
    """
    start_time = time.time()
    context_id = f"{name}_{int(start_time * 1000) % 1000000}"

    logger.info(
        f"→ ENTER | {name}",
        extra={"context_id": context_id, **metadata},
    )

    try:
        yield context_id
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.exception(
            f"← EXIT (ERROR) | {name} | {duration_ms:.0f}ms",
            extra={
                "context_id": context_id,
                "duration_ms": duration_ms,
                "error": str(e),
            },
        )
        raise
    else:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"← EXIT (OK) | {name} | {duration_ms:.0f}ms",
            extra={"context_id": context_id, "duration_ms": duration_ms},
        )


def log_api_call(
    service: str,
    endpoint: str,
    request_body: Optional[dict] = None,
    response_status: int = 200,
    response_body: Optional[dict] = None,
    duration_ms: float = 0,
    error: Optional[str] = None,
) -> None:
    """Log external API calls (Grok, Gemini, etc.) for monitoring.

    Args:
        service: API provider (grok, gemini, openai, chroma)
        endpoint: API endpoint called
        request_body: Request payload (sanitized)
        response_status: HTTP status code
        response_body: Response data (partial for large responses)
        duration_ms: Latency
        error: Error message if failed
    """
    success = 200 <= response_status < 300
    status_icon = "✓" if success else "✗"

    msg = f"{status_icon} {service.upper()} | {endpoint} | {response_status} | {duration_ms:.0f}ms"

    level = logging.INFO if success else logging.WARNING
    logger.log(
        level,
        msg,
        extra={
            "api_call": True,
            "service": service,
            "endpoint": endpoint,
            "status_code": response_status,
            "duration_ms": duration_ms,
            "error": error,
            "request_body": _sanitize(request_body),
            "response_body": _sanitize(response_body, max_len=500),
        },
    )


def _sanitize(data: Optional[dict], max_len: int = 1000) -> Optional[dict]:
    """Remove sensitive fields before logging."""
    if not data:
        return None

    sanitized = data.copy()
    for key in ["api_key", "password", "token", "secret"]:
        if key in sanitized:
            sanitized[key] = "***REDACTED***"

    # Truncate large values
    for key, value in sanitized.items():
        if isinstance(value, str) and len(value) > max_len:
            sanitized[key] = value[:max_len] + f"... (+{len(value) - max_len} chars)"

    return sanitized
