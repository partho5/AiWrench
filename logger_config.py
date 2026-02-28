"""
Production logging — JSON structured logs to /logs only.

Environment variables:
  - LOG_LEVEL : DEBUG, INFO, WARNING, ERROR  (default: INFO)
  - LOG_FILE  : path to log file             (default: logs/arwrench.log)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from pythonjsonlogger.json import JsonFormatter


def setup_logging() -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/arwrench.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.handlers.clear()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB per file
        backupCount=5,              # keep 5 rotated files → 60MB max
    )
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(
        JsonFormatter(
            fmt="%(timestamp)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d",
            timestamp=True,
        )
    )
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "urllib3", "chromadb", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


logger = setup_logging().getChild("arwrench")


class LogContext:
    """Request-scoped metadata injected into every log record."""

    _context: dict[str, str] = {}

    @classmethod
    def set(cls, **kwargs) -> None:
        cls._context.update(kwargs)

    @classmethod
    def get(cls, key: str, default: str = "") -> str:
        return cls._context.get(key, default)

    @classmethod
    def clear(cls) -> None:
        cls._context.clear()


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = LogContext.get("request_id", "")
        record.asset_id = LogContext.get("asset_id", "")
        return True


for handler in logger.root.handlers:
    handler.addFilter(ContextFilter())
