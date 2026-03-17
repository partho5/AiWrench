"""Anthropic Claude API wrapper — used for deep tier reasoning.

Grok Vision extracts image/PDF content as text upstream.
Claude receives that extracted text in the system prompt and does
all diagnostic reasoning from there.

Uses a module-level AsyncAnthropic client for connection reuse.
Initialised lazily on first call so import order with dotenv is safe.
"""
from __future__ import annotations

import os
import time
from typing import AsyncGenerator

import anthropic

from logger_config import logger
from log_utils import log_api_call

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        _client = anthropic.AsyncAnthropic(api_key=api_key)
    return _client


async def call_claude(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
    model: str = "claude-opus-4-6",
    max_tokens: int = 2048,
) -> str:
    """Non-streaming Claude call. Returns full response text."""
    client = _get_client()
    start_time = time.time()

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        )
        duration_ms = (time.time() - start_time) * 1000
        content = response.content[0].text if response.content else ""
        log_api_call(
            service="claude",
            endpoint=f"messages/{model}",
            request_body={"model": model, "messages_count": len(messages)},
            response_status=200,
            response_body={"token_usage": len(content)},
            duration_ms=duration_ms,
        )
        return content

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "Claude call failed: %s",
            type(e).__name__,
            extra={"duration_ms": duration_ms, "error": str(e)},
        )
        raise


async def call_claude_stream(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
    model: str = "claude-opus-4-6",
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """Streaming Claude call. Yields text chunks as they arrive."""
    client = _get_client()
    start_time = time.time()
    full_content = ""

    try:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for chunk in stream.text_stream:
                full_content += chunk
                yield chunk

        duration_ms = (time.time() - start_time) * 1000
        log_api_call(
            service="claude_stream",
            endpoint=f"messages/{model}",
            request_body={"model": model, "messages_count": len(messages)},
            response_body={"token_usage": len(full_content)},
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "Claude stream failed: %s",
            type(e).__name__,
            extra={"duration_ms": duration_ms, "error": str(e)},
        )
        raise
