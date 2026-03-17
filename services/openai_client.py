"""OpenAI API wrapper — used for first_message tier (fast, non-reasoning).

Uses a module-level AsyncOpenAI client for connection reuse across requests.
The client is initialised lazily on first call so import order with dotenv is safe.
"""
from __future__ import annotations

import os
import time
from typing import AsyncGenerator

from openai import AsyncOpenAI

from logger_config import logger
from log_utils import log_api_call

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def call_openai(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
    model: str = "gpt-4o-mini",
    max_tokens: int = 512,
) -> str:
    """Non-streaming OpenAI call. Returns full response text."""
    client = _get_client()
    start_time = time.time()
    all_messages = [{"role": "system", "content": system_prompt}, *messages]

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        duration_ms = (time.time() - start_time) * 1000
        content = response.choices[0].message.content or ""
        log_api_call(
            service="openai",
            endpoint=f"chat/completions/{model}",
            request_body={"model": model, "messages_count": len(messages)},
            response_status=200,
            response_body={"token_usage": len(content)},
            duration_ms=duration_ms,
        )
        return content

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "OpenAI call failed: %s",
            type(e).__name__,
            extra={"duration_ms": duration_ms, "error": str(e)},
        )
        raise


async def call_openai_stream(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
    model: str = "gpt-4o-mini",
    max_tokens: int = 512,
) -> AsyncGenerator[str, None]:
    """Streaming OpenAI call. Yields text chunks as they arrive."""
    client = _get_client()
    start_time = time.time()
    all_messages = [{"role": "system", "content": system_prompt}, *messages]
    full_content = ""

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=all_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_content += delta.content
                yield delta.content

        duration_ms = (time.time() - start_time) * 1000
        log_api_call(
            service="openai_stream",
            endpoint=f"chat/completions/{model}",
            request_body={"model": model, "messages_count": len(messages)},
            response_body={"token_usage": len(full_content)},
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "OpenAI stream failed: %s",
            type(e).__name__,
            extra={"duration_ms": duration_ms, "error": str(e)},
        )
        raise
