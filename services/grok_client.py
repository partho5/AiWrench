"""Grok API wrapper — all Grok calls go through here."""
from __future__ import annotations

import json
import os
import time

import httpx

from logger_config import logger
from log_utils import log_api_call

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")


def _get_api_key() -> str:
    key = os.getenv("GROK_API_KEY")
    if not key:
        raise ValueError("GROK_API_KEY environment variable not set")
    return key


async def call_grok(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Call Grok and return the raw text response content."""
    api_key = _get_api_key()
    start_time = time.time()

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                GROK_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            duration_ms = (time.time() - start_time) * 1000

            if not response.is_success:
                log_api_call(
                    service="grok",
                    endpoint=GROK_API_URL,
                    request_body={"model": GROK_MODEL, "messages_count": len(messages)},
                    response_status=response.status_code,
                    duration_ms=duration_ms,
                    error=response.text[:200],
                )
                response.raise_for_status()

            data = response.json()
            log_api_call(
                service="grok",
                endpoint=GROK_API_URL,
                request_body={"model": GROK_MODEL, "messages_count": len(messages)},
                response_status=response.status_code,
                response_body={"token_usage": len(data.get("choices", [{}])[0].get("message", {}).get("content", ""))},
                duration_ms=duration_ms,
            )

            return data["choices"][0]["message"]["content"]

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Grok API call failed: {type(e).__name__}", extra={"duration_ms": duration_ms, "error": str(e)})
        raise


def parse_json_response(raw: str) -> dict:
    """Extract JSON from a Grok response string.

    Tries: direct parse → ```json block → ``` block → braces.
    Returns {"_raw": raw, "_parse_failed": True} if nothing works.
    """
    text = raw.strip()

    try:
        result = json.loads(text)
        logger.debug("JSON parsed directly from Grok response")
        return result
    except json.JSONDecodeError:
        pass

    if "```json" in text:
        try:
            start = text.index("```json") + 7
            end = text.index("```", start)
            result = json.loads(text[start:end].strip())
            logger.debug("JSON extracted from ```json block")
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    if "```" in text:
        try:
            start = text.index("```") + 3
            end = text.index("```", start)
            result = json.loads(text[start:end].strip())
            logger.debug("JSON extracted from ``` block")
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            result = json.loads(text[brace_start : brace_end + 1])
            logger.debug("JSON extracted from braces")
            return result
        except json.JSONDecodeError:
            pass

    logger.warning(
        "JSON parse failed on Grok response",
        extra={"response_preview": raw[:200], "response_length": len(raw)},
    )
    return {"_raw": raw, "_parse_failed": True}


async def call_grok_stream(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """Stream Grok response as server-sent events.

    Yields chunks of text as they arrive from Grok.
    """
    api_key = _get_api_key()
    start_time = time.time()

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                GROK_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                if not response.is_success:
                    error_text = await response.aread()
                    logger.error(
                        f"Grok stream API error: {response.status_code}",
                        extra={"status_code": response.status_code, "error": error_text[:200].decode('utf-8', errors='ignore')},
                    )
                    response.raise_for_status()

                full_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_text = line[6:]  # Remove "data: " prefix
                        if chunk_text == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(chunk_text)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    full_content += content
                                    yield content
                        except json.JSONDecodeError:
                            pass

                duration_ms = (time.time() - start_time) * 1000
                log_api_call(
                    service="grok_stream",
                    endpoint=GROK_API_URL,
                    request_body={"model": GROK_MODEL, "messages_count": len(messages)},
                    response_body={"token_usage": len(full_content)},
                    duration_ms=duration_ms,
                )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Grok stream call failed: {type(e).__name__}", extra={"duration_ms": duration_ms, "error": str(e)})
        raise
