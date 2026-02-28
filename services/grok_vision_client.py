"""Grok Vision API wrapper — image analysis calls go through here.

Uses grok-2-vision-1212, the only xAI model with multimodal (vision) support.
Follows the OpenAI-compatible chat completions API with image_url content parts.
"""
from __future__ import annotations

import base64
import os
import time

import httpx

from logger_config import logger
from log_utils import log_api_call

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_VISION_MODEL = os.getenv("GROK_VISION_MODEL", "grok-2-vision-1212")

_SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _get_api_key() -> str:
    key = os.getenv("GROK_API_KEY")
    if not key:
        raise ValueError("GROK_API_KEY environment variable not set")
    return key


async def _fetch_image_as_base64(url: str) -> tuple[str, str]:
    """Download an image URL and return (base64_data, mime_type)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "image/jpeg")
    mime_type = content_type.split(";")[0].strip()
    if mime_type not in _SUPPORTED_MIME_TYPES:
        mime_type = "image/jpeg"

    b64 = base64.b64encode(response.content).decode("utf-8")
    return b64, mime_type


async def analyze_image(image_url: str, prompt: str) -> str:
    """Call Grok Vision with an image URL + text prompt.

    Downloads the image, encodes it as base64, and sends it alongside the
    prompt using the OpenAI-compatible multimodal content format.

    Returns the raw text content of the first response candidate.
    Raises httpx.HTTPStatusError on non-2xx responses.
    Raises ValueError when GROK_API_KEY is not configured.
    """
    api_key = _get_api_key()
    start_time = time.time()

    b64_data, mime_type = await _fetch_image_as_base64(image_url)

    payload = {
        "model": GROK_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                    },
                ],
            }
        ],
        "temperature": 0.4,
        "max_tokens": 1024,
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
                    service="grok_vision",
                    endpoint=GROK_API_URL,
                    request_body={"model": GROK_VISION_MODEL, "image_url": image_url[:60]},
                    response_status=response.status_code,
                    duration_ms=duration_ms,
                    error=response.text[:200],
                )
                response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            log_api_call(
                service="grok_vision",
                endpoint=GROK_API_URL,
                request_body={"model": GROK_VISION_MODEL, "image_url": image_url[:60]},
                response_status=response.status_code,
                response_body={"response_length": len(content)},
                duration_ms=duration_ms,
            )

            return content

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "Grok Vision API call failed: %s",
            type(e).__name__,
            extra={"duration_ms": duration_ms, "error": str(e)},
        )
        raise
