"""Gemini Vision API wrapper — all Gemini calls go through here."""
from __future__ import annotations

import base64
import logging
import os

import httpx

logger = logging.getLogger(__name__)

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-1.5-flash:generateContent"
)


def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return key


async def _fetch_image_as_base64(url: str) -> tuple[str, str]:
    """Download an image URL and return (base64_data, mime_type)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "image/jpeg")
    # Normalise mime type — strip parameters like "; charset=..."
    mime_type = content_type.split(";")[0].strip()
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        mime_type = "image/jpeg"

    b64 = base64.b64encode(response.content).decode("utf-8")
    return b64, mime_type


async def analyze_image(image_url: str, prompt: str) -> str:
    """
    Call Gemini Vision with an image URL + text prompt.
    Returns the raw text content of the first candidate.
    """
    api_key = _get_api_key()

    b64_data, mime_type = await _fetch_image_as_base64(image_url)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": b64_data,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{GEMINI_API_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        logger.warning("Gemini returned no candidates")
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""

    return parts[0].get("text", "")
