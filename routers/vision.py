"""Vision analysis endpoint.

Receives an image URL + optional asset context, calls Grok Vision,
and returns structured analysis + confidence.
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from logger_config import logger, LogContext
from log_utils import log_step, log_result
from models import StructuredFindings, VisionRequest, VisionResponse
from services.grok_vision_client import analyze_image
from services.grok_client import parse_json_response

router = APIRouter()


def _build_vision_prompt(req: VisionRequest) -> str:
    asset_info = ""
    if req.assetContext:
        parts = [
            req.assetContext.get("type", ""),
            req.assetContext.get("make", ""),
            req.assetContext.get("model", ""),
        ]
        desc = " ".join(p for p in parts if p).strip()
        if desc:
            asset_info = f" belonging to a {desc}"

    base = req.systemPrompt or f"Analyse this photo{asset_info}."

    return f"""{base}

Identify the part(s) visible and assess their condition.

IMPORTANT: Respond ONLY in this exact JSON format — no extra text:
{{
  "analysis": "Detailed description of what you see and your assessment...",
  "part": "Name of the identified part, or 'unknown'",
  "condition": "good | worn | damaged | critical | unknown",
  "severity": "none | minor | moderate | critical",
  "confidence": <integer 0-100>,
  "observations": ["observation 1", "observation 2"]
}}"""


@router.post("/vision", response_model=VisionResponse)
async def vision(req: VisionRequest):
    LogContext.set(image_url=req.imageUrl[:50] + "...")  # Truncate for logging

    try:
        log_step("vision_started", has_asset_context=bool(req.assetContext))

        prompt = _build_vision_prompt(req)
        raw = await analyze_image(req.imageUrl, prompt)

        if not raw:
            raise HTTPException(status_code=500, detail="Grok Vision returned empty response")

        parsed = parse_json_response(raw)

        if parsed.get("_parse_failed"):
            logger.warning("Grok Vision response parse failed for image %s", req.imageUrl)
            return VisionResponse(
                analysis=parsed.get("_raw", "Unable to analyse image."),
                confidence=30,
                structured_findings=StructuredFindings(),
            )

        confidence = max(0, min(100, int(parsed.get("confidence", 50))))

        return VisionResponse(
            analysis=parsed.get("analysis", "No analysis available."),
            confidence=confidence,
            structured_findings=StructuredFindings(
                part=parsed.get("part"),
                condition=parsed.get("condition"),
                severity=parsed.get("severity"),
                observations=parsed.get("observations", []),
            ),
        )

    except httpx.HTTPStatusError as exc:
        logger.error("Grok Vision API HTTP error: %s", exc.response.status_code)
        raise HTTPException(status_code=500, detail="Grok Vision API error")
    except httpx.TimeoutException:
        logger.error("Grok Vision API timeout for image %s", req.imageUrl)
        raise HTTPException(status_code=500, detail="Grok Vision API timeout")
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Server configuration error")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /vision: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Internal server error")
