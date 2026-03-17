"""Asset classification endpoints.

/classify: Combines vision result + text description → structured asset specs.
/classify/refine: Takes initial classification + user Q&A answers → refined specs.
"""
from __future__ import annotations

import os
import httpx
from fastapi import APIRouter, HTTPException

from logger_config import logger, LogContext
from log_utils import log_step, log_result
from models import ClassifyRequest, ClassifyResponse, RefineRequest, RefineResponse
from services.grok_client import parse_json_response
from services.model_router import call_llm, get_model_config, route_classify
from services.rag_service import retrieve_relevant_chunks

router = APIRouter()

APP_NAME = os.getenv("APP_NAME", "AR Wrench")

# Skill level → persona directive, mirroring the enrich endpoint (1-10 user knowledge scale)

SKILL_LEVEL_PREFIXES: dict[int, str] = {
    1: "The user has NO mechanical knowledge. Use simple everyday language and analogies. No DIY steps — recommend a professional for anything beyond a visual check.",
    2: "The user has minimal experience. Plain language only. Brief technical terms must be explained. Recommend professionals for anything beyond fluid checks or wiper blades.",
    3: "Basic DIYer (oil changes, air filters). Give 2–3 ranked causes. Simple step-by-step for easy fixes. Recommend a shop for anything requiring disassembly. Basic hand tools assumed.",
    4: "Comfortable basic DIYer who follows guides. Clear technical language. Step-by-step for moderate jobs. List specific tools needed. Flag tasks at the edge of this skill level.",
    5: "Intermediate DIYer, confident with common repairs. Standard technical language. Full differential diagnosis (3–4 causes). Complete procedures with torque and fluid specs. Standard toolset assumed.",
    6: "Solid hands-on experience. Full technical language. Ranked differential with reasoning. Precise procedures: part numbers, torque specs, clearances. Scan tool available. Brief targeted safety notes only.",
    7: "Advanced home mechanic. Professional language. Thorough differential including cascading faults. OBD-II codes, sensor test values, resistance specs. Mention TSBs. Full workshop assumed.",
    8: "Advanced mechanic, full powertrain and electrical diagnostics. Direct and precise. Lead with most probable root cause. Fault codes, live data PIDs, component pass/fail values. OEM vs aftermarket trade-offs.",
    9: "Professional technician. OEM terminology. Ranked fault tree with diagnostic procedure. DTC codes, freeze frame, live data PIDs. OEM repair procedures: torque sequences, calibration steps, TSBs and recall numbers.",
    10: "Master technician / automotive engineer. Peer-level response. Fastest path to root cause confirmation. Full fault tree, system interaction effects, OEM design limitations. Part numbers, supersession notes, relearn sequences.",
}

GROK_LLM_TEMPERATURE: dict[int, float] = {
    1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.55,
    6: 0.6, 7: 0.65, 8: 0.7, 9: 0.75, 10: 0.8,
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_CLASSIFY_SCHEMA = """
IMPORTANT: Respond ONLY in this exact JSON format — no extra text:
{
  "assetSpecs": {
    "type": "<free-form asset type, e.g. car, motorcycle, HVAC unit, washing machine, generator, CNC machine, drone — or unknown>",
    "make": "...",
    "model": "...",
    "year": <integer or null>,
    "mileage": <integer or null>
  },
  "condition": "good | maintenance_needed | repair_needed | critical",
  "confidence": <integer 0-100>,
  "recommendations": ["action 1", "action 2"],
  "reminders": [
    {"title": "...", "dueDate": "YYYY-MM-DD or null"}
  ]
}"""

_REFINE_SCHEMA = """
IMPORTANT: Respond ONLY in this exact JSON format — no extra text:
{
  "assetSpecs": {
    "type": "<free-form asset type, e.g. car, motorcycle, HVAC unit, washing machine, generator, CNC machine, drone — or unknown>",
    "make": "...",
    "model": "...",
    "year": <integer or null>,
    "mileage": <integer or null>
  },
  "condition": "good | maintenance_needed | repair_needed | critical",
  "confidence": <integer 0-100>,
  "refined_recommendations": ["action 1", "action 2"],
  "reminders": [
    {"title": "...", "due": "YYYY-MM-DD or null"}
  ]
}"""


@router.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    LogContext.set(asset_id=req.assetId)

    try:
        log_step("classify_started", asset_id=req.assetId)

        # Handle empty text gracefully — skip RAG, use fallback message
        text_desc = (req.textDescription or "").strip()
        rag_section = ""

        if text_desc:
            rag_context = await retrieve_relevant_chunks(text_desc, req.assetId)
            rag_section = (
                f"\nRelevant documentation for this asset:\n{rag_context}\n"
                if rag_context
                else ""
            )

        system_prompt = (
            f"You are {APP_NAME}, an asset classification and maintenance expert.\n"
            "You classify any physical asset — vehicles, equipment, appliances, "
            "tools, or machinery — and provide maintenance recommendations "
            "based on visual analysis and user description.\n"
            "If the input is a greeting or unrelated to an asset, "
            "return all string fields as \"unknown\" and all number fields as null.\n"
            f"{rag_section}"
            f"{_CLASSIFY_SCHEMA}"
        )

        vision_text = req.visionResult or "No image analysis available."
        user_message = (
            f"Visual analysis: {vision_text}\n"
            f"User description: {text_desc if text_desc else '(No text description provided)'}"
        )

        model_cfg = get_model_config(route_classify())
        raw = await call_llm(
            model_cfg,
            system_prompt,
            [{"role": "user", "content": user_message}],
            temperature=0.7,
        )
        parsed = parse_json_response(raw)

        if parsed.get("_parse_failed"):
            logger.warning("Classify parse failed for asset %s", req.assetId)
            raise HTTPException(status_code=500, detail="Failed to parse classification")

        return ClassifyResponse(
            assetSpecs=parsed.get("assetSpecs", {}),
            condition=parsed.get("condition", "unknown"),
            confidence=max(0, min(100, int(parsed.get("confidence") or 50))),
            recommendations=parsed.get("recommendations", []),
            reminders=parsed.get("reminders", []),
        )

    except httpx.HTTPStatusError as exc:
        logger.error("Grok API HTTP error in /classify: %s", exc.response.status_code)
        raise HTTPException(status_code=500, detail="Grok API error")
    except httpx.TimeoutException:
        logger.error("Grok timeout in /classify for asset %s", req.assetId)
        raise HTTPException(status_code=500, detail="Grok API timeout")
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Server configuration error")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /classify: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Internal server error")


# POST /classify/refine — Refinement endpoint

_SELF_CRITIQUE_ADDENDUM = """
Your previous confidence was {score}/100.
Re-examine your classification critically before finalising:
- Did the user's answers change anything significant?
- Are there alternative interpretations you dismissed too quickly?
- Is the confidence score still justified?

Provide a revised response in the same JSON format."""


@router.post("/classify/refine", response_model=RefineResponse)
async def classify_refine(req: RefineRequest):
    asset_id = req.initialClassification.get("assetSpecs", {}).get("assetId", "unknown")
    LogContext.set(asset_id=asset_id)

    try:
        log_step("refine_started", initial_confidence=req.initialClassification.get("confidence", 0))

        temp_prefix = SKILL_LEVEL_PREFIXES.get(req.skillLevel, SKILL_LEVEL_PREFIXES[5])
        grok_temp = GROK_LLM_TEMPERATURE.get(req.skillLevel, 0.55)

        initial = req.initialClassification
        initial_summary = (
            f"Initial asset specs: {initial.get('assetSpecs', {})}\n"
            f"Initial condition: {initial.get('condition', 'unknown')}\n"
            f"Initial confidence: {initial.get('confidence', 0)}/100"
        )

        qa_lines = "\n".join(f"Q: {ua.q}\nA: {ua.a}" for ua in req.userAnswers)

        system_prompt = (
            f"You are {APP_NAME}, an asset classification and maintenance expert.\n"
            f"{temp_prefix}\n\n"
            f"{initial_summary}\n\n"
            "The user has answered follow-up questions. "
            "Refine your classification based on the new information.\n"
            f"{_REFINE_SCHEMA}"
        )

        user_message = f"Follow-up answers from user:\n{qa_lines}"

        # First pass
        model_cfg = get_model_config(route_classify())
        raw = await call_llm(
            model_cfg,
            system_prompt,
            [{"role": "user", "content": user_message}],
            temperature=grok_temp,
        )
        parsed = parse_json_response(raw)

        if parsed.get("_parse_failed"):
            logger.warning("Refine first-pass parse failed")
            raise HTTPException(status_code=500, detail="Failed to parse refined classification")

        confidence = max(0, min(100, int(parsed.get("confidence") or 50)))

        # Dual-pass validation (Milestone 3): if confidence < 50, self-critique
        if confidence < 50:
            logger.info("Confidence %d < 50 — triggering self-critique pass", confidence)
            critique_prompt = (
                system_prompt
                + "\n\n"
                + _SELF_CRITIQUE_ADDENDUM.format(score=confidence)
            )
            raw2 = await call_llm(
                model_cfg,
                critique_prompt,
                [{"role": "user", "content": user_message}],
                temperature=grok_temp,
            )
            parsed2 = parse_json_response(raw2)
            if not parsed2.get("_parse_failed"):
                parsed = parsed2
                confidence = max(0, min(100, int(parsed.get("confidence") or confidence)))
                logger.info("After self-critique confidence: %d", confidence)

        return RefineResponse(
            assetSpecs=parsed.get("assetSpecs", {}),
            condition=parsed.get("condition", "unknown"),
            confidence=confidence,
            reminders=parsed.get("reminders", []),
            refined_recommendations=parsed.get("refined_recommendations", []),
        )

    except httpx.HTTPStatusError as exc:
        logger.error("Grok API HTTP error in /classify/refine: %s", exc.response.status_code)
        raise HTTPException(status_code=500, detail="Grok API error")
    except httpx.TimeoutException:
        logger.error("Grok timeout in /classify/refine")
        raise HTTPException(status_code=500, detail="Grok API timeout")
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Server configuration error")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /classify/refine: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Internal server error")
