"""POST /enrich — Asset enrichment endpoint.

Receives chat history + asset context from Convex, enriches the prompt,
calls Grok, scores confidence, and returns a structured final answer.
"""
from __future__ import annotations

import json
import os
import re
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from logger_config import logger, LogContext
from log_utils import log_step, log_result
from models import EnrichRequest, EnrichResponse
from services.grok_client import parse_json_response
from services.grok_vision_client import analyze_image
from services.model_router import call_llm, call_llm_stream, get_model_config, route_enrich

router = APIRouter()

# AI assistant name — users may address the AI by this name; it must not be
# misidentified as an asset or equipment type.
AI_NAME = os.getenv("AI_NAME", "Wrench")

# Skill level → persona directive (1-10 maps user tech knowledge to response calibration)
#
# Each level controls 6 dimensions simultaneously:
#   language · diagnosis depth · solution quality · tools assumed · safety verbosity · data density
#
# Tiers:
#   1-2  Novice          — no mechanical background, safety-first, always defer to professional
#   3-4  Basic DIYer     — basic maintenance capable, needs step-by-step guidance
#   5-6  Comfortable DIY — standard repairs, understands procedures, has common tools
#   7-8  Advanced DIYer  — home mechanic, reads OBD codes, full workshop capable
#   9-10 Professional    — certified technician, OEM specs, advanced diagnostics

SKILL_LEVEL_PREFIXES: dict[int, str] = {
    1: (
        "The user has NO technical knowledge. Use simple everyday language and relatable analogies only — "
        "never use technical terms without explaining them in plain words. "
        "Give ONE most likely cause. Do not provide hands-on repair or disassembly steps. "
        "For anything beyond a basic visual check or simple setting change, firmly recommend a professional. "
        "Keep safety warnings prominent and friendly. Assume the user has no tools and no prior experience. "
        "If you ask for a photo, tell them exactly how to take it: what to point the camera at, how close, and in good lighting."
    ),
    2: (
        "The user has minimal technical experience. Use plain language; briefly explain any term you must use. "
        "Limit self-fix suggestions to very simple actions (restarting, checking connections, replacing a fuse, basic settings). "
        "For anything more involved, recommend a professional or service center. "
        "Give one or two likely causes. Mention typical repair costs so they can make an informed decision. "
        "If you ask for a photo, describe clearly what part to find, what angle to use, and that lighting matters."
    ),
    3: (
        "The user is comfortable with basic troubleshooting (simple replacements, following guides, basic settings). "
        "Use plain language with some technical terms where needed. "
        "Give 2–3 possible causes ranked by likelihood. "
        "Provide simple step-by-step guidance for straightforward fixes. "
        "Flag anything requiring disassembly or specialty tools — recommend a professional for those. "
        "Assume the user has basic tools (screwdrivers, pliers) and can follow an online guide. "
        "If you ask for a photo, name the part and specify the angle (e.g. 'close-up of the back of the panel')."
    ),
    4: (
        "The user is comfortable following detailed repair or troubleshooting guides. "
        "Use clear technical language. Give 2–3 ranked causes with brief reasoning. "
        "Provide step-by-step procedures for moderate jobs. List required tools or equipment specifically. "
        "Include safety reminders for critical steps. "
        "Note when a task is at the edge of what this skill level should attempt. "
        "If you ask for a photo, name the component and the specific view needed."
    ),
    5: (
        "The user is an intermediate technician confident with most common repairs and diagnostics. "
        "Use standard technical language freely. Provide a full differential diagnosis (3–4 causes, ranked). "
        "Give complete procedures including component names and sequence of steps. "
        "Include relevant specs (voltages, settings, tolerances) where applicable. "
        "Assume a standard toolkit and basic diagnostic equipment (multimeter, etc.). "
        "Balance DIY and professional options honestly based on complexity and risk. "
        "If you ask for a photo, specify the component name and exact view required."
    ),
    6: (
        "The user has solid hands-on technical experience and understands systems well. "
        "Use full technical language. Provide ranked differential diagnosis with reasoning for each cause. "
        "Give precise procedures: part or component identifiers where known, specs, tolerances. "
        "Point out common failure patterns and related components to inspect while the job is open. "
        "Safety notes should be brief and targeted, not repetitive. "
        "Photo requests: component name and view only — no guidance needed on how to take it."
    ),
    7: (
        "The user is an advanced technician capable of complex diagnostics and repairs. "
        "Use professional technical language. Provide a thorough differential including secondary and cascading faults. "
        "Include error codes or diagnostic indicators likely to be present, specific test values, and pass/fail specs. "
        "Give complete procedures as a professional would execute them. "
        "Mention known model-specific issues or published service bulletins if applicable. "
        "Do not over-explain fundamentals — the user understands systems deeply. "
        "Photo requests: part name and view only."
    ),
    8: (
        "The user is an advanced technician comfortable with full system and electrical diagnostics. "
        "Be direct and precise. Lead with the most probable root cause supported by diagnostic logic. "
        "Include specific fault codes or indicators, live parameters to monitor, and component test procedures with pass/fail values. "
        "Give repair specs: tolerances, part identifiers, OEM vs aftermarket trade-offs. "
        "Mention related components worth inspecting or replacing preventively. "
        "Assume all standard and specialty tools available. "
        "Photo requests: component and view only — one line."
    ),
    9: (
        "The user is a professional-level technician. Use precise technical terminology throughout. "
        "Lead with a ranked fault tree and the procedure to confirm root cause. "
        "Include all relevant diagnostic codes, test parameters, and live data to monitor. "
        "Provide complete standard repair procedures: sequences, clearance specs, calibration steps. "
        "Reference known pattern failures, service bulletins, and recall numbers where applicable. "
        "List required specialty tools or programming/calibration requirements. "
        "Be comprehensive — this user needs depth, not hand-holding. "
        "Photo requests: component identifier only."
    ),
    10: (
        "The user is a master technician or domain engineer. Respond as a peer. "
        "Lead with the most precise diagnostic hypothesis and the fastest path to confirm it. "
        "Include full fault tree analysis, system interaction effects, and known design limitations or revisions. "
        "Provide complete specs: tolerances, programming procedures, relearn or calibration sequences. "
        "Reference applicable service bulletins, preliminary information, and any field fixes. "
        "Include part identifiers, supersession notes, and OEM vs aftermarket considerations. "
        "Assume mastery — omit basic explanations entirely. Prioritize diagnostic precision. "
        "Photo requests: component only."
    ),
}

# Grok LLM temperature per skill level.
# Lower = more deterministic (better for novice — one clear answer).
# Higher = allows more nuanced technical exploration (better for expert differential diagnosis).
GROK_LLM_TEMPERATURE: dict[int, float] = {
    1: 0.2,
    2: 0.3,
    3: 0.4,
    4: 0.5,
    5: 0.55,
    6: 0.6,
    7: 0.65,
    8: 0.7,
    9: 0.75,
    10: 0.8,
}



def _format_service_history(service_history: list) -> str:
    if not service_history:
        return "No service history on record."
    lines = []
    for record in service_history:
        if isinstance(record, dict):
            date = record.get("date", "Unknown date")
            service = record.get("service", "Unknown service")
            notes = record.get("notes", "")
            line = f"- {date}: {service}"
            if notes:
                line += f" ({notes})"
            lines.append(line)
        else:
            lines.append(f"- {record}")
    return "\n".join(lines)


def summarize_messages(messages: list) -> tuple[list[dict], str]:
    """
    Split messages into (recent_last_5, older_summary_text).
    The older summary is injected into the system prompt so Grok has full context
    without exceeding token limits on long threads.
    """
    if len(messages) <= 5:
        return [{"role": m.role, "content": m.content} for m in messages], ""

    older = messages[:-5]
    recent = messages[-5:]

    parts = []
    for m in older:
        label = "User" if m.role == "user" else "Mechanic AI"
        snippet = m.content[:400] + ("…" if len(m.content) > 400 else "")
        parts.append(f"{label}: {snippet}")

    summary = "\n".join(parts)
    return [{"role": m.role, "content": m.content} for m in recent], summary


# ---------------------------------------------------------------------------
# Stream separator detection — three-layer defense
# ---------------------------------------------------------------------------
# Layer 1: strict separator (nominal format with newlines)
#   Handles: \n---META---\n  \n---\n  \n--- meta ---\n  etc.
_SEP_RE = re.compile(r"\n-{2,}[- \t]*(?:META)?[- \t]*\n", re.IGNORECASE)

# Layer 1b: lenient separator (catches missing leading/trailing newlines)
#   Handles: ---META---  (no newlines)  ...text---META---{...json...}
_SEP_LENIENT_RE = re.compile(r"-{2,}[- \t]*(?:META)?[- \t]*(?=\s*\{)", re.IGNORECASE)

_HOLDBACK = 60  # chars held back during streaming to catch a separator that spans chunks

# Layer 2: trailing JSON heuristic.  If the LLM omits the separator entirely,
# strip any compact JSON blob at the tail of the answer that contains our
# known metadata keys (symptom_match is always present).
_TRAILING_META_RE = re.compile(
    r"\s*\{[^{}]*\"symptom_match\"\s*:\s*\d+[^{}]*\}\s*$",
    re.DOTALL,
)


def _parse_meta_str(meta_str: str) -> dict:
    """Parse the metadata JSON string; return {} on any failure."""
    try:
        return json.loads(meta_str.strip()) if meta_str else {}
    except Exception:
        return {}


def build_stream_prompt(req: EnrichRequest, conversation_summary: str = "", vision_analysis: str | None = None) -> str:
    """Prompt for /enrich/stream — answer first as plain text, metadata JSON after separator."""
    ctx = req.assetContext

    parts = []
    if ctx.year:
        parts.append(str(ctx.year))
    if ctx.make:
        parts.append(ctx.make)
    if ctx.model:
        parts.append(ctx.model)
    asset_desc = " ".join(parts) if parts else (ctx.type or "")
    if ctx.mileage:
        asset_desc += f" ({ctx.mileage:,} miles)"
    elif ctx.type and ctx.type != "vehicle" and asset_desc:
        asset_desc = f"{ctx.type} — {asset_desc}"

    toolbox_text = ", ".join(req.toolbox) if req.toolbox else "No tools specified"
    history_text = _format_service_history(req.serviceHistory)
    temp_prefix = SKILL_LEVEL_PREFIXES.get(req.skillLevel, SKILL_LEVEL_PREFIXES[5])
    summary_section = (
        f"\nEarlier conversation (summarised):\n{conversation_summary}\n"
        if conversation_summary else ""
    )
    vision_section = (
        f"\nPhoto provided by user — visual analysis:\n{vision_analysis}\n"
        if vision_analysis else ""
    )

    if asset_desc:
        identity = (
            f"You are {AI_NAME}, a professional diagnostic AI assistant troubleshooting a {asset_desc}. "
            f"Your name is {AI_NAME} — if the user addresses you by name, respond naturally; never treat your own name as an asset or equipment."
        )
    else:
        identity = (
            f"You are {AI_NAME}, a professional diagnostic AI assistant. "
            "No specific device or equipment has been identified yet. "
            "Help the user identify what they need help with, ask clarifying questions if needed, "
            f"then diagnose the issue. Your name is {AI_NAME} — if the user addresses you by name, respond naturally; never treat your own name as an asset or equipment."
        )

    return f"""{identity}

Asset service history:
{history_text}

Owner's available tools: {toolbox_text}
{summary_section}{vision_section}
{temp_prefix}

DIAGNOSTIC ROLE: You are the lead on this diagnosis. Your job is to reach a confirmed root cause and solution as efficiently as possible — not to answer questions in isolation. Every turn must move the conversation measurably closer to a fix.

Each turn, follow this internal process:
1. Form or update your leading hypothesis (the single most likely cause given everything known so far).
2. Identify the one piece of information that would most confirm or rule it out.
3. Decide: is that best answered by text (a symptom detail, a test result) or by seeing the part (a photo)?
4. Ask for exactly that — and nothing else.

Progress through diagnostic phases deliberately:
- GATHERING: symptoms still unclear → ask the one question that would sharpen them most.
- HYPOTHESISING: symptoms clear → commit to one leading cause, briefly say why.
- CONFIRMING: ask for the specific test or photo that confirms the cause.
- SOLVING: cause confirmed → give the fix, as concise as the complexity allows.

PHOTO REQUESTS: When a visual would give more diagnostic information than any text answer, ask for a photo. Be specific — name the exact part, area, and angle (e.g. "Can you send a close-up of the key blade next to the lock cylinder?" not "Can you send a photo?"). Only ask when it would genuinely change what you diagnose or recommend.

TOOLBOX: The user has access to these tools — use them actively. When a hands-on test would confirm your hypothesis and the user has the right tool, suggest the specific test with a clear pass/fail value (e.g. "With a multimeter, measure voltage across the battery terminals — 12.6V or above is healthy, below 12V means the battery is the likely cause").

CONVERSATION RULES:
- Default reply length: 2–4 sentences. Short and focused.
- State ONE most likely cause — never list multiple possibilities at once.
- End with ONE question or ONE photo request — whichever gives the most diagnostic value. Never both.
- Skip the question entirely if the user has already stated or confirmed the cause — give the next actionable step instead.
- Reveal information gradually as the user confirms or rules out symptoms — not all at once.
- A longer answer is only justified when ALL of the following are true:
    (a) root cause confirmed through prior turns, AND
    (b) user is ready for the solution, AND
    (c) a shorter answer would genuinely omit critical safety or procedural information.
  Even then: no padding, no repetition.

CRITICAL: Output format — exactly three parts, strictly separated:

PART 1: Your full diagnostic answer as plain readable text.

---META---
PART 2: ONLY this JSON (one line, no extra formatting):
{{"symptom_match":<0-100>,"history_alignment":<0-100>,"specificity":<0-100>,"safety_flag":<true/false>,"safety_explanation":"<text>","abstain":<true/false>,"affiliate_links":[...]}}

METADATA GUIDE:
- symptom_match (0-100): Alignment with known failure patterns
- history_alignment (0-100): Service history relevance to diagnosis
- specificity (0-100): Precision and actionability (high = specific solution, low = generic guess)
- safety_flag: true for electrical/gas/structural/personal safety hazards
- safety_explanation: Brief reason if safety_flag is true, empty string if false
- abstain: true only if issue is outside technical scope or too dangerous to diagnose
- affiliate_links: array of purchasable items directly relevant to THIS diagnostic step. Include only when a specific part, fluid, consumable, or tool purchase would help confirm or resolve the issue. Use [] when nothing specific is needed right now. Each item format: {{"type":"affiliate_link","label":"<concise buy label>","data":"{{\\"category\\":\\"<broad category>\\",\\"name\\":\\"<specific product>\\"}}"}}.
  Category examples: Oil, Filter, Battery, Brake Pad, Spark Plug, Sensor, Fuse, Belt, Fluid, Tool, Diagnostic Tool, Sealant, Cleaner — use whatever fits the product.

REMEMBER: Every response MUST have a blank line, then ---META---, then a blank line, then JSON."""


def build_enriched_prompt(req: EnrichRequest, conversation_summary: str = "") -> str:
    ctx = req.assetContext

    # Build a human-readable asset description
    parts = []
    if ctx.year:
        parts.append(str(ctx.year))
    if ctx.make:
        parts.append(ctx.make)
    if ctx.model:
        parts.append(ctx.model)
    asset_desc = " ".join(parts) if parts else (ctx.type or "unknown asset")
    if ctx.mileage:
        asset_desc += f" ({ctx.mileage:,} miles)"
    elif ctx.type and ctx.type != "vehicle":
        asset_desc = f"{ctx.type} — {asset_desc}"

    toolbox_text = (
        ", ".join(req.toolbox) if req.toolbox else "No tools specified"
    )
    history_text = _format_service_history(req.serviceHistory)
    temp_prefix = SKILL_LEVEL_PREFIXES.get(req.skillLevel, SKILL_LEVEL_PREFIXES[5])

    summary_section = (
        f"\nEarlier conversation (summarised):\n{conversation_summary}\n"
        if conversation_summary
        else ""
    )

    if asset_desc and asset_desc != "unknown asset":
        identity = f"You are a professional diagnostic AI assistant troubleshooting a {asset_desc}."
    else:
        identity = (
            "You are a professional diagnostic AI assistant. "
            "No specific device or equipment has been identified yet. "
            "Help the user identify what they need help with, ask clarifying questions if needed, "
            "then diagnose the issue."
        )

    return f"""{identity}

Asset service history:
{history_text}

Owner's available tools: {toolbox_text}
{summary_section}
{temp_prefix}

Analyse the reported symptoms and provide a complete diagnosis.

IMPORTANT: Respond ONLY in this exact JSON format — no extra text, no markdown outside the JSON:
{{
  "answer": "Your full diagnosis and recommendations here...",
  "symptom_match": <integer 0-100>,
  "history_alignment": <integer 0-100>,
  "specificity": <integer 0-100>,
  "safety_flag": <true or false>,
  "safety_explanation": "Brief explanation if safety_flag is true, otherwise empty string",
  "abstain": <true or false>,
  "affiliate_links": [
    {{"type": "affiliate_link", "label": "<concise buy label>", "data": "{{\\"category\\":\\"<broad category>\\",\\"name\\":\\"<specific product>\\"}}"}},
    ...
  ]
}}

Guidelines:
- symptom_match: how well the symptoms align with a known failure pattern
- history_alignment: how well the service history supports or contradicts your diagnosis
- specificity: how precise and actionable your diagnosis is (generic guesses score low)
- safety_flag: set true for anything involving electrical hazards, gas, structural integrity, or personal safety risk
- abstain: set true only if the issue is completely outside technical troubleshooting or would be dangerous to guess
- affiliate_links: list purchasable items (parts, fluids, consumables, tools) directly relevant to resolving or confirming the diagnosis. Use an empty array [] when nothing specific needs to be purchased. Category examples: Oil, Filter, Battery, Brake Pad, Spark Plug, Sensor, Fuse, Belt, Fluid, Tool, Diagnostic Tool, Sealant, Cleaner."""


def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, value))


def compute_confidence(parsed: dict) -> tuple[int, list[str]]:
    """Derive aggregate confidence score and human-readable reasons."""
    symptom_match = _clamp(int(parsed.get("symptom_match", 50)))
    history_alignment = _clamp(int(parsed.get("history_alignment", 50)))
    specificity = _clamp(int(parsed.get("specificity", 50)))

    score = _clamp((symptom_match + history_alignment + specificity) // 3)

    reasons = [
        f"Symptom match: {symptom_match}/100",
        f"History alignment: {history_alignment}/100",
        f"Specificity: {specificity}/100",
    ]

    if parsed.get("safety_flag"):
        explanation = parsed.get("safety_explanation") or "Safety-critical system"
        reasons.append(f"Safety concern: {explanation}")

    return score, reasons


@router.post("/enrich", response_model=EnrichResponse)
async def enrich(req: EnrichRequest):
    LogContext.set(thread_id=req.threadId, asset_type=req.assetContext.type)

    try:
        log_step("enrich_started", thread_id=req.threadId, asset_type=req.assetContext.type)

        recent_messages, older_summary = summarize_messages(req.messages)
        log_step("messages_processed", recent_count=len(recent_messages), has_older_summary=bool(older_summary))

        system_prompt = build_enriched_prompt(req, older_summary)
        grok_temp = GROK_LLM_TEMPERATURE.get(req.skillLevel, 0.55)

        model_cfg = get_model_config(route_enrich(req))
        raw = await call_llm(model_cfg, system_prompt, recent_messages, grok_temp)
        log_step("grok_response_received", raw_length=len(raw))

        parsed = parse_json_response(raw)

        if parsed.get("_parse_failed"):
            log_step("json_parse_failed", status="warning")
            return EnrichResponse(
                answer=parsed.get("_raw", "Unable to parse diagnosis."),
                confidence=30,
                confidence_reasons=["Response format unexpected — confidence reduced"],
                abstain=False,
                safety_flag=False,
            )

        confidence, reasons = compute_confidence(parsed)
        log_result(
            "enrich_success",
            success=True,
            confidence=confidence,
            safety_flag=bool(parsed.get("safety_flag", False)),
            abstain=bool(parsed.get("abstain", False)),
        )

        return EnrichResponse(
            answer=parsed.get("answer", "No diagnosis provided."),
            confidence=confidence,
            confidence_reasons=reasons,
            abstain=bool(parsed.get("abstain", False)),
            safety_flag=bool(parsed.get("safety_flag", False)),
            affiliate_links=parsed.get("affiliate_links", []),
        )

    except httpx.HTTPStatusError as exc:
        logger.error(f"Grok API HTTP error: {exc.response.status_code}", extra={"status_code": exc.response.status_code})
        raise HTTPException(status_code=500, detail="Grok API error")
    except httpx.TimeoutException:
        logger.error("Grok API timeout", extra={"thread_id": req.threadId})
        raise HTTPException(status_code=500, detail="Grok API timeout")
    except ValueError as exc:
        logger.error(f"Configuration error: {exc}")
        raise HTTPException(status_code=500, detail="Server configuration error")
    except Exception as exc:
        logger.exception(f"Unexpected error in enrich: {type(exc).__name__}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/enrich/stream")
async def enrich_stream(req: EnrichRequest):
    """Stream enrichment response as Server-Sent Events (word-by-word)."""
    LogContext.set(thread_id=req.threadId, asset_type=req.assetContext.type)

    async def event_generator():
        try:
            log_step("enrich_stream_started", thread_id=req.threadId, asset_type=req.assetContext.type)

            recent_messages, older_summary = summarize_messages(req.messages)

            # Vision pre-analysis — call Grok Vision if the client attached a photo
            vision_analysis = None
            if req.imageUrl:
                try:
                    log_step("vision_analysis_started", url_prefix=req.imageUrl[:60])
                    vision_prompt = (
                        "Describe what is visible in this photo in 2-3 sentences. "
                        "Focus on the component shown, its apparent condition, and any visible "
                        "defects, damage, wear, or anomalies. Plain text only, no JSON."
                    )
                    vision_analysis = await analyze_image(req.imageUrl, vision_prompt)
                    log_step("vision_analysis_completed", chars=len(vision_analysis))
                except Exception as exc:
                    logger.warning("Vision pre-analysis failed (%s) — continuing without it", exc)
                    vision_analysis = None

            system_prompt = build_stream_prompt(req, older_summary, vision_analysis=vision_analysis)
            grok_temp = GROK_LLM_TEMPERATURE.get(req.skillLevel, 0.55)
            model_cfg = get_model_config(route_enrich(req))

            full_response = ""
            streamed_pos = 0   # bytes already emitted to the client
            meta_start = -1    # index in full_response where metadata begins (after separator)

            async for chunk in call_llm_stream(model_cfg, system_prompt, recent_messages, grok_temp):
                full_response += chunk

                if meta_start == -1:
                    # Layer 1: strict separator search (with newlines)
                    search_from = max(0, streamed_pos - 5)
                    m = _SEP_RE.search(full_response, search_from)
                    if m:
                        # Separator found — emit answer up to it, mark where meta begins
                        meta_start = m.end()
                        to_emit = full_response[streamed_pos:m.start()]
                        if to_emit:
                            yield f"data: {json.dumps({'chunk': to_emit})}\n\n"
                        streamed_pos = meta_start
                        logger.debug("Stream separator (strict) found at pos %d", m.start())
                    else:
                        # Layer 1b: lenient separator search (catches missing newlines)
                        # Only search the tail to avoid expensive full-text rescans
                        tail_search_from = max(0, len(full_response) - 100)
                        m_lenient = _SEP_LENIENT_RE.search(full_response, tail_search_from)
                        if m_lenient:
                            # Lenient separator found — emit answer up to it
                            meta_start = m_lenient.end()
                            to_emit = full_response[streamed_pos:m_lenient.start()]
                            if to_emit:
                                yield f"data: {json.dumps({'chunk': to_emit})}\n\n"
                            streamed_pos = meta_start
                            logger.debug("Stream separator (lenient) found at pos %d", m_lenient.start())
                        else:
                            # Hold back _HOLDBACK chars so a separator spanning chunks isn't
                            # accidentally emitted as answer text before we can detect it
                            safe_end = max(streamed_pos, len(full_response) - _HOLDBACK)
                            to_emit = full_response[streamed_pos:safe_end]
                            if to_emit:
                                yield f"data: {json.dumps({'chunk': to_emit})}\n\n"
                                streamed_pos = safe_end

            # ── Stream finished ─────────────────────────────────────────────────

            if meta_start == -1:
                # Separator was never found during streaming — apply layer 2: trailing JSON heuristic
                remaining = full_response[streamed_pos:]

                # Try to find and extract trailing metadata JSON
                trail = _TRAILING_META_RE.search(remaining)
                if trail:
                    logger.warning(
                        "Separator absent — stripped trailing metadata via heuristic "
                        "(thread=%s). Consider improving Grok separator compliance.",
                        req.threadId,
                    )
                    clean = remaining[: trail.start()].rstrip()
                    if clean:
                        yield f"data: {json.dumps({'chunk': clean})}\n\n"
                    metadata = _parse_meta_str(trail.group())
                else:
                    # No trailing metadata found — try one more layer: lenient separator in full response
                    m_final = _SEP_LENIENT_RE.search(full_response[streamed_pos:])
                    if m_final:
                        logger.warning(
                            "Separator found after streaming completed (lenient match) "
                            "(thread=%s). Parsing metadata.",
                            req.threadId,
                        )
                        sep_pos = streamed_pos + m_final.start()
                        meta_start = streamed_pos + m_final.end()
                        # Emit any remaining answer text before the separator
                        final_answer = full_response[streamed_pos:sep_pos].rstrip()
                        if final_answer:
                            yield f"data: {json.dumps({'chunk': final_answer})}\n\n"
                        metadata = _parse_meta_str(full_response[meta_start:])
                    else:
                        # No metadata found at all — emit remainder as plain answer
                        logger.warning(
                            "No separator or metadata found in stream (thread=%s). "
                            "Emitting full response as answer with default confidence.",
                            req.threadId,
                        )
                        if remaining:
                            yield f"data: {json.dumps({'chunk': remaining})}\n\n"
                        metadata = {}
            else:
                # Separator was found during streaming — parse the tail as metadata
                metadata = _parse_meta_str(full_response[meta_start:])

            # ── Emit typed completion event ──────────────────────────────────────
            confidence, reasons = compute_confidence(metadata)
            yield f"data: {json.dumps({'status': 'complete', 'confidence': confidence, 'abstain': bool(metadata.get('abstain', False)), 'safety_flag': bool(metadata.get('safety_flag', False)), 'affiliate_links': metadata.get('affiliate_links', [])})}\n\n"
            log_step("enrich_stream_completed", thread_id=req.threadId, confidence=confidence)

        except httpx.HTTPStatusError as exc:
            logger.error("Grok API HTTP error in /enrich/stream: %s", exc.response.status_code)
            yield f"data: {json.dumps({'error': 'Grok API error'})}\n\n"
        except httpx.TimeoutException:
            logger.error("Grok timeout in /enrich/stream for thread %s", req.threadId)
            yield f"data: {json.dumps({'error': 'Grok API timeout'})}\n\n"
        except ValueError as exc:
            logger.error("Configuration error: %s", exc)
            yield f"data: {json.dumps({'error': 'Server configuration error'})}\n\n"
        except Exception as exc:
            logger.exception("Unexpected error in /enrich/stream: %s", type(exc).__name__)
            yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
        finally:
            LogContext.clear()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
