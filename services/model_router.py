"""Model router — selects provider + config per request, dispatches to the right client.

Tier map
--------
  first_message  fastest available model — OpenAI gpt-4o-mini if key present, else Grok no-reasoning
  standard       grok-3-mini-fast, reasoning_effort: low
  deep           grok-3-mini-fast, reasoning_effort: high  (swap to claude when claude_client.py added)
  turbo          grok-3-mini-fast, reasoning_effort: none  — classify only (JSON extraction)

Routing signals for /enrich (checked in priority order)
---------------------------------------------------------
  imageUrl present              → deep   (multimodal — Grok handles internally, deep for diagnosis)
  "[user uploaded pdf" in msg   → deep   (dense doc, needs careful extraction)
  OBD/fault codes (P####)       → deep   (specific codes need precise reasoning)
  safety keywords in last msg   → deep   (wrong answer = physical risk)
  first message, no signals     → first_message
  everything else               → standard

/classify always → turbo  (structured JSON extraction, no reasoning needed)

Adding a new provider
---------------------
  1. Create services/<provider>_client.py with call_<p>() and call_<p>_stream()
  2. Add a tier entry in _build_models() with provider="<p>"
  3. Add dispatch branches in call_llm() and call_llm_stream()
  4. Flip the relevant tier's "provider" key — zero other changes required
"""
from __future__ import annotations

import os
import re
from typing import AsyncGenerator

from logger_config import logger

# ---------------------------------------------------------------------------
# Routing signal constants
# ---------------------------------------------------------------------------
_SAFETY_KEYWORDS = frozenset({
    "smoke", "fire", "burning", "spark", "sparks",
    "gas leak", "gas smell", "no brakes", "won't stop", "wont stop",
    "catch fire", "overheating", "explosion", "electric shock",
    "smell", "leak",
})

_FAULT_CODE_RE = re.compile(r'\b[PCBU]\d{4}\b', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Model config registry — built lazily so .env is guaranteed loaded
# ---------------------------------------------------------------------------
_MODELS_CACHE: dict[str, dict] | None = None


def _build_models() -> dict[str, dict]:
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    # first_message: pick the fastest non-reasoning model available
    if has_openai:
        first_message_cfg: dict = {
            "provider":  "openai",
            "model":     "gpt-4o-mini",
            "max_tokens": 512,
        }
    else:
        # Grok with no reasoning is the fallback — still fast
        first_message_cfg = {
            "provider":         "grok",
            "model":            "grok-3-mini-fast",
            "reasoning_effort": "none",
            "max_tokens":       512,
        }

    return {
        "first_message": first_message_cfg,
        "standard": {
            "provider":         "grok",
            "model":            "grok-3-mini-fast",
            "reasoning_effort": "low",
            "max_tokens":       1024,
        },
        "deep": {
            # To use Claude: set provider="claude", model="claude-opus-4-6"
            # and add services/claude_client.py + dispatch branches below
            "provider":         "grok",
            "model":            "grok-3-mini-fast",
            "reasoning_effort": "high",
            "max_tokens":       2048,
        },
        "turbo": {
            "provider":         "grok",
            "model":            "grok-3-mini-fast",
            "reasoning_effort": "none",
            "max_tokens":       1024,   # classify responses include recs + reminders, 512 can truncate
        },
    }


def get_model_config(tier: str) -> dict:
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        _MODELS_CACHE = _build_models()
    cfg = _MODELS_CACHE.get(tier, _MODELS_CACHE["standard"])
    logger.debug(
        "model_router: tier=%s provider=%s model=%s reasoning=%s",
        tier, cfg["provider"], cfg["model"], cfg.get("reasoning_effort", "n/a"),
    )
    return cfg


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _last_user_content(messages: list) -> str:
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if role == "user":
            return (content or "").lower()
    return ""


def route_enrich(req) -> str:
    """Return the tier name for an EnrichRequest."""
    is_first = len(req.messages) <= 1
    last = _last_user_content(req.messages)

    # --- Deep conditions (evaluated before first-message shortcut) ---
    if req.imageUrl:
        return "deep"

    if "[user uploaded pdf" in last:
        return "deep"

    if _FAULT_CODE_RE.search(last):
        return "deep"

    if any(kw in last for kw in _SAFETY_KEYWORDS):
        return "deep"

    # --- First message with no deep signals ---
    if is_first:
        return "first_message"

    return "standard"


def route_classify() -> str:
    """Classify is always turbo — JSON extraction, no conversational reasoning."""
    return "turbo"


# ---------------------------------------------------------------------------
# Dispatch — non-streaming
# ---------------------------------------------------------------------------

async def call_llm(
    config: dict,
    system_prompt: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None = None,
) -> str:
    """Dispatch a non-streaming LLM call to the provider specified in config."""
    provider = config["provider"]
    _max = max_tokens or config.get("max_tokens", 1024)

    if provider == "grok":
        from services.grok_client import call_grok
        return await call_grok(
            system_prompt,
            messages,
            temperature,
            max_tokens=_max,
            model=config.get("model"),
            reasoning_effort=config.get("reasoning_effort"),
        )

    if provider == "openai":
        from services.openai_client import call_openai
        return await call_openai(
            system_prompt,
            messages,
            temperature,
            model=config["model"],
            max_tokens=_max,
        )

    # provider == "claude":
    #   from services.claude_client import call_claude
    #   return await call_claude(system_prompt, messages, temperature,
    #                            model=config["model"], max_tokens=_max)

    raise ValueError(f"model_router: unknown provider {provider!r}")


# ---------------------------------------------------------------------------
# Dispatch — streaming
# ---------------------------------------------------------------------------

async def call_llm_stream(
    config: dict,
    system_prompt: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None = None,
) -> AsyncGenerator[str, None]:
    """Dispatch a streaming LLM call. Yields text chunks as they arrive."""
    provider = config["provider"]
    _max = max_tokens or config.get("max_tokens", 1024)

    if provider == "grok":
        from services.grok_client import call_grok_stream
        async for chunk in call_grok_stream(
            system_prompt,
            messages,
            temperature,
            max_tokens=_max,
            model=config.get("model"),
            reasoning_effort=config.get("reasoning_effort"),
        ):
            yield chunk
        return

    if provider == "openai":
        from services.openai_client import call_openai_stream
        async for chunk in call_openai_stream(
            system_prompt,
            messages,
            temperature,
            model=config["model"],
            max_tokens=_max,
        ):
            yield chunk
        return

    # provider == "claude":
    #   from services.claude_client import call_claude_stream
    #   async for chunk in call_claude_stream(system_prompt, messages, temperature,
    #                                         model=config["model"], max_tokens=_max):
    #       yield chunk
    #   return

    raise ValueError(f"model_router: unknown provider {provider!r}")
