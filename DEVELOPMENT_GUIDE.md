# ARWrench API — Development Guide

For the AI engineer building and maintaining this FastAPI layer.

---

## What This Project Is

ARWrench is an "AI mechanic in your pocket." The iOS app is already built. This FastAPI layer is the brain — it owns all AI reasoning.

**The product goal:** A user describes a problem they don't fully understand. The AI asks the right questions — progressively narrowing the diagnosis — until it has enough confidence to tell the user exactly what's wrong and what to do. The mobile app is a dumb UI bridge; it renders what the AI sends and forwards user input. Every decision — what to ask next, when to request a photo, when to declare a diagnosis — belongs to this project.

**Your job:** Build and maintain the FastAPI layer that makes the AI mechanic real.

---

## Architecture

```
iOS (renders only)
    ↓
Convex (gathers DB data, calls FastAPI, saves result)
    ↓
FastAPI ← you own this
    ↓
Grok / Grok Vision (external AI APIs)
```

**Clean split:**
- Convex: data storage, auth, real-time sync, DB queries
- FastAPI: all AI logic — prompt design, Grok calls, confidence scoring, RAG
- iOS: display only

FastAPI is stateless. It receives everything it needs in the request. It stores nothing between calls (except Chroma for RAG).

---

## Project Structure

```
ArWrench-Api/
├── main.py                   # FastAPI app, middleware, router registration
├── models.py                 # All Pydantic request/response schemas
├── requirements.txt
├── .env                      # API keys — never commit
├── routers/
│   ├── enrich.py             # POST /enrich — the conversation engine
│   ├── vision.py             # POST /vision — image analysis
│   └── classify.py           # POST /classify + POST /classify/refine
├── services/
│   ├── grok_client.py        # call_grok() + parse_json_response()
│   ├── grok_vision_client.py # analyze_image() — fetches URL, base64, sends to Grok Vision
│   └── rag_service.py        # ingest_pdf(), retrieve_relevant_chunks() — Chroma + embeddings
├── logs/
│   └── arwrench.log          # JSON structured logs (auto-rotated)
└── tests/
    └── test_endpoints.py     # Full mock test suite
```

---

## Run Locally

```bash
cd /home/haku/projects/python/ArWrench-Api
source venv/bin/activate
python -m uvicorn main:app --reload --port 8000

# Verify
curl http://localhost:8000/health
```

Environment variables required in `.env`:
```
GROK_API_KEY=xai-...
GROK_MODEL=grok-3-mini
GROK_VISION_MODEL=grok-2-vision-1212
LOG_LEVEL=INFO
```

---

## Endpoint Roles and Design Intent

### POST /enrich — the conversation engine

**Role:** Every turn of the diagnostic conversation, from session open to close. This is the core of the product.

**Design intent:** The system prompt must put Grok in *diagnostic mode* — the AI is a mechanic who does not yet have enough information. It asks the single most useful next question to narrow the problem. It does not guess a diagnosis until its confidence is high. When it has enough, it delivers the finding and tells the user what to do or not do.

**Key files:** [routers/enrich.py](routers/enrich.py), [services/grok_client.py](services/grok_client.py)

**Current system prompt stance:** "Analyse the reported symptoms and provide a complete diagnosis."

**Required stance:** The AI should be in discovery mode first. It should ask questions. It should only deliver a diagnosis when it has gathered sufficient context. The prompt needs to reflect this — this is the primary prompt gap to address.

**Confidence scoring:** Computed from `symptom_match`, `history_alignment`, `specificity` returned by Grok. Low early in a session is expected — the AI is still asking questions. High when it's ready to diagnose.

**Message summarisation:** Long threads are handled in `summarize_messages()` — keeps the last 5 messages verbatim, compresses older ones into a summary injected into the system prompt. This prevents token overflow on long sessions.

---

### POST /vision — image analysis (on demand)

**Role:** Analyze a specific photo when the AI's conversation asks for one. Not a fixed pipeline step.

**Design intent:** The AI mechanic asks "can you take a photo of the compressor unit?" in its `/enrich` answer. The app captures the photo, calls `/vision`, gets the analysis back, and injects it into the `messages[]` array as an assistant observation. The next `/enrich` call has that visual context and continues the conversation.

**Key files:** [routers/vision.py](routers/vision.py), [services/grok_vision_client.py](services/grok_vision_client.py)

**Current state:** Fully functional. Grok Vision analyzes the image and returns structured findings. The injection pattern (vision result as assistant message) is handled on the Convex side.

---

### POST /classify — session start / asset identification

**Role:** Called once at session start. Extracts structured asset specs (make/model/year/type) from the user's initial description. Populates the asset record in Convex so `/enrich` has structured `assetContext` to work with.

**Design intent:** Lightweight. Get the asset identified. The actual diagnostic conversation starts in `/enrich`. This endpoint is not the diagnosis — it is the intake.

**Key files:** [routers/classify.py](routers/classify.py)

**Current state:** Works. Note that `recommendations` in the response is a rough initial read — the real diagnosis happens through the `/enrich` conversation. Convex should NOT use these recommendations to generate a Q&A list for the user; the AI will ask what it needs through conversation.

---

### POST /classify/refine — asset record improvement

**Role:** Refines asset specs when initial classification confidence is low. Takes structured Q&A answers and updates the asset record in Convex with more accurate data (e.g., actual mileage, confirmed model year).

**Design intent:** This is about improving data quality in Convex, not about the diagnostic conversation. The conversation happens through `/enrich`. The dual-pass self-critique (second Grok call when confidence < 50) lives here.

**Key files:** [routers/classify.py](routers/classify.py) — `classify_refine()` function

---

## Known Gaps — What Needs to Change

### 1. /enrich system prompt is in answer mode, not discovery mode

**File:** [routers/enrich.py:201](routers/enrich.py#L201) — `build_enriched_prompt()`

**Current:**
```python
"Analyse the reported symptoms and provide a complete diagnosis."
```

**Required:** The prompt must instruct the AI to:
- Be in discovery mode until it has sufficient confidence
- Ask one focused question per turn to narrow the problem
- Only declare a final diagnosis when it genuinely has enough context
- Tell the user clearly what to do / not do once diagnosed

This is the most important prompt change — it transforms the system from "answer machine" to "diagnostic mechanic."

### 2. No photo_request field in /enrich response

**File:** [models.py](models.py) — `EnrichResponse`, [routers/enrich.py](routers/enrich.py) — Grok JSON schema

The AI currently has no way to signal "I need a photo of [specific part]" in a structured way. It can mention it in `answer` text, but the app has no structured field to trigger the camera flow.

**What's needed:** Add `photo_request: { part: str, instructions: str } | None` to the response schema, and instruct Grok to populate it when visual evidence would help diagnosis.

### 3. /classify prompt doesn't distinguish asset ID from diagnosis

The current `/classify` prompt asks for full diagnosis. Its intent should be limited to asset identification. The system prompt should be scoped to: "identify the asset type, make, model, year from the description — the diagnostic conversation happens separately."

---

## Confidence Scoring

Computed in `compute_confidence()` in [routers/enrich.py](routers/enrich.py):

```python
score = (symptom_match + history_alignment + specificity) // 3
```

Grok returns these three integers (0–100) in its JSON response. Low scores early in a conversation are expected — the AI is still gathering information. High scores indicate the AI has enough context to be specific.

The `safety_flag` is set by Grok when the issue involves brakes, fuel, structural, or electrical systems. It triggers a warning banner in the app.

`abstain: true` means the AI cannot reliably help. The app shows a "consult a professional" message.

---

## Skill Level System

10 levels mapping user technical knowledge to AI response style. Defined in both `routers/enrich.py` and `routers/classify.py` as `SKILL_LEVEL_PREFIXES`.

- 1–2: Plain language, always defers to professional
- 3–4: Step-by-step for simple jobs
- 5–6: Full procedures, torque specs (default: 5)
- 7–8: OBD codes, differential diagnosis
- 9–10: OEM procedures, TSBs, fault tree analysis

Temperature also scales with skill level (0.2 at level 1 → 0.8 at level 10). Lower = more deterministic for novices. Higher = more nuanced for experts.

---

## RAG (Milestone 2)

When users upload PDFs (e.g., service manuals, owner documents), they are chunked, embedded, and stored in Chroma per asset.

**Critical rule:** Every RAG query filters by `asset_id`. No cross-asset data ever.

**Key file:** [services/rag_service.py](services/rag_service.py)

RAG context is injected into the `/classify` system prompt when relevant chunks exist for the asset.

---

## Logging

Structured JSON logs written to `logs/arwrench.log` (auto-rotated at 10MB, 5 backups).

```bash
# View errors
cat logs/arwrench.log | jq 'select(.level == "ERROR")'

# Find slow Grok calls
cat logs/arwrench.log | jq 'select(.service == "grok") | .duration_ms'
```

Log level controlled by `LOG_LEVEL` env var. Set `DEBUG` for verbose output during development.

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

Tests use mocked Grok responses — no real API keys needed for the test suite. Cross-asset isolation tests verify that RAG queries never leak data between assets.

---

## Security Rules

- All API keys from environment — never hardcode, never log, never expose in responses
- Every RAG query filters by `asset_id` — no cross-asset data
- FastAPI has no auth — Convex is already authenticated; FastAPI receives trusted data
- FastAPI stores nothing between requests (Chroma is the only persistent state, and it's keyed by asset)

---

## Milestones

| Milestone | Endpoints | Status |
|-----------|-----------|--------|
| M1 — $250 | `/enrich` (Grok, confidence scoring) | ✅ Built |
| M2 — $250 | `/vision` (Grok Vision), `/classify` (RAG) | ✅ Built |
| M3 — $250 | `/classify/refine` (dual-pass), full test suite | ✅ Built |

**Primary remaining work:** Prompt design — specifically the `/enrich` system prompt needs to be reoriented to discovery/diagnostic mode. The endpoints are built; the AI mechanic persona is the gap.
