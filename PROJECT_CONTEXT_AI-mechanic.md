# ARWrench – AI Enrichment Layer: Project Context

---

## What This Project Is

ARWrench is an iOS app ("mechanic in your pocket") that helps users diagnose and maintain vehicles, equipment, and appliances. The **mobile app is already built** with Clerk auth + Convex backend.

**Your job:** Build a FastAPI enrichment layer that increases AI reliability and reasoning quality.

**Typical enriched flow:**

1. iOS user describes a symptom via text or photo
2. iOS → Convex → **your FastAPI** (full AI request)
3. FastAPI receives: chat history, asset context, toolbox, service history
4. FastAPI enriches context, calls Grok internally, returns final answer + confidence
5. Convex saves the answer (no Grok call in Convex)
6. Answer saved in Convex, real-time subscribed iOS gets result

**Your responsibility:** FastAPI owns ALL AI calls (Grok + Gemini). Convex only passes data and saves results. That's the clean split.

---

## Milestone Breakdown (Fixed Payment = 3 × $250)

**Priority: Hit all 3 milestones over future scalability.**

### Milestone 1 — $250
- `/enrich` endpoint: receive chat + asset context, return enriched prompt + confidence score
- Confidence scoring (symptom match, history alignment, specificity)
- Grok call wrapper (call Grok via FastAPI, return structured JSON)
- Per-asset isolation (no cross-asset data contamination)

### Milestone 2 — $250
- `/vision` endpoint: Gemini analyzes photos, returns structured analysis
- `/classify` endpoint: extract asset specs from vision + text description
- Lightweight RAG: embed user PDFs, retrieve relevant chunks with asset_id filtering
- Return structured JSON (specs, condition, confidence)

### Milestone 3 — $250
- Dual-pass validation: low confidence (<50) triggers self-critique loop
- Temperature slider: 5 levels map to prompt behavior (simple language → expert detail)
- Full test suite: cross-asset tests, hallucination checks, mock Grok responses
- Documentation: what each endpoint does, request/response schema, how to integrate

**Discipline: Do exactly this. No more, no less. Ship, get paid, move on.**

---

## What To Build (4 Endpoints Only)

### Endpoint 1: POST /enrich (Milestone 1)
**Input** (from Convex):
```json
{
  "threadId": "t_123",
  "messages": [{"role": "user", "content": "..."}, ...],
  "assetContext": {"type": "vehicle", "model": "Camry", "year": 2018, "specs": {...}},
  "toolbox": ["wrench", "socket_set", ...],
  "serviceHistory": [{...}, ...],
  "temperature": 3
}
```

**What FastAPI does internally:**
1. Enriches prompt with asset context, toolbox, service history
2. Calls Grok with enriched prompt
3. Parses Grok response + scores confidence
4. Returns final answer to Convex

**Output** (Convex saves this directly):
```json
{
  "answer": "Based on your Camry's history and the grinding noise...",
  "confidence": 78,
  "confidence_reasons": ["symptom match strong", "history supports"],
  "abstain": false,
  "safety_flag": false,
  "role": "assistant"
}
```

### Endpoint 2: POST /vision (Milestone 2)
**Input** (from Convex):
```json
{
  "imageUrl": "https://...",
  "systemPrompt": "You are analyzing a car engine part",
  "assetContext": {...}
}
```

**Output**:
```json
{
  "analysis": "Appears to be a cracked water pump gasket. Oil leak visible...",
  "confidence": 85,
  "structured_findings": {"part": "water_pump", "condition": "damaged", ...}
}
```

### Endpoint 3: POST /classify (Milestone 2)
**Input** (from Convex):
```json
{
  "visionResult": "...from /vision",
  "textDescription": "Car overheating, see photo",
  "assetId": "asset_456"
}
```

**Output**:
```json
{
  "assetSpecs": {"type": "vehicle", "make": "Toyota", "model": "Camry", ...},
  "condition": "maintenance_needed",
  "confidence": 88,
  "recommendations": ["replace gasket", "refill coolant"]
}
```

### Endpoint 4: POST /classify/refine (Milestone 3)
**Input** (from Convex):
```json
{
  "initialClassification": {...from /classify},
  "userAnswers": [{"q": "When did it start?", "a": "3 days ago"}, ...],
  "temperature": 3
}
```

**Output**:
```json
{
  "refined_specs": {...updated specs...},
  "confidence": 92,
  "refined_recommendations": [...]
}
```

---

## What NOT To Build

- Anything except these 4 endpoints
- User interface
- Database schema (Convex already has it)
- Auth handling (Convex handles it)
- Affiliate links
- Push notifications

---

## Architecture: FastAPI Enrichment Layer on Convex

**FastAPI owns all AI. Convex passes data + saves results.**

```
iOS (Clerk + Convex)
  ↓
Convex gathers data from its DB:
  - chat messages, asset specs, service history, toolbox, reminders
  ↓
Convex calls POST /enrich (your FastAPI)
  ↓
FastAPI:
  1. Enriches prompt (injects all context)
  2. Calls Grok API internally
  3. Parses Grok response
  4. Scores confidence
  ↓
FastAPI returns: {answer, confidence, safety_flag, ...}
  ↓
Convex saves answer to chatMessages (no Grok call in Convex)
  ↓
iOS real-time subscription delivers answer to user
```

**Clean split:**
- **Convex:** data storage, auth, real-time sync, scheduling
- **FastAPI:** all AI logic (prompt building, Grok calls, confidence scoring, RAG)

**Convex no longer calls Grok directly.** All 5 existing `fetch()` calls to Grok/Gemini in Convex get replaced with calls to your FastAPI endpoints.

---

## API Security & Key Management

**All API calls (Grok, Gemini) go through FastAPI, never from mobile app.**

- API keys stored in `.env` (local dev) or environment variables (deployed)
- Never hardcode, never log, never expose in responses
- FastAPI is the gateway — if Convex needs Grok, it goes through you
- Convex already knows your FastAPI URL (in Constants.swift or CLAUDE.md)

**For testing:** Use `.env.local` with:
```
GROK_API_KEY=your-key
GEMINI_API_KEY=your-key
```

---

## RAG (Milestone 2 — Lightweight)

**Why:** Grok hallucinates on specifics (torque specs, error codes). RAG grounds it in documents.

**How (minimal):**
1. User uploads PDF via Convex → `/classify/enrich` receives it
2. Chunk + embed with OpenAI embeddings API
3. Store chunks in **Chroma** (local, simple, no extra service)
4. Tag each chunk with `asset_id`
5. When enriching, retrieve top-3 chunks matching query + asset_id filter
6. Inject chunks into enriched_prompt

**Tools:** Use only LangChain's `document_loaders` and `text_splitter`. Everything else plain Python.

**Code outline:**
```python
from langchain_community.document_loaders import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Load + chunk PDF
loader = PDFLoader(pdf_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)
chunks = splitter.split_documents(docs)

# Embed + store in Chroma
client = chromadb.Client()
collection = client.get_or_create_collection(name=f"asset_{asset_id}")
for chunk in chunks:
    collection.add(documents=[chunk.page_content], ids=[chunk.metadata.get("id")])

# Retrieve at inference time
results = collection.query(query_texts=[user_question], n_results=3)
rag_context = "\n".join([r for r in results["documents"][0]])
```

**Critical:** Every query filters by `asset_id`. No cross-asset data.

---

## Endpoint Logic (per milestone)

### Milestone 1: POST /enrich
```
Input: {messages, assetContext, toolbox, serviceHistory, temperature}
  → Summarize messages (keep last 5, compress older)
  → Inject asset specs: "2018 Toyota Camry, 120k miles"
  → Inject toolbox: "Available: wrench, socket set, ..."
  → Inject service history: "Last oil change: 2024-01-15..."
  → Build enriched system prompt
  → Call Grok API with enriched prompt + messages
  → Parse Grok response
  → Score confidence on the response
  → Return {answer, confidence, safety_flag, abstain}
  (Convex saves answer directly — no second AI call needed)
```

### Milestone 2: POST /vision → /classify
```
/vision:
Input: {imageUrl, assetContext, systemPrompt}
  → Call Gemini Vision API with image
  → Extract structured findings: {part, condition, severity}
  → Return {analysis, findings, confidence}

/classify:
Input: {visionResult, textDescription, assetId}
  → Combine vision + text
  → Retrieve relevant RAG chunks (if PDF exists for this asset)
  → Infer asset specs from vision + text + RAG
  → Return {assetSpecs, condition, recommendations, confidence}
```

### Milestone 3: POST /classify/refine
```
Input: {initialClassification, userAnswers, temperature}
  → Build follow-up prompt with user answers
  → Call Grok with refined context
  → Compare to initial classification
  → Trigger dual-pass validation if confidence < 50
  → Return {refined_specs, refined_confidence, recommendations}
```

---

## Confidence Scoring (Milestone 1 & 3)

Grok evaluates against criteria in the prompt. Ask Grok to return a confidence score 0–100 based on:
- **Symptom match:** How well do symptoms align with known failure patterns?
- **History alignment:** Does asset history support or contradict the diagnosis?
- **Specificity:** Is this a precise diagnosis or generic guess?
- **Safety:** Flag anything electrical, fuel, structural (auto-flag = lower confidence)

**Prompt injection (Milestone 1):**
```
You are diagnosing a [{asset_type}].
History: {service_history}
Available tools: {toolbox}
Current issue: {symptom}

After your diagnosis, rate your confidence 0-100 on:
- Symptom match (0-100): [explain]
- History alignment (0-100): [explain]
- Specificity (0-100): [explain]
- Safety concerns: [yes/no/maybe]

Confidence score = (symptom_match + history + specificity) / 3
```

**Milestone 3 (dual-pass):**
If confidence < 50, Grok self-critiques:
```
Your confidence was {score}. Is this reasonable?
Have you missed anything? What else should you check?
```

Return format:
```json
{
  "confidence": 78,
  "confidence_reasons": ["symptom match 80", "history supports 75", "specific diagnosis"],
  "safety_flag": false,
  "abstain": false
}
```

---

## Temperature Slider Mapping (Milestone 3)

Convex sends `temperature: 1-5` in every request. Adjust prompt behavior:

| Level | Prompt Prefix |
|-------|---------------|
| 1 | "Explain simply in everyday language. Assume no technical knowledge. If risky, strongly recommend a professional." |
| 2 | "Use basic explanations. Be conservative with DIY recommendations." |
| 3 | "Standard technical detail. DIY is OK if safe." |
| 4 | "Assume technical knowledge. Full depth. Assume tools available." |
| 5 | "Expert level. Professional terminology. Minimal hand-holding." |

**Implementation:** Inject temperature prompt prefix into enriched_prompt before calling Grok.

---

## Prompt Templates (All Milestones)

Keep it simple. One template per endpoint:

**M1: /enrich template**
```
You are diagnosing a {asset_type} ({asset_make} {asset_model} {asset_year}).

Asset history:
- Last service: {last_service}
- Known issues: {known_issues}
- Owner tools: {toolbox}

User has been experiencing this issue for {duration}.
Symptom: {current_symptom}

{temperature_prefix}

Provide diagnosis with:
1. Root cause
2. Severity (minor/moderate/critical)
3. DIY steps (if safe) OR recommend professional
4. Safety warnings
5. Your confidence (0-100) with reasoning
```

**M2: /vision template**
```
Analyze this photo of a {asset_type}.
What part is visible? What's its condition?
Format: {"part": "...", "condition": "...", "observations": [...]}
```

**M2: /classify template**
```
Based on this vision analysis and user description,
what are the asset specs and recommended next steps?
```

**M3: /classify/refine template**
```
User answered your clarifying questions.
Refine your initial classification.
Are you more confident now?
```

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Framework | FastAPI | Simple, fast, async-ready |
| Web server | uvicorn | FastAPI default |
| Vector DB | Chroma | Local, no extra service needed (M2) |
| Primary AI | Grok API | Convex already uses it |
| Vision AI | Gemini Vision API | Multi-modal (M2) |
| Embeddings | OpenAI API | For RAG (M2) |
| Orchestration | Plain Python | No LangChain agents/chains |
| RAG tools | LangChain loaders + splitter | PDF parsing only (M2) |
| Secrets | `.env` (dev) or env vars (prod) | Never hardcode |
| Testing | pytest + mock | No dependencies needed |
| Deployment | Local first, then Render/Railway | TBD after M1 ships |

---

## Security Rules (Critical)

- All API keys in `.env` (dev) or environment variables (prod) — **never hardcode**
- No keys logged or exposed in responses/errors
- **Asset isolation:** Every RAG query filters by `asset_id`. No cross-asset data.
- **Stateless design:** FastAPI doesn't store user/asset data. Convex does.
- **Convex already authenticated:** Don't add auth to FastAPI. Convex sends trusted data.
- Test locally with `.env.local`; CI/CD injects vars at deploy time

---

## FastAPI Endpoints (4 Total)

Convex calls these. Nothing else.

```
POST /enrich           (M1) — Milestone 1
POST /vision           (M2) — Milestone 2
POST /classify         (M2) — Milestone 2
POST /classify/refine  (M3) — Milestone 3
GET  /health           — health check
```

**Request/Response schemas:** See "Endpoint Logic" section above.

**Error handling:** Return 400 with `{"error": "message"}` if input invalid, 500 for API failures.

---

## Getting Started

**Resources:**
- `CLAUDE.md` — iOS app architecture (Clerk, Convex, SwiftUI)
- `DEVELOPMENT_GUIDE.md` — Quick start, setup, test commands
- `API_INTEGRATION_MAP.md` — Where Convex calls your endpoints (if exists)

**Files to create:**
```
/home/haku/projects/python/ArWrench-Api/
├── main.py                 # FastAPI app + endpoints
├── .env                    # API keys (dev only)
├── requirements.txt        # dependencies
├── models.py               # Pydantic models
├── services/
│   ├── grok_client.py     # Grok API wrapper
│   ├── gemini_client.py   # Gemini Vision wrapper
│   └── rag_service.py     # Chroma + embedding (M2)
├── routers/
│   ├── enrich.py          # /enrich endpoint (M1)
│   ├── vision.py          # /vision endpoint (M2)
│   └── classify.py        # /classify endpoints (M2/M3)
└── tests/
    └── test_endpoints.py  # Mock tests
```

---

## Definition of Done

### Milestone 1 ✅ = $250
- [ ] `/enrich` endpoint receives messages + asset context + toolbox + service history
- [ ] Returns enriched_prompt (context injected) + confidence score (0-100) + reasoning
- [ ] Confidence scoring works (symptom match, history alignment, specificity, safety flag)
- [ ] Grok API called successfully, response parsed
- [ ] Per-asset isolation verified (test: asset A context doesn't leak to B)
- [ ] `.env` secrets working (no hardcoded keys)
- [ ] Test with Postman collection

### Milestone 2 ✅ = $250
- [ ] `/vision` endpoint calls Gemini Vision, returns structured analysis + confidence
- [ ] `/classify` endpoint ingests vision result + text description, returns asset specs
- [ ] RAG basics working: PDF upload → chunk → embed → store in Chroma
- [ ] RAG retrieval: asset_id filter enforced, top-3 chunks injected into enriched_prompt
- [ ] Structured JSON output for all responses

### Milestone 3 ✅ = $250
- [ ] `/classify/refine` endpoint works with dual-pass validation
- [ ] Low confidence (<50) triggers self-critique loop
- [ ] Temperature slider (1-5) changes prompt behavior correctly
- [ ] Test suite: cross-asset tests, hallucination checks, mock Grok responses passing
- [ ] Documentation: endpoint schemas, request/response examples
- [ ] All 4 endpoints documented + tested

**Payment:** 3 milestones × $250 = $750 total. Ship each milestone separately.
