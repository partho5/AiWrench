# ARWrench API — Mobile Client Guide

The only document a mobile/Convex developer needs.

---

## Mental Model — Read This First

**The AI is the mechanic. The app is a display.**

- The AI drives every turn of the diagnostic: it asks questions, requests photos, and declares when it has found the problem
- The app renders what comes back and forwards user responses — nothing more
- **Never build app-side logic to decide what question to ask, when to switch phases, or what the user should do next** — that is the AI's job
- The `messages[]` array is the single source of truth for the entire session. It grows with every turn and is passed in full on every `/enrich` call

---

## The Session Flow

```
1. User opens the app and describes a problem (text)
         ↓
   POST /classify
   → Identifies the asset (make/model/year)
   → Save assetSpecs to your DB
   → Use assetSpecs as assetContext in all /enrich calls
         ↓
2. Conversation begins — AI asks first diagnostic question
         ↓
   POST /enrich (Turn 1)
   messages: [user's initial description]
   → AI responds with a question to narrow the problem
   → Append response to messages as {role: "assistant"}
         ↓
3. User answers
         ↓
   POST /enrich (Turn 2)
   messages: [user_1, assistant_1, user_2]
   → AI asks next question, or requests a photo, or diagnoses
         ↓
4. If AI asks for a photo:
         ↓
   → Show camera UI to user
   → User takes the specific photo the AI asked for
         ↓
   POST /vision
   → Analyze that specific photo
   → Inject result into messages as:
     { role: "assistant", content: "[Photo] " + vision.analysis }
         ↓
   POST /enrich (next turn)
   → AI now has visual context and continues diagnosing
         ↓
5. AI reaches sufficient confidence → delivers diagnosis + instructions
         ↓
6. User asks follow-up questions → continue POST /enrich loop
```

The AI decides when it has enough. The app just keeps the loop going.

---

## Endpoint 1 — POST /classify

**Purpose:** Start a session. Identify the asset. Get the structured specs needed for `/enrich`.

**When to call:** Once, at session start, before any conversation.

### Request

```json
{
  "visionResult": null,
  "textDescription": "My car AC stopped working. Makes a loud humming when I turn it on.",
  "assetId": "asset_abc_123"
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `textDescription` | string | ✅ | User's initial problem description |
| `assetId` | string | ✅ | Your DB ID for this asset |
| `visionResult` | string \| null | No | Only if the user uploaded a photo at session start |

### Response

```json
{
  "assetSpecs": {
    "type": "vehicle",
    "make": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": null
  },
  "condition": "repair_needed",
  "confidence": 55,
  "recommendations": ["..."],
  "reminders": []
}
```

**What to do:**
1. Save `assetSpecs` to your DB (asset record)
2. Pass `assetSpecs` as `assetContext` in every subsequent `/enrich` call
3. Save `reminders` to your reminders table

> Do not use `recommendations` to generate your own Q&A list. The AI asks what it needs through the `/enrich` conversation.

---

## Endpoint 2 — POST /enrich (the conversation engine)

**Purpose:** Every single turn of the diagnostic conversation — from the first AI question through final diagnosis and any follow-up chat. This is the core loop.

**When to call:** After `/classify`, for every user message, for every turn.

### Request — Turn 1 (opening the conversation)

```json
{
  "threadId": "thread_camry_ac_001",
  "messages": [
    {
      "role": "user",
      "content": "My car AC stopped working. Makes a loud humming when I turn it on."
    }
  ],
  "assetContext": {
    "type": "vehicle",
    "make": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": null
  },
  "toolbox": [],
  "serviceHistory": [],
  "skillLevel": 3
}
```

### Request — Turn 2 (user replied)

Append the AI's previous answer and the new user message, then send the full array:

```json
{
  "threadId": "thread_camry_ac_001",
  "messages": [
    {"role": "user", "content": "My car AC stopped working. Makes a loud humming when I turn it on."},
    {"role": "assistant", "content": "How long has this been happening, and does the humming occur only when the AC is on?"},
    {"role": "user", "content": "Started yesterday. Yes, only when AC is on, not when I use just the fan."}
  ],
  "assetContext": {
    "type": "vehicle",
    "make": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": null
  },
  "toolbox": [],
  "serviceHistory": [],
  "skillLevel": 3
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `threadId` | string | ✅ | Unique ID for this session thread |
| `messages` | object[] | ✅ | Full conversation history. Grows by 2 per turn (user + assistant) |
| `messages[].role` | string | ✅ | `"user"` or `"assistant"` |
| `messages[].content` | string | ✅ | Message text |
| `assetContext` | object | ✅ | Use `assetSpecs` from `/classify` response. Pass every turn |
| `toolbox` | string[] | No | Tools the user has available. Empty array if unknown |
| `serviceHistory` | object[] | No | Past service records |
| `serviceHistory[].date` | string | No | ISO date, e.g. `"2023-08-10"` |
| `serviceHistory[].service` | string | No | e.g. `"Oil change"` |
| `skillLevel` | integer 1–10 | No | Defaults to 5. Set from user profile |

### Response

```json
{
  "answer": "Does the humming stop immediately when you turn the AC off, or does it linger for a few seconds?",
  "confidence": 38,
  "confidence_reasons": [
    "Symptom match: 45/100",
    "History alignment: 25/100",
    "Specificity: 45/100"
  ],
  "abstain": false,
  "safety_flag": false,
  "role": "assistant"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `answer` | string | What the AI says to the user. Display this in chat |
| `confidence` | integer 0–100 | Current diagnostic confidence. Will be low early — the AI is still gathering information |
| `confidence_reasons` | string[] | Breakdown of the confidence score. Display optionally |
| `abstain` | boolean | If `true`, the AI cannot reliably help. Show: *"I'm not confident enough to answer. Please consult a qualified technician."* |
| `safety_flag` | boolean | If `true`, show a safety warning banner above the answer |
| `role` | string | Always `"assistant"`. Use as `role` when appending to `messages` |

### Streaming variant — POST /enrich/stream

An alternative to the standard `/enrich` endpoint that returns the AI's response word-by-word as it is generated, rather than waiting for the full response before returning.

**When to use:** Preferred for the chat UI — the user sees text appear immediately instead of staring at a spinner for several seconds.

**Format:** Server-Sent Events (SSE). Each event is a line beginning with `data: ` followed by a JSON object.

#### Event types

**1 — Chunk events** (many, arrive progressively)

```json
{"chunk": "Most likely this is a"}
{"chunk": " faulty start relay —"}
{"chunk": " does it click when you"}
```

Append each `chunk` string to the displayed message as it arrives. This is all you render to the user during streaming.

**2 — Completion event** (one, always last)

```json
{
  "status": "complete",
  "confidence": 72,
  "safety_flag": false,
  "abstain": false
}
```

Signals the stream is finished. The full assembled text is now the complete AI message. At this point you also receive the metadata fields below — handle them in a way that is meaningful to your UI.

| Field | Type | What it means |
|-------|------|---------------|
| `confidence` | integer 0–100 | How certain the AI is about its current diagnosis. This is expected to be low (30–50) early in the conversation while the AI is still asking questions. It rises as symptoms are confirmed. |
| `safety_flag` | boolean | `true` means this response involves something potentially dangerous (electrical, gas, structural). The AI will have mentioned the risk in its text — but you may want to reinforce it visually. |
| `abstain` | boolean | `true` means the AI declined to answer — either the question is outside technical troubleshooting, or guessing would be dangerous. If `abstain` is `true`, the streamed text will explain why. |

**3 — Error event** (only on failure)

```json
{"error": "Grok API timeout"}
```

The stream failed mid-way. Display an error state; do not add the partial text to `messages[]`.

#### Minimal integration sketch

```swift
// Pseudocode — adapt to your HTTP/SSE library

var assembled = ""

onChunk { chunk in
    assembled += chunk
    renderMessage(assembled)           // update UI live
}

onComplete { event in
    saveToMessages(assembled)          // persist the full text

    // Handle metadata — how you surface these is up to you,
    // but they should be reflected in the UI in some meaningful way
    if event.abstain {
        // AI declined — the streamed text already explains why.
        // Consider replacing or annotating the message bubble.
    }

    if event.safety_flag {
        // AI flagged a safety concern — already mentioned in text.
        // Consider a visual cue (banner, icon, colour) to make sure
        // the user doesn't miss it.
    }

    // confidence is available if you want to show diagnostic progress
    // e.g. a subtle indicator that builds over the conversation.
    // It is intentionally low early — do not alarm the user at turn 1.
    updateConfidenceIndicator(event.confidence)
}

onError { error in
    showErrorState()
    // do not append partial assembled text to messages[]
}
```

> **Note on `abstain`:** If `abstain` is `true`, do not append the assembled text to `messages[]` as a normal assistant turn — it is not a diagnostic response. Save it for display only, or show a dedicated "unable to help" UI state.

---

### Managing the messages[] array

**After every `/enrich` call:**

1. Append `{ role: "assistant", content: response.answer }` to your local messages array
2. On the next user message, append `{ role: "user", content: userInput }`
3. Send the full grown array on the next call

```
Turn 1 request:  messages = [user_1]
Turn 1 response: answer = "..."

Turn 2 request:  messages = [user_1, {role:"assistant", content: turn1_answer}, user_2]
Turn 2 response: answer = "..."

Turn 3 request:  messages = [user_1, assistant_1, user_2, assistant_2, user_3]
```

Never truncate or reset `messages[]` mid-session. The AI's reasoning depends on the full history.

**Handle `abstain: true`:**
```
if response.abstain:
    show: "I'm not confident enough to answer this. Please consult a qualified technician."
else:
    show: response.answer
```

**Handle `safety_flag: true`:**
```
if response.safety_flag:
    show banner: "⚠️ Safety alert — this may involve safety risks. Do not attempt without professional guidance."
```

---

## Endpoint 3 — POST /vision (on demand, mid-conversation)

**Purpose:** Analyze a specific photo. Called when the AI's `answer` indicates it needs to see something — not upfront by default.

**When to call:** When the AI asks for a photo in its answer (e.g., *"Can you take a close-up photo of the compressor unit on the driver's side?"*). After calling `/vision`, inject the result back into `messages` and continue the `/enrich` loop.

### Request

```json
{
  "imageUrl": "https://your-storage.com/photo.jpg",
  "systemPrompt": null,
  "assetContext": {
    "type": "vehicle",
    "make": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": null
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `imageUrl` | string | ✅ | Publicly accessible URL. JPEG, PNG, WebP |
| `systemPrompt` | string \| null | No | Pass `null` for default behaviour |
| `assetContext` | object \| null | No | Same context used in `/enrich` |

### Response

```json
{
  "analysis": "The AC compressor clutch is visibly disengaged. The clutch plate shows wear consistent with slipping. The electromagnetic coil gap appears excessive.",
  "confidence": 84,
  "structured_findings": {
    "part": "AC compressor clutch",
    "condition": "worn",
    "severity": "moderate",
    "observations": [
      "Clutch plate slippage wear marks",
      "Excessive electromagnetic coil gap",
      "Clutch not engaging when AC activated"
    ]
  }
}
```

| Field | Type | Notes |
|-------|------|-------|
| `analysis` | string | Full description of what the AI saw. This is what you inject into `messages` |
| `confidence` | integer 0–100 | How certain the visual analysis is |
| `structured_findings.part` | string | Identified part, or `"unknown"` |
| `structured_findings.condition` | string | `good` \| `worn` \| `damaged` \| `critical` \| `unknown` |
| `structured_findings.severity` | string | `none` \| `minor` \| `moderate` \| `critical` |
| `structured_findings.observations` | string[] | Specific visual observations |

**After calling `/vision` — inject into the conversation:**

```json
{
  "role": "assistant",
  "content": "[Photo] The AC compressor clutch is visibly disengaged. The clutch plate shows wear consistent with slipping. The electromagnetic coil gap appears excessive."
}
```

Append this to `messages`, then call `/enrich` as normal. The AI reads this visual observation as part of the conversation and continues diagnosing.

---

## Endpoint 4 — POST /classify/refine

**Purpose:** Refine the initial asset classification with structured Q&A answers. Primarily used to improve asset record accuracy (mileage, year, model confirmation) — not for the diagnostic conversation itself, which happens through `/enrich`.

**When to call:** If initial `/classify` confidence was low and you need more accurate asset specs in your DB.

### Request

```json
{
  "initialClassification": {
    "assetSpecs": {
      "type": "vehicle",
      "make": "Toyota",
      "model": "Camry",
      "year": 2020,
      "mileage": null
    },
    "condition": "repair_needed",
    "confidence": 55
  },
  "userAnswers": [
    {"q": "What is the approximate mileage?", "a": "45,000 miles"},
    {"q": "Has the AC been serviced before?", "a": "Never"}
  ],
  "skillLevel": 3
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `initialClassification` | object | ✅ | Full response from `/classify` |
| `userAnswers` | object[] | ✅ | At least 1 Q&A pair |
| `userAnswers[].q` | string | ✅ | The question shown to the user |
| `userAnswers[].a` | string | ✅ | The user's answer |
| `skillLevel` | integer 1–10 | No | Defaults to 5 |

### Response

```json
{
  "assetSpecs": {
    "type": "vehicle",
    "make": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": 45000
  },
  "condition": "repair_needed",
  "confidence": 78,
  "reminders": [
    {"title": "AC system service", "due": null}
  ],
  "refined_recommendations": ["AC compressor clutch inspection required"]
}
```

**What to do:**
1. Overwrite `assetSpecs` in your DB with this version (more accurate than `/classify` output)
2. Replace previous reminders with this `reminders` list
3. Use `assetSpecs` as the `assetContext` going forward in `/enrich`

---

## File Handling

### Overview

The server never receives raw file bytes. Files are handled in two ways depending on type:

| File type | What the app does | What the server receives |
|-----------|------------------|--------------------------|
| Image (JPEG, PNG, WebP) | Upload to your CDN → get a URL | A publicly accessible URL string |
| PDF (service manual, receipt, etc.) | Extract text on-device → send as text | Plain text string in a message |

---

### Images

#### 1. Upload to your CDN

The server requires a **publicly accessible URL** — it fetches and analyzes the image itself. The app is responsible for uploading the file first.

Use any cloud storage your backend is set up with (AWS S3, Firebase Storage, Cloudinary, etc.). The URL must be reachable without auth headers.

```swift
// iOS pseudocode
let url = try await uploadToStorage(imageData, mimeType: "image/jpeg")
// → "https://your-cdn.com/photos/abc123.jpg"
```

#### 2. Pass the URL to the correct endpoint

**Mid-conversation photo** (AI asked for it):
```json
POST /vision
{
  "imageUrl": "https://your-cdn.com/photos/abc123.jpg",
  "systemPrompt": null,
  "assetContext": { ... }
}
```
Then inject the vision result into `messages[]` and call `/enrich` (see Endpoint 3 above).

**Photo sent at session start** (user opened with a photo):
```json
POST /enrich/stream
{
  "threadId": "...",
  "imageUrl": "https://your-cdn.com/photos/abc123.jpg",
  "messages": [{ "role": "user", "content": "[User sent a photo: engine-bay.jpg]" }],
  "assetContext": { ... },
  ...
}
```
The server runs vision analysis internally before generating the AI response. No separate `/vision` call needed.

#### Constraints

- **Formats supported:** JPEG, PNG, WebP
- **URL must be public:** No signed URLs or auth-gated endpoints
- **Size:** Keep under 10 MB — larger files may time out at the AI vision layer

---

### PDFs

PDFs are **extracted on-device** and sent as plain text. The server has no PDF parser — it receives text only.

#### 1. Extract text on-device

```swift
// iOS — PDFKit (built-in, no dependency needed)
import PDFKit

func extractText(from url: URL, maxPages: Int = 4, maxChars: Int = 4000) -> String {
    guard let pdf = PDFDocument(url: url) else { return "" }
    var text = ""
    let pages = min(pdf.pageCount, maxPages)
    for i in 0..<pages {
        text += pdf.page(at: i)?.string ?? ""
        if text.count >= maxChars { break }
    }
    return String(text.prefix(maxChars))
}
```

> **Android:** Use `PdfRenderer` (built-in) for rendering, or `iText` / `Apache PdfBox` for text extraction.

#### 2. Send the extracted text

**At session start** — user uploaded a service manual or receipt with their initial message:

```json
POST /classify
{
  "textDescription": "[User uploaded PDF: service-manual.pdf]\nSection 4: AC System...\n<extracted text>",
  "assetId": "asset_abc_123",
  "visionResult": null
}
```

**Mid-conversation** — user uploaded a document in response to an AI question:

Append to `messages[]` as the user's message:
```json
{
  "role": "user",
  "content": "[User uploaded PDF: service-receipt.pdf]\nDate: 2023-06-10\nService: AC recharge, refrigerant topped up...\n<extracted text>"
}
```
Then call `/enrich` with the updated `messages[]`.

#### Constraints

- **Extract first 4 pages maximum** — content beyond that is dropped
- **Cap at 4,000 characters** — truncate before sending
- **Label the content** — always prefix with `[User uploaded PDF: filename.pdf]` so the AI knows this is a document, not typed text

---

### Choosing the right flow at a glance

```
User attaches a file
       │
       ├─ Image?
       │      ├─ Upload to CDN
       │      ├─ Get URL
       │      └─ Pass imageUrl to /vision (mid-conversation)
       │         or imageUrl in /enrich/stream body (session start)
       │
       └─ PDF?
              ├─ Extract text on-device (PDFKit / PdfRenderer)
              ├─ Truncate to 4 pages / 4000 chars
              └─ Send as textDescription (session start → /classify)
                 or as user message content (mid-conversation → /enrich)
```

---

## Error Responses

```json
{"detail": "Error description"}
```

| Status | Meaning | Action |
|--------|---------|--------|
| `422` | Validation error — missing or invalid field | Check required fields. The body names the bad field |
| `500` | Server or AI error | Retry once after 2–3 seconds. If it persists, show a generic error |

---

## skillLevel Reference

| Value | Who | AI behaviour |
|-------|-----|-------------|
| 1–2 | No mechanical knowledge | Plain English, always defers to professional |
| 3–4 | Basic DIYer | Step-by-step for simple jobs, flags advanced steps |
| 5–6 | Comfortable DIYer | Full procedures, torque specs, part numbers |
| 7–8 | Advanced home mechanic | OBD codes, differential diagnosis, specs |
| 9–10 | Professional technician | OEM procedures, TSBs, fault tree analysis |

---

## Files to Ignore

| File | Reason |
|------|--------|
| `README.md` | Server setup for Python engineers |
| `DEVELOPMENT_GUIDE.md` | Internal AI engineer guide |
| `API_INTEGRATION_MAP.md` | Convex integration reference — not needed for mobile |
| `PROJECT_CONTEXT_AI-mechanic.md` | Project brief — internal only |
