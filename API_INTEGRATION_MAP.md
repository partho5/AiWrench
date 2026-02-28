# ARWrench API — Integration Map

For Convex developers connecting the iOS app to this FastAPI layer.

---

## The Split

**FastAPI owns all AI.** Convex gathers data from its DB, calls FastAPI, saves the result.

```
iOS App (UI only — renders what it receives)
    ↓
Convex action: gather chatMessages, asset, serviceRecords, toolbox from DB
    ↓
Convex calls FastAPI (one HTTP POST per turn)
    ↓
FastAPI: enriches context → calls Grok internally → returns final answer
    ↓
Convex saves {answer, confidence, ...} to chatMessages
    ↓
iOS real-time subscription delivers the answer automatically
```

**Convex responsibility:** data storage, auth, real-time sync, scheduling, DB queries.
**FastAPI responsibility:** all AI reasoning — prompt building, Grok/Vision calls, confidence scoring.
**iOS responsibility:** display what it receives. No diagnostic logic lives in the app.

---

## Design Rule: The AI Drives

The AI mechanic asks questions, requests photos, and declares diagnosis — the app never decides these things. Convex's job is to faithfully relay the full conversation history on every call and save whatever comes back.

The `messages[]` array is the single shared state of the session. It must be passed in full on every `/enrich` call and grown correctly after each response.

---

## The 5 Convex Fetch() Calls to Replace

| File | Lines | Remove | Replace with |
|------|-------|--------|--------------|
| `grokChat.ts` | 426–441 | `fetch("https://api.x.ai/...")` | `fetch(FASTAPI_URL/enrich)` |
| `grokVision.ts` | 88–101 | `fetch("https://generativelanguage...")` | `fetch(FASTAPI_URL/vision)` |
| `grokVision.ts` | 149–161 | `fetch("https://api.x.ai/...")` | `fetch(FASTAPI_URL/vision)` |
| `classifyAsset.ts` | 131–146 | `fetch("https://api.x.ai/...")` | `fetch(FASTAPI_URL/classify)` |
| `classifyAsset.ts` | 207–222 | `fetch("https://api.x.ai/...")` | `fetch(FASTAPI_URL/classify/refine)` |

Add to Convex constants:
```typescript
const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";
```

---

## Endpoint 1 — POST /enrich (the conversation engine)

**Role in the flow:** Every turn of the diagnostic conversation. This is the core loop — called once per user message from session open to close.

**What Convex must pass:** The full `messages[]` history every time. Never truncate it mid-session.

### Convex call

```typescript
// In grokChat.ts — replaces the Grok fetch() entirely

const messages = await db
  .query("chatMessages")
  .withIndex("by_thread", q => q.eq("threadId", threadId))
  .order("asc")
  .collect();

const thread = await db.get(threadId);
const asset  = await db.get(thread.assetId);
const history = await db
  .query("serviceRecords")
  .withIndex("by_asset", q => q.eq("assetId", asset._id))
  .collect();

const res = await fetch(`${FASTAPI_URL}/enrich`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    threadId,
    messages: messages.map(m => ({ role: m.role, content: m.content })),
    assetContext: {
      type:    asset.type,
      make:    asset.make,
      model:   asset.model,
      year:    asset.year,
      mileage: asset.mileage ?? null,
    },
    toolbox:        thread.toolbox ?? [],
    serviceHistory: history.map(r => ({ date: r.date, service: r.service })),
    skillLevel:     thread.skillLevel ?? 5,
  }),
});

const result = await res.json();

// Save AI answer to chatMessages
await ctx.runMutation(internal.chatMessages.saveAIResponse, {
  threadId,
  content:    result.answer,
  confidence: result.confidence,
  safetyFlag: result.safety_flag,
  role: "assistant",
});
```

### Request schema

```json
{
  "threadId": "thread_abc_001",
  "messages": [
    {"role": "user",      "content": "My AC stopped working, makes a humming noise"},
    {"role": "assistant", "content": "How long has this been happening?"},
    {"role": "user",      "content": "Since yesterday, only when AC is on"}
  ],
  "assetContext": {
    "type": "vehicle",
    "make": "Toyota",
    "model": "Camry",
    "year": 2020,
    "mileage": null
  },
  "toolbox": [],
  "serviceHistory": [
    {"date": "2023-06-10", "service": "Oil change"}
  ],
  "skillLevel": 3
}
```

### Response schema

```json
{
  "answer":   "Does the humming stop the moment you switch from AC to fan-only mode?",
  "confidence": 40,
  "confidence_reasons": [
    "Symptom match: 50/100",
    "History alignment: 25/100",
    "Specificity: 45/100"
  ],
  "abstain":    false,
  "safety_flag": false,
  "role": "assistant"
}
```

**What Convex saves:** `result.answer` as a new chatMessage with `role: "assistant"`.
On the next user message, query chatMessages again — the new assistant message is already in DB — and send the full updated array to `/enrich`.

---

## Endpoint 2 — POST /vision (on demand, mid-conversation)

**Role in the flow:** Analyze a specific photo when the AI's answer asks for one. Not a fixed first step — called at any point in the conversation when needed.

**What Convex must do after:** Inject the vision analysis into `chatMessages` as an assistant message, then trigger the next `/enrich` turn.

### Convex call

```typescript
// In grokVision.ts — replaces Gemini + Grok vision fetches

const res = await fetch(`${FASTAPI_URL}/vision`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    imageUrl:     imageUrl,
    systemPrompt: null,
    assetContext: {
      type:  asset.type,
      make:  asset.make,
      model: asset.model,
    },
  }),
});

const result = await res.json();

// Inject vision analysis as an assistant message in the thread
await ctx.runMutation(internal.chatMessages.saveAIResponse, {
  threadId,
  content: `[Photo] ${result.analysis}`,
  role:    "assistant",
});

// Then schedule the next /enrich turn so the AI continues with visual context
await ctx.scheduler.runAfter(0, internal.grokChat.respond, { threadId });
```

### Request schema

```json
{
  "imageUrl":     "https://storage.url/photo.jpg",
  "systemPrompt": null,
  "assetContext": {
    "type":  "vehicle",
    "make":  "Toyota",
    "model": "Camry"
  }
}
```

### Response schema

```json
{
  "analysis": "The AC compressor clutch is visibly disengaged. Clutch plate shows slippage wear. Coil gap appears excessive.",
  "confidence": 84,
  "structured_findings": {
    "part":      "AC compressor clutch",
    "condition": "worn",
    "severity":  "moderate",
    "observations": [
      "Clutch plate slippage wear marks",
      "Excessive electromagnetic coil gap"
    ]
  }
}
```

**What Convex saves:** `"[Photo] " + result.analysis` as a chatMessage with `role: "assistant"`.
Then triggers the next `/enrich` call — the AI reads the injected photo analysis as part of the conversation.

---

## Endpoint 3 — POST /classify (session start)

**Role in the flow:** Called once at session start. Identifies the asset and returns structured specs for the DB. The diagnostic conversation itself happens through `/enrich`.

### Convex call

```typescript
// In classifyAsset.ts — replaces the classify Grok fetch()

const res = await fetch(`${FASTAPI_URL}/classify`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    visionResult:    visionAnalysis ?? null,
    textDescription: userDescription,
    assetId:         asset._id,
  }),
});

const result = await res.json();

// Save identified asset specs to DB
await ctx.runMutation(internal.assets.updateSpecs, {
  assetId:   asset._id,
  type:      result.assetSpecs.type,
  make:      result.assetSpecs.make,
  model:     result.assetSpecs.model,
  year:      result.assetSpecs.year,
  mileage:   result.assetSpecs.mileage,
  condition: result.condition,
});

// Save suggested reminders
for (const reminder of result.reminders) {
  await ctx.runMutation(internal.reminders.create, {
    assetId: asset._id,
    title:   reminder.title,
    dueDate: reminder.dueDate,
  });
}
```

### Request schema

```json
{
  "visionResult":    null,
  "textDescription": "My car AC stopped working. Makes a loud humming when I turn it on.",
  "assetId":         "asset_abc_123"
}
```

### Response schema

```json
{
  "assetSpecs": {
    "type":    "vehicle",
    "make":    "Toyota",
    "model":   "Camry",
    "year":    2020,
    "mileage": null
  },
  "condition":       "repair_needed",
  "confidence":      55,
  "recommendations": ["AC system inspection required"],
  "reminders": [
    {"title": "Schedule AC service", "dueDate": null}
  ]
}
```

**What Convex saves:** `assetSpecs` → assets table. `reminders` → reminders table.

---

## Endpoint 4 — POST /classify/refine (asset record improvement)

**Role in the flow:** Refines asset specs when initial `/classify` confidence is low. Improves the asset DB record — the diagnostic conversation still happens through `/enrich`.

### Convex call

```typescript
// In classifyAsset.ts — replaces the refine Grok fetch()

const res = await fetch(`${FASTAPI_URL}/classify/refine`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    initialClassification: initialClassifyResult,
    userAnswers:           userQAAnswers,
    skillLevel:            thread.skillLevel ?? 5,
  }),
});

const result = await res.json();

// Overwrite asset specs with the more accurate version
await ctx.runMutation(internal.assets.updateSpecs, {
  assetId:  asset._id,
  ...result.assetSpecs,
  condition: result.condition,
});

// Replace reminders with refined list
await ctx.runMutation(internal.reminders.replaceForAsset, {
  assetId:   asset._id,
  reminders: result.reminders,
});
```

### Request schema

```json
{
  "initialClassification": {
    "assetSpecs":  {"type": "vehicle", "make": "Toyota", "model": "Camry", "year": 2020, "mileage": null},
    "condition":   "repair_needed",
    "confidence":  55
  },
  "userAnswers": [
    {"q": "What is the approximate mileage?", "a": "45,000 miles"},
    {"q": "Has the AC ever been serviced?",   "a": "Never"}
  ],
  "skillLevel": 3
}
```

### Response schema

```json
{
  "assetSpecs": {
    "type": "vehicle", "make": "Toyota", "model": "Camry",
    "year": 2020, "mileage": 45000
  },
  "condition":             "repair_needed",
  "confidence":            78,
  "reminders":             [{"title": "AC system service", "due": null}],
  "refined_recommendations": ["AC compressor clutch inspection required"]
}
```

---

## Error Handling

FastAPI returns standard HTTP errors:

```json
{"detail": "Error description"}
```

| Status | Meaning | Convex action |
|--------|---------|---------------|
| `422` | Bad request — missing or invalid field | Log, surface error to user |
| `500` | AI or server failure | Retry once after 2–3s. If persists, show generic error in chat |

**Never fall back to calling Grok directly from Convex on FastAPI failure.** Surface the error instead.

---

## Health Check

```
GET /health
→ {"status": "ok", "service": "ARWrench AI Enrichment API", "version": "1.0.0"}
```

Use this for load balancer health checks and deploy verification.

---

## Deployment

| Environment | FastAPI URL |
|-------------|-------------|
| Development | `http://localhost:8000` |
| Production  | Set `FASTAPI_URL` env var in Convex to your deployed URL (Render, Railway, etc.) |
