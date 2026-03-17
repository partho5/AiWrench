# ARWrench AI Enrichment API — Server Setup Guide

**This is the Python/FastAPI server.** For integration docs, see:
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) — AI engineer reference (design intent, gaps, prompts)
- [API_INTEGRATION_MAP.md](API_INTEGRATION_MAP.md) — Convex integration (how to wire endpoints)
- [MOBILE_CLIENT_GUIDE.md](MOBILE_CLIENT_GUIDE.md) — Mobile/Convex developer reference (schemas, flow)

---

**What this is:** FastAPI layer that owns all AI reasoning for the ARWrench "mechanic in your pocket" app. The AI drives the diagnostic conversation — asking the right questions, requesting photos when needed, and declaring findings when it has enough confidence. The iOS app and Convex backend are pure relays; this project is the brain.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Architecture](#architecture)
- [Modifying Features](#modifying-features)
- [Adding New Endpoints](#adding-new-endpoints)
- [Logging & Monitoring](#logging--monitoring)
- [Testing](#testing)
- [Deployment](#deployment)

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd ArWrench-Api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# activate
source venv/bin/activate
# Start development server
uvicorn main:app --reload
```

Server runs on `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## Prerequisites

- **Python 3.10+**
- **pip** or **conda**
- **Grok API key** (from x.ai) — used for all AI calls including vision
- **Chroma** (optional, for local RAG embeddings)

## Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd ArWrench-Api
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- **fastapi** — Web framework
- **uvicorn** — ASGI server
- **pydantic** — Data validation
- **httpx** — Async HTTP client
- **python-json-logger** — JSON structured logging
- **langchain** — LLM orchestration
- **chromadb** — Vector database for RAG
- **pypdf** — PDF processing

### 4. Verify Installation

```bash
python -c "import fastapi, pydantic, httpx; print('All imports OK')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
GROK_API_KEY=xai-...
GROK_MODEL=grok-3-mini

# Logging
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/arwrench.log  # Production: /var/log/arwrench/api.log

# Server
HOST=0.0.0.0
PORT=8000

# Optional: RAG/Chroma
CHROMA_COLLECTION=arwrench_docs
```

### Environment Variable Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROK_API_KEY` | (required) | Grok API authentication — used for all AI calls |
| `GROK_MODEL` | `grok-3-mini` | Grok model for text endpoints |
| `GROK_VISION_MODEL` | `grok-2-vision-1212` | Grok model for `/vision` endpoint |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_FILE` | `logs/arwrench.log` | Log file path |

## Running the Server

### Development

```bash
# Auto-reload on file changes
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Single worker
uvicorn main:app --host 0.0.0.0 --port 8000

# Multiple workers (with Gunicorn)
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### Verify Server is Running

```bash
curl http://localhost:8000/health
# Response: {"status": "ok", "service": "ARWrench AI Enrichment API", "version": "1.0.0"}
```

## API Endpoints

### 1. Health Check

```
GET /health
```

Returns service status. Use for load balancer health checks.

**Response:**
```json
{
  "status": "ok",
  "service": "ARWrench AI Enrichment API",
  "version": "1.0.0"
}
```

### 2. Enrich Endpoint

```
POST /enrich
```

Receives chat history + asset context, enriches the prompt, calls Grok, scores confidence.

**Request:**
```json
{
  "threadId": "thread_123",
  "messages": [
    {"role": "user", "content": "My car is making a grinding noise"},
    {"role": "assistant", "content": "What does it sound like..."}
  ],
  "assetContext": {
    "type": "vehicle",
    "make": "Honda",
    "model": "Civic",
    "year": 2020,
    "mileage": 95000
  },
  "toolbox": ["wrench", "screwdriver"],
  "serviceHistory": [
    {"date": "2024-01-15", "service": "Oil change", "notes": "5W-30"}
  ],
  "skillLevel": 5
}
```

**Response:**
```json
{
  "answer": "The grinding noise likely indicates...",
  "confidence": 82,
  "confidence_reasons": [
    "Symptom match: 85/100",
    "History alignment: 80/100",
    "Specificity: 81/100"
  ],
  "abstain": false,
  "safety_flag": false,
  "role": "assistant"
}
```

**Skill Levels (1–10 user knowledge scale):**
- `1–2`: No mechanical knowledge — plain English, always defer to professional
- `3–4`: Basic DIYer — step-by-step for simple jobs, flags anything complex
- `5–6`: Comfortable DIYer (default: 5) — standard repairs, full procedures, torque specs
- `7–8`: Advanced home mechanic — OBD codes, differential diagnosis, component specs
- `9–10`: Professional technician — OEM procedures, TSBs, fault tree analysis

### 3. Vision Endpoint

```
POST /vision
```

Analyzes vehicle images. Returns condition assessment and structured findings.

**Request:**
```json
{
  "imageUrl": "https://example.com/vehicle.jpg",
  "systemPrompt": "Analyze this vehicle's condition...",
  "assetContext": {
    "type": "vehicle",
    "make": "Ford",
    "model": "F-150"
  }
}
```

**Response:**
```json
{
  "analysis": "The vehicle shows moderate wear...",
  "confidence": 75,
  "structured_findings": {
    "part": "exterior",
    "condition": "fair",
    "severity": "moderate",
    "observations": ["paint peeling", "rust on bumper"]
  }
}
```

### 4. Classify Endpoint

```
POST /classify
```

Classifies asset condition and generates recommendations based on vision + text description.

**Request:**
```json
{
  "visionResult": "moderate wear detected",
  "textDescription": "Engine starts but runs rough, needs tune-up",
  "assetId": "asset_456"
}
```

**Response:**
```json
{
  "assetSpecs": {
    "type": "vehicle",
    "make": "Toyota",
    "model": "Corolla",
    "year": 2018
  },
  "condition": "fair",
  "confidence": 78,
  "recommendations": [
    "Change spark plugs",
    "Check fuel injectors",
    "Inspect ignition system"
  ],
  "reminders": []
}
```

### 5. Classify Refine Endpoint

```
POST /classify/refine
```

Refines initial classification based on user answers to clarifying questions.

**Request:**
```json
{
  "initialClassification": {
    "condition": "fair",
    "confidence": 78
  },
  "userAnswers": [
    {"q": "How long has this been happening?", "a": "About 2 weeks"},
    {"q": "Any other symptoms?", "a": "Slight vibration at idle"}
  ],
  "skillLevel": 5
}
```

**Response:**
```json
{
  "assetSpecs": {...},
  "condition": "fair",
  "confidence": 85,
  "reminders": [],
  "refined_recommendations": [...]
}
```

## Architecture

### Project Structure

```
ArWrench-Api/
├── main.py                 # FastAPI app, middleware, entry point
├── models.py               # Pydantic request/response schemas
├── logger_config.py        # Logging setup (JSON to file)
├── log_utils.py            # Logging utilities (decorators, context managers)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── routers/
│   ├── enrich.py          # POST /enrich endpoint
│   ├── vision.py          # POST /vision endpoint
│   └── classify.py        # POST /classify endpoints
├── services/
│   ├── grok_client.py     # Grok API wrapper
│   ├── grok_vision_client.py  # Grok Vision API wrapper
│   └── rag_retriever.py   # Chroma RAG integration
├── logs/                   # Runtime log files
└── tests/
    └── test_*.py          # Integration tests
```

### Request Flow

```
Client Request
    ↓
RequestLoggingMiddleware (logs incoming request, generates request_id)
    ↓
Router Handler (e.g., /enrich)
    ├─→ LogContext.set() — inject request metadata
    ├─→ log_step() — track operation steps
    ├─→ External API Calls (Grok / Grok Vision)
    ├─→ log_api_call() — track API metrics
    ├─→ log_result() — log final outcome
    ↓
Response + Log Entry (JSON to logs/arwrench.log)
```

### Data Models

Key models in `models.py`:

- **EnrichRequest** — Chat history + asset context for enrichment
- **EnrichResponse** — Diagnosis + confidence score
- **VisionRequest** — Image URL + context
- **VisionResponse** — Image analysis + findings
- **ClassifyRequest** — Vision result + text description
- **ClassifyResponse** — Asset specs + recommendations

## Modifying Features

### Changing Skill Level Behavior

In `routers/enrich.py`:

```python
SKILL_LEVEL_PREFIXES: dict[int, str] = {
    1: "Your updated novice persona prompt...",
    2: "Your updated basic persona prompt...",
    # etc — 10 levels total
}

GROK_LLM_TEMPERATURE: dict[int, float] = {
    1: 0.2,  # More deterministic for novice
    2: 0.3,
    # etc
}
```

Then restart the server.

### Updating Confidence Scoring

In `routers/enrich.py`, function `compute_confidence()`:

```python
def compute_confidence(parsed: dict) -> tuple[int, list[str]]:
    """Modify scoring logic here."""
    symptom_match = _clamp(int(parsed.get("symptom_match", 50)))
    history_alignment = _clamp(int(parsed.get("history_alignment", 50)))
    specificity = _clamp(int(parsed.get("specificity", 50)))

    # Change how confidence is calculated
    score = _clamp((symptom_match + history_alignment + specificity) // 3)

    # Add or remove scoring factors
    reasons = [
        f"Symptom match: {symptom_match}/100",
        # Add new factors here
    ]

    return score, reasons
```

### Modifying External API Calls

**Grok call** in `services/grok_client.py`:

```python
async def call_grok(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,  # Increase/decrease response length
) -> str:
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Add new parameters here (e.g., top_p, frequency_penalty)
    }
```

**Grok Vision call** in `services/grok_vision_client.py`:

```python
async def analyze_image(image_url: str, prompt: str) -> str:
    # Modify model via GROK_VISION_MODEL env var
    # Downloads image, encodes base64, sends to grok-2-vision-1212
```

### Changing Log Output

JSON logs are written to `logs/arwrench.log`. To change:

1. **Log level** — Set `LOG_LEVEL=DEBUG` for verbose output
2. **Log file path** — Set `LOG_FILE=/path/to/file.log`
3. **Log format** — Edit `logger_config.py`, `JsonFormatter` format string

In `logger_config.py`:

```python
file_handler.setFormatter(
    JsonFormatter(
        fmt="%(timestamp)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d",
        timestamp=True,
    )
)
```

### Adding Custom Log Fields

In any endpoint:

```python
from logger_config import LogContext
from log_utils import log_step

@router.post("/my-endpoint")
async def my_endpoint(req: MyRequest):
    # Inject custom context that appears in all logs for this request
    LogContext.set(custom_field="value", user_id=req.user_id)

    log_step("operation_started", extra_data=123)
```

## Adding New Endpoints

### 1. Create Request/Response Models

In `models.py`:

```python
class MyRequest(BaseModel):
    field1: str
    field2: int = Field(default=10, ge=1, le=100)

class MyResponse(BaseModel):
    result: str
    status: int
```

### 2. Create Router Module

Create `routers/my_feature.py`:

```python
from fastapi import APIRouter, HTTPException
from logger_config import logger, LogContext
from log_utils import log_step, log_result
from models import MyRequest, MyResponse

router = APIRouter()

@router.post("/my-feature", response_model=MyResponse)
async def my_feature(req: MyRequest):
    LogContext.set(feature="my_feature")

    try:
        log_step("feature_started", field1=req.field1)

        # Your business logic here
        result = await process_request(req)

        log_result("feature_success", success=True, result=result)
        return MyResponse(result=result, status=200)

    except Exception as e:
        logger.exception(f"Feature failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal error")

async def process_request(req: MyRequest) -> str:
    # Implementation
    return "result"
```

### 3. Register Router

In `main.py`:

```python
from routers import enrich, vision, classify, my_feature

app.include_router(my_feature.router, tags=["My Feature"])
```

### 4. Test the Endpoint

```bash
curl -X POST http://localhost:8000/my-feature \
  -H "Content-Type: application/json" \
  -d '{"field1": "value", "field2": 50}'
```

## Logging & Monitoring

### Log Files

Logs are written to `logs/arwrench.log` as JSON. Each line is a complete JSON object:

```json
{"timestamp": "2026-02-25T10:45:32.123Z", "level": "INFO", "name": "arwrench.enrich", "message": "→ POST /enrich", "request_id": "a1b2c3d4"}
{"timestamp": "2026-02-25T10:45:33.234Z", "level": "INFO", "name": "arwrench.enrich", "message": "enrich_started", "thread_id": "thread_123", "asset_type": "vehicle"}
```

### Querying Logs

```bash
# View errors
cat logs/arwrench.log | jq 'select(.level == "ERROR")'

# Find slow operations (> 2000ms)
cat logs/arwrench.log | jq 'select(.duration_ms > 2000)'

# Track API call latencies
cat logs/arwrench.log | jq 'select(.service == "grok") | .duration_ms'

# Count errors by endpoint
cat logs/arwrench.log | jq 'select(.level == "ERROR") | .path' | sort | uniq -c
```

### Log Rotation

Files rotate automatically:
- Max size per file: 10MB
- Max backup files: 5
- Total storage: ~60MB

Old files are archived as `arwrench.log.1`, `arwrench.log.2`, etc.

### Production Monitoring Integration

Point your monitoring tool to `logs/arwrench.log`:

```yaml
# Datadog
logs:
  - type: file
    path: /var/log/arwrench/api.log
    service: arwrench-api
    source: fastapi
    parser: json
```

See [LOGGING.md](./LOGGING.md) for detailed monitoring setup.

## Testing

### Run Integration Tests

```bash
pytest tests/ -v
```

### Run Specific Test

```bash
pytest tests/test_enrich.py::test_enrich_success -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Testing an Endpoint Manually

```bash
# Test /enrich
curl -X POST http://localhost:8000/enrich \
  -H "Content-Type: application/json" \
  -d '{
    "threadId": "test_123",
    "messages": [{"role": "user", "content": "test"}],
    "assetContext": {"type": "vehicle", "make": "Honda"},
    "skillLevel": 5
  }'
```

## Deployment

### Docker (Recommended)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LOG_FILE=/var/log/arwrench/api.log
RUN mkdir -p /var/log/arwrench

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t arwrench-api .
docker run -p 8000:8000 \
  -e GROK_API_KEY=$GROK_API_KEY \
  -v /var/log/arwrench:/var/log/arwrench \
  arwrench-api
```

### Kubernetes

Use the Docker image with a Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arwrench-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arwrench-api
  template:
    metadata:
      labels:
        app: arwrench-api
    spec:
      containers:
      - name: api
        image: arwrench-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: GROK_API_KEY
          valueFrom:
            secretKeyRef:
              name: arwrench-secrets
              key: grok-key
        volumeMounts:
        - name: logs
          mountPath: /var/log/arwrench
      volumes:
      - name: logs
        emptyDir: {}
```

### Health Checks

Configure your load balancer to hit:

```
GET /health
```

Expected response:
```json
{"status": "ok", "service": "ARWrench AI Enrichment API", "version": "1.0.0"}
```

### Performance Tuning

1. **Increase workers** — Use Gunicorn with `--workers 4`
2. **Optimize timeouts** — Increase Grok timeout in `services/grok_client.py` if needed
3. **Enable caching** — Add Redis for response caching (future enhancement)
4. **Monitor resources** — Track API latency and error rates in logs

## Troubleshooting

### API Key Errors

```
ValueError: GROK_API_KEY not found in environment
```

**Solution**: Ensure `.env` file is created and `GROK_API_KEY` is set.

```bash
export GROK_API_KEY=xai-...
```

### Port Already in Use

```
OSError: [Errno 48] Address already in use
```

**Solution**: Change port or kill existing process.

```bash
uvicorn main:app --port 9000
# Or
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Slow API Responses

**Solution**: Check logs for external API latency.

```bash
cat logs/arwrench.log | jq 'select(.service == "grok") | {duration_ms, timestamp}' | tail -20
```

### Missing Logs

**Solution**: Verify log file path and permissions.

```bash
ls -la logs/
# Should see: arwrench.log with write permissions
```

## Contributing

1. Create a new branch: `git checkout -b feature/my-feature`
2. Make changes and test: `pytest tests/`
3. Commit with clear message: `git commit -m "Add new feature"`
4. Push and open a pull request

## License

Proprietary — ARWrench, Inc.

## Support

For issues or questions:
- Check logs: `logs/arwrench.log`
- Review API docs: `http://localhost:8000/docs`
- See [LOGGING.md](./LOGGING.md) for monitoring setup
