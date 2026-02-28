# Production Logging & Monitoring

## Overview

The ARWrench API uses **Rich + JSON logging** for production-grade observability:
- **Console**: Rich formatting for human-readable output (dev/staging)
- **File**: JSON structured logs for analytics and monitoring (production)

## Configuration

### Environment Variables

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR) — default: INFO
export LOG_LEVEL=INFO

# Log output format (rich, json) — default: rich
export LOG_FORMAT=rich

# Optional: Write JSON logs to file for production monitoring
export LOG_FILE=/var/log/arwrench/api.log
```

## Log Structure

### Console Output (Rich)
```
[2026-02-25 10:45:32] → POST /enrich
[2026-02-25 10:45:33] ✓ grok | https://api.x.ai/v1/chat/completions | 200 | 1250ms
[2026-02-25 10:45:33] ✓ POST /enrich | 200 | 1500ms
```

### File Output (JSON)
```json
{
  "timestamp": "2026-02-25T10:45:32.123Z",
  "level": "INFO",
  "name": "arwrench",
  "message": "→ POST /enrich",
  "request_id": "a1b2c3d4",
  "module": "main",
  "funcName": "dispatch",
  "lineno": 45
}
```

## Tracking Operations

### Request-Scoped Context

The `LogContext` automatically injects metadata into all logs within a request:

```python
# Automatically available in all logs for this request
LogContext.set(
    request_id="a1b2c3d4",
    thread_id="thread_567",
    asset_id="asset_890"
)
```

### Step Logging

Track what worked and how:

```python
from log_utils import log_step, log_result

# Log individual steps
log_step("pdf_ingestion_started", asset_id="motor_1", file_size_mb=12.5)
log_step("chunks_extracted", status="success", chunk_count=45)

# Log final result with duration
log_result(
    "classify_success",
    success=True,
    duration_ms=1250,
    confidence=0.92,
    asset_type="vehicle"
)
```

### API Call Tracking

All external API calls (Grok, Gemini, Chroma) are logged:

```
✓ GROK | https://api.x.ai/v1/chat/completions | 200 | 1250ms
✓ GEMINI | https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent | 200 | 850ms
✗ OPENAI | https://api.openai.com/v1/embeddings | 429 | 200ms
```

## Monitoring in Production

### What to Monitor

1. **API Latency**: `duration_ms` field
   - Enrich endpoint: target < 2000ms
   - Vision endpoint: target < 1500ms
   - Classify endpoint: target < 3000ms

2. **Error Rates**: Track `"success": false` logs
   - High Grok error rate → check API quota/keys
   - High Gemini error rate → check image formats
   - Parse failures → check Grok output format

3. **Confidence Scores**: For audit trails
   - Symptom match, history alignment, specificity
   - Safety flags on critical operations

4. **Token Usage**: Implicit in API calls
   - Monitor for runaway costs on Grok/Gemini

### Log Analysis (using jq)

```bash
# Watch real-time errors
tail -f /var/log/arwrench/api.log | jq 'select(.level=="ERROR")'

# Count errors by service
cat api.log | jq 'select(.service) | .service' | sort | uniq -c

# Find slow operations (> 2000ms)
cat api.log | jq 'select(.duration_ms > 2000)'

# Track confidence scores
cat api.log | jq 'select(.operation=="enrich_success") | .result.confidence' | sort -n
```

## Integration with Monitoring Tools

### Datadog / Splunk / CloudWatch

Point your aggregator to `/var/log/arwrench/api.log`:

```yaml
# Example Datadog agent config
logs:
  - type: file
    path: /var/log/arwrench/api.log
    service: arwrench-api
    source: fastapi
    parser: json
```

### Metrics to Track

- **API Endpoints**
  - `POST /enrich`: latency, error rate, confidence distribution
  - `POST /vision`: latency, error rate, detection accuracy
  - `POST /classify`: latency, error rate, confidence scores

- **External Services**
  - Grok API: call count, latency, error rate, token spend
  - Gemini Vision: call count, latency, error rate
  - Chroma RAG: retrieval latency, chunk relevance

## Troubleshooting

### Missing logs?
1. Check `LOG_LEVEL` — ensure not set to ERROR
2. Check `LOG_FILE` — ensure directory exists and is writable
3. Check application logs — exceptions in logger setup will appear in console

### Too many logs?
Set `LOG_LEVEL=WARNING` to reduce noise in production.

### Need to add new operation?

```python
from log_utils import log_operation

@log_operation("new_feature")
async def my_feature():
    # Automatically logs start, end, duration, exceptions
    pass
```

## Key Differences from Old Logging

| Old | New |
|-----|-----|
| Simple format strings | Structured JSON logs |
| Scattered logger.warning() calls | Consistent log_step() / log_result() |
| No context tracking | Automatic request_id, asset_id injection |
| No API call visibility | Full Grok/Gemini/Chroma logging |
| No duration tracking | Automatic timing on all operations |
