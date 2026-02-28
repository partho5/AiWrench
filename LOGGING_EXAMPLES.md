# Logging Usage Examples

## Quick Start

### In any Python file:

```python
from logger_config import logger
from log_utils import log_step, log_result, log_operation, log_context

# Basic logging
logger.info("Something happened")
logger.error("Something went wrong")

# Structured step tracking
log_step("pdf_loaded", file_size_mb=12.5, chunk_count=45)
log_result("classify_success", success=True, duration_ms=1250, confidence=0.92)
```

## Real-World Examples

### Tracking an API enrichment flow:

```python
@router.post("/enrich")
async def enrich(req: EnrichRequest):
    LogContext.set(thread_id=req.threadId, asset_type=req.assetContext.type)

    try:
        log_step("enrich_started", temperature_level=req.temperature)

        # Process messages
        recent_messages, summary = summarize_messages(req.messages)
        log_step("messages_summarized",
                 recent=len(recent_messages),
                 older_summary_chars=len(summary))

        # Call Grok
        raw_response = await call_grok(system_prompt, recent_messages)
        log_step("grok_called", response_length=len(raw_response))

        # Parse JSON
        parsed = parse_json_response(raw_response)
        if parsed.get("_parse_failed"):
            log_step("json_parse_failed", status="warning")

        # Compute confidence
        confidence, reasons = compute_confidence(parsed)

        # Final result
        log_result("enrich_complete",
                   success=True,
                   confidence=confidence,
                   safety_flag=bool(parsed.get("safety_flag")),
                   has_abstain=bool(parsed.get("abstain")))

        return EnrichResponse(...)

    except Exception as e:
        log_result("enrich_failed", success=False, error=str(e))
        raise
```

### Monitoring external API calls:

```python
# In services/grok_client.py
from log_utils import log_api_call

async def call_grok(system_prompt, messages, temperature=0.7):
    start_time = time.time()

    try:
        response = await client.post(GROK_API_URL, json=payload)
        duration_ms = (time.time() - start_time) * 1000

        log_api_call(
            service="grok",
            endpoint=GROK_API_URL,
            request_body={"model": GROK_MODEL, "messages_count": len(messages)},
            response_status=response.status_code,
            response_body={"tokens_used": response.json().get("usage", {})},
            duration_ms=duration_ms
        )

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_api_call(
            service="grok",
            endpoint=GROK_API_URL,
            response_status=500,
            duration_ms=duration_ms,
            error=str(e)
        )
        raise
```

### Using context managers for nested operations:

```python
from log_utils import log_context

async def classify_with_rag(asset_id, text):
    with log_context("classify_with_rag", asset_id=asset_id) as context_id:
        # Retrieve RAG docs
        with log_context("rag_retrieval", asset_id=asset_id):
            docs = await retrieve_relevant_chunks(text, asset_id)
            logger.info(f"Retrieved {len(docs)} chunks")

        # Call Grok
        with log_context("grok_call"):
            response = await call_grok(system_prompt, messages)

        return parse_json_response(response)

# Output:
# → ENTER | classify_with_rag
#   → ENTER | rag_retrieval
#   Retrieved 5 chunks
#   ← EXIT (OK) | rag_retrieval | 125ms
#   → ENTER | grok_call
#   ← EXIT (OK) | grok_call | 1250ms
# ← EXIT (OK) | classify_with_rag | 1500ms
```

### Using decorators for complete operation tracking:

```python
from log_utils import log_operation

@log_operation("classify_asset")
async def classify_asset(asset_id, description):
    # Automatically logs:
    # ▶ START | classify_asset
    # [... your code ...]
    # ✓ DONE | classify_asset | 1250ms

    rag_context = await retrieve_relevant_chunks(description, asset_id)
    response = await call_grok(system_prompt, messages)
    return parse_json_response(response)
```

## Log Output Examples

### Console (Rich formatting):

```
[2026-02-25 10:45:32] INFO     → POST /enrich
[2026-02-25 10:45:33] INFO     [INFO] enrich_started  {'temperature_level': 3}
[2026-02-25 10:45:33] INFO     [INFO] messages_summarized  {'recent': 5, 'older_summary_chars': 0}
[2026-02-25 10:45:33] INFO     ✓ GROK | https://api.x.ai/v1/chat/completions | 200 | 1250ms
[2026-02-25 10:45:33] INFO     [INFO] grok_called  {'response_length': 1524}
[2026-02-25 10:45:33] INFO     ✓ SUCCESS | enrich_complete | 1500ms
[2026-02-25 10:45:33] INFO     ✓ POST /enrich | 200 | 1500ms
```

### File (JSON for analysis):

```json
{"timestamp": "2026-02-25T10:45:32.123Z", "level": "INFO", "name": "arwrench.enrich", "message": "→ POST /enrich", "request_id": "a1b2c3d4"}
{"timestamp": "2026-02-25T10:45:32.234Z", "level": "INFO", "name": "arwrench.enrich", "message": "[INFO] enrich_started", "metadata": {"temperature_level": 3}}
{"timestamp": "2026-02-25T10:45:33.485Z", "level": "INFO", "name": "arwrench.grok_client", "message": "✓ GROK | ... | 200 | 1250ms", "api_call": true, "service": "grok", "duration_ms": 1250}
{"timestamp": "2026-02-25T10:45:33.500Z", "level": "INFO", "name": "arwrench.enrich", "message": "✓ SUCCESS | enrich_complete | 1500ms", "operation": "enrich_complete", "result": {"confidence": 85, "safety_flag": true}}
```

## Monitoring Queries

### Find all failed enrich operations:

```bash
cat api.log | jq 'select(.message | contains("enrich")) | select(.level=="ERROR")'
```

### Get average Grok latency:

```bash
cat api.log | jq 'select(.service=="grok") | .duration_ms' | \
  awk '{sum+=$1; count++} END {print "Avg: " sum/count "ms"}'
```

### Track confidence distribution:

```bash
cat api.log | jq 'select(.operation=="enrich_complete") | .result.confidence' | sort
```

### Find slow operations (> 2s):

```bash
cat api.log | jq 'select(.duration_ms > 2000)'
```

## Production Deployment

### Environment configuration:

```bash
export LOG_LEVEL=INFO              # Production
export LOG_FILE=/var/log/arwrench/api.log
export LOG_FORMAT=rich             # Rich for console, JSON goes to file
```

### Log rotation:

The `RotatingFileHandler` automatically creates backups:
- Max file size: 10MB
- Keep 5 backup files
- Total storage: ~60MB max

### Integration with monitoring:

```yaml
# Datadog
logs:
  - type: file
    path: /var/log/arwrench/api.log
    service: arwrench-api
    source: fastapi
    parser: json

# New Relic
logs:
  - type: file
    path: /var/log/arwrench/api.log
    attributes:
      service: arwrench-api
      env: production
```
