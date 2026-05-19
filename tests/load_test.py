"""ARWrench API — Load / Stress Test
=====================================
Runs concurrent requests against a live API instance and reports
latency percentiles, throughput, and error rates.

No extra dependencies — uses only Python stdlib (asyncio + httpx which
is already in requirements.txt).

Usage
-----
# Against local server (default)
python tests/load_test.py

# Against Railway (or any base URL)
BASE_URL=https://your-app.up.railway.app python tests/load_test.py

# Custom concurrency / iterations
CONCURRENCY=20 ITERATIONS=100 python tests/load_test.py

# With a real API token
API_SECRET_TOKEN=your-token python tests/load_test.py

Environment variables
---------------------
BASE_URL            Base URL of the running API  (default: http://localhost:8000)
API_SECRET_TOKEN    Shared secret for X-API-Token header (default: empty string)
CONCURRENCY         Simultaneous workers         (default: 10)
ITERATIONS          Total requests per scenario  (default: 50)
TIMEOUT             Per-request timeout seconds  (default: 30)
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.getenv("API_SECRET_TOKEN", "")
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))
ITERATIONS = int(os.getenv("ITERATIONS", "50"))
TIMEOUT = float(os.getenv("TIMEOUT", "30"))

HEADERS = {"Content-Type": "application/json"}
if API_TOKEN:
    HEADERS["X-API-Token"] = API_TOKEN


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class Result:
    scenario: str
    status: int
    duration_ms: float
    error: str = ""


@dataclass
class ScenarioStats:
    name: str
    results: list[Result] = field(default_factory=list)

    @property
    def durations(self) -> list[float]:
        return [r.duration_ms for r in self.results if r.status == 200]

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status not in (200, 429))

    @property
    def rate_limited_count(self) -> int:
        return sum(1 for r in self.results if r.status == 429)

    def print_summary(self) -> None:
        total = len(self.results)
        ok = len(self.durations)
        errors = self.error_count
        rate_limited = self.rate_limited_count

        print(f"\n{'─' * 55}")
        print(f"  Scenario : {self.name}")
        print(f"  Requests : {total}  |  OK: {ok}  |  Errors: {errors}  |  429s: {rate_limited}")

        if self.durations:
            sorted_d = sorted(self.durations)
            print(f"  Latency  : min={sorted_d[0]:.0f}ms  "
                  f"median={statistics.median(sorted_d):.0f}ms  "
                  f"p95={_percentile(sorted_d, 95):.0f}ms  "
                  f"max={sorted_d[-1]:.0f}ms")
        else:
            print("  Latency  : no successful responses")

        if errors > 0:
            sample_errors = [r.error for r in self.results if r.error][:3]
            for e in sample_errors:
                print(f"  ⚠  {e}")
        print(f"{'─' * 55}")


def _percentile(sorted_data: list[float], p: int) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Request payloads
# ---------------------------------------------------------------------------
HEALTH_PAYLOAD: dict[str, Any] = {}  # GET — no body

CLASSIFY_PAYLOAD = {
    "textDescription": "My car AC stopped working. Makes a humming noise when I turn it on.",
    "assetId": "load_test_asset_001",
    "visionResult": None,
}

ENRICH_PAYLOAD = {
    "threadId": "load_test_thread_001",
    "messages": [
        {"role": "user", "content": "My 2019 Toyota Camry AC makes a loud humming noise."}
    ],
    "assetContext": {
        "type": "vehicle",
        "make": "Toyota",
        "model": "Camry",
        "year": 2019,
        "mileage": None,
    },
    "toolbox": [],
    "serviceHistory": [],
    "skillLevel": 5,
}

REFINE_PAYLOAD = {
    "initialClassification": {
        "assetSpecs": {
            "type": "vehicle",
            "make": "Toyota",
            "model": "Camry",
            "year": 2019,
            "mileage": None,
        },
        "condition": "repair_needed",
        "confidence": 55,
    },
    "userAnswers": [
        {"q": "What is the approximate mileage?", "a": "45,000 miles"},
        {"q": "Has the AC been serviced before?", "a": "No"},
    ],
    "skillLevel": 5,
}


# ---------------------------------------------------------------------------
# Single-request worker
# ---------------------------------------------------------------------------
async def make_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    payload: dict | None,
    scenario: str,
    semaphore: asyncio.Semaphore,
) -> Result:
    async with semaphore:
        start = time.monotonic()
        try:
            if method == "GET":
                resp = await client.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
            else:
                resp = await client.post(
                    f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT
                )
            duration_ms = (time.monotonic() - start) * 1000
            return Result(scenario=scenario, status=resp.status_code, duration_ms=duration_ms)

        except httpx.TimeoutException:
            duration_ms = (time.monotonic() - start) * 1000
            return Result(scenario=scenario, status=0, duration_ms=duration_ms, error="Timeout")
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            return Result(scenario=scenario, status=0, duration_ms=duration_ms, error=str(exc))


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------
async def run_scenario(
    name: str,
    method: str,
    path: str,
    payload: dict | None,
    iterations: int,
    concurrency: int,
) -> ScenarioStats:
    stats = ScenarioStats(name=name)
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(headers=HEADERS) as client:
        tasks = [
            make_request(client, method, path, payload, name, semaphore)
            for _ in range(iterations)
        ]
        results = await asyncio.gather(*tasks)

    stats.results = list(results)
    return stats


# ---------------------------------------------------------------------------
# Health check — prerequisite
# ---------------------------------------------------------------------------
async def check_health() -> bool:
    print(f"Checking {BASE_URL}/health ...")
    try:
        async with httpx.AsyncClient(headers=HEADERS) as client:
            resp = await client.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                print("  ✓ API is reachable\n")
                return True
            print(f"  ✗ /health returned {resp.status_code}")
            return False
    except Exception as exc:
        print(f"  ✗ Cannot reach API: {exc}")
        return False


# ---------------------------------------------------------------------------
# Rate limit verification
# ---------------------------------------------------------------------------
async def verify_rate_limit() -> None:
    """Fire requests well above the rate limit cap and confirm 429s appear."""
    print("Verifying rate limiting (burst of 30 rapid requests) ...")
    semaphore = asyncio.Semaphore(30)
    stats = ScenarioStats(name="rate-limit-burst")

    async with httpx.AsyncClient(headers=HEADERS) as client:
        tasks = [
            make_request(client, "GET", "/health", None, "rate-limit-burst", semaphore)
            for _ in range(30)
        ]
        results = await asyncio.gather(*tasks)

    stats.results = list(results)
    ok = sum(1 for r in results if r.status == 200)
    limited = sum(1 for r in results if r.status == 429)
    print(f"  200 OK: {ok}  |  429 Rate-Limited: {limited}")
    if limited > 0:
        print("  ✓ Rate limiting is active\n")
    else:
        print("  ℹ  No 429s — either limit not reached or RATE_LIMIT_RPM is high\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    print("=" * 55)
    print("  ARWrench API — Load Test")
    print(f"  Target     : {BASE_URL}")
    print(f"  Concurrency: {CONCURRENCY} workers")
    print(f"  Iterations : {ITERATIONS} per scenario")
    print(f"  Auth       : {'token set' if API_TOKEN else 'no token'}")
    print("=" * 55 + "\n")

    if not await check_health():
        print("Aborting — API not reachable.")
        return

    scenarios = [
        # (name, method, path, payload)
        ("GET /health", "GET", "/health", None),
        ("POST /classify", "POST", "/classify", CLASSIFY_PAYLOAD),
        ("POST /enrich", "POST", "/enrich", ENRICH_PAYLOAD),
        ("POST /classify/refine", "POST", "/classify/refine", REFINE_PAYLOAD),
    ]

    all_stats: list[ScenarioStats] = []

    for name, method, path, payload in scenarios:
        print(f"Running: {name} × {ITERATIONS} requests ({CONCURRENCY} concurrent) ...")
        stats = await run_scenario(name, method, path, payload, ITERATIONS, CONCURRENCY)
        stats.print_summary()
        all_stats.append(stats)

    # Rate limit spot-check (uses /health only — no AI API cost)
    await verify_rate_limit()

    # Overall summary
    print("\n" + "=" * 55)
    print("  OVERALL SUMMARY")
    print("=" * 55)
    total_req = sum(len(s.results) for s in all_stats)
    total_ok = sum(len(s.durations) for s in all_stats)
    total_errors = sum(s.error_count for s in all_stats)
    all_durations = [d for s in all_stats for d in s.durations]

    print(f"  Total requests : {total_req}")
    print(f"  Successful     : {total_ok}")
    print(f"  Errors         : {total_errors}")
    if all_durations:
        sorted_all = sorted(all_durations)
        print(f"  Overall p95    : {_percentile(sorted_all, 95):.0f}ms")
        print(f"  Overall p99    : {_percentile(sorted_all, 99):.0f}ms")

    passed = total_errors == 0
    print(f"\n  Result: {'✓ PASS — no unexpected errors' if passed else '✗ FAIL — errors detected'}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
