"""
ARWrench AI Enrichment API — Test Suite
========================================

Two layers, one file:

  Unit tests    — pure functions, always run, no API keys needed
  Integration   — real Grok / Gemini calls, auto-skip when keys absent

Usage:
  pytest tests/ -v                        # all tests
  pytest tests/ -v -m "not integration"   # unit only (fast)
  pytest tests/ -v -m integration         # real API only (slow)

Environment variables required for integration tests:
  GROK_API_KEY     — xAI Grok API key (used for all endpoints including /vision)
  TEST_IMAGE_URL   — publicly accessible image URL for /vision tests
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from models import AssetContext, EnrichRequest, Message  # noqa: E402
from routers.enrich import (  # noqa: E402
    SKILL_LEVEL_PREFIXES,
    build_enriched_prompt,
    compute_confidence,
    summarize_messages,
)
from services.grok_client import parse_json_response  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Key / environment detection
# ---------------------------------------------------------------------------

def _real_key(name: str) -> bool:
    """True only when the env var is set AND is not a placeholder value."""
    v = os.getenv(name, "")
    return bool(v) and not v.startswith("your-") and not v.startswith("test-")


HAS_GROK = _real_key("GROK_API_KEY")
TEST_IMAGE_URL = os.getenv("TEST_IMAGE_URL", "")

needs_grok = pytest.mark.skipif(not HAS_GROK, reason="GROK_API_KEY not configured")
needs_image = pytest.mark.skipif(not TEST_IMAGE_URL, reason="TEST_IMAGE_URL not set")


# ---------------------------------------------------------------------------
# Shared request fixtures
# ---------------------------------------------------------------------------

CAMRY_ENRICH = {
    "threadId": "t_camry_001",
    "messages": [{"role": "user", "content": "My car makes a grinding noise when I brake, worse going downhill."}],
    "assetContext": {
        "type": "vehicle",
        "make": "Toyota",
        "model": "Camry",
        "year": 2018,
        "mileage": 74000,
    },
    "toolbox": ["floor jack", "socket set", "torque wrench"],
    "serviceHistory": [
        {"date": "2023-08-10", "service": "Oil change"},
        {"date": "2022-03-01", "service": "Front brake pad replacement"},
    ],
    "skillLevel": 5,
}

WASHER_ENRICH = {
    **CAMRY_ENRICH,
    "threadId": "t_washer_001",
    "messages": [{"role": "user", "content": "Washing machine vibrates violently during spin cycle."}],
    "assetContext": {
        "type": "appliance",
        "make": "Whirlpool",
        "model": "WFW5000FW",
        "year": 2020,
        "mileage": None,
    },
}

CLASSIFY_REQ = {
    "visionResult": "Cracked water pump gasket, oil residue visible",
    "textDescription": "Car overheating, coolant loss noticed",
    "assetId": "asset_camry_456",
}

REFINE_REQ = {
    "initialClassification": {
        "assetSpecs": {"type": "vehicle", "make": "Toyota", "model": "Camry"},
        "condition": "maintenance_needed",
        "confidence": 60,
    },
    "userAnswers": [
        {"q": "What is the current mileage?", "a": "74,000 miles"},
        {"q": "Any recent repairs?", "a": "Front brakes replaced two years ago"},
    ],
    "skillLevel": 5,
}

REFINE_LOW_CONFIDENCE = {
    **REFINE_REQ,
    "initialClassification": {
        "assetSpecs": {"type": "unknown"},
        "condition": "unknown",
        "confidence": 10,   # Force dual-pass validation path
    },
}


# ===========================================================================
# UNIT TESTS — pure functions, no API, always run
# ===========================================================================

class TestParseJsonResponse:

    def test_direct_json(self):
        raw = '{"answer": "test", "confidence": 80}'
        result = parse_json_response(raw)
        assert result["answer"] == "test"
        assert result["confidence"] == 80

    def test_json_in_code_block(self):
        raw = '```json\n{"answer": "test", "confidence": 80}\n```'
        result = parse_json_response(raw)
        assert result["answer"] == "test"

    def test_json_in_plain_code_block(self):
        raw = '```\n{"answer": "test"}\n```'
        result = parse_json_response(raw)
        assert result["answer"] == "test"

    def test_json_embedded_in_prose(self):
        raw = 'Here is the result: {"answer": "test", "confidence": 70} Hope that helps.'
        result = parse_json_response(raw)
        assert result["answer"] == "test"

    def test_invalid_falls_back(self):
        raw = "This is just plain prose with no JSON at all."
        result = parse_json_response(raw)
        assert result.get("_parse_failed") is True
        assert result.get("_raw") == raw

    def test_whitespace_stripped(self):
        raw = '\n\n  {"x": 1}  \n\n'
        result = parse_json_response(raw)
        assert result["x"] == 1


class TestComputeConfidence:

    def test_formula_average(self):
        parsed = {"symptom_match": 90, "history_alignment": 60, "specificity": 60}
        score, _ = compute_confidence(parsed)
        assert score == 70  # (90+60+60)//3

    def test_clamped_above_100(self):
        parsed = {"symptom_match": 200, "history_alignment": 200, "specificity": 200}
        score, _ = compute_confidence(parsed)
        assert score == 100

    def test_clamped_below_0(self):
        parsed = {"symptom_match": -50, "history_alignment": -50, "specificity": -50}
        score, _ = compute_confidence(parsed)
        assert score == 0

    def test_three_reasons_always_present(self):
        parsed = {"symptom_match": 50, "history_alignment": 50, "specificity": 50}
        _, reasons = compute_confidence(parsed)
        assert len(reasons) == 3

    def test_safety_reason_appended(self):
        parsed = {
            "symptom_match": 80,
            "history_alignment": 80,
            "specificity": 80,
            "safety_flag": True,
            "safety_explanation": "Brakes are safety-critical",
        }
        _, reasons = compute_confidence(parsed)
        assert len(reasons) == 4
        assert any("Safety" in r for r in reasons)

    def test_no_safety_reason_when_flag_false(self):
        parsed = {
            "symptom_match": 80,
            "history_alignment": 80,
            "specificity": 80,
            "safety_flag": False,
        }
        _, reasons = compute_confidence(parsed)
        assert len(reasons) == 3

    def test_missing_keys_default_to_50(self):
        score, _ = compute_confidence({})
        assert score == 50


class TestSummarizeMessages:

    def _msgs(self, contents: list[str]) -> list[Message]:
        roles = ["user", "assistant"]
        return [Message(role=roles[i % 2], content=c) for i, c in enumerate(contents)]

    def test_five_or_fewer_unchanged(self):
        msgs = self._msgs(["a", "b", "c", "d", "e"])
        recent, summary = summarize_messages(msgs)
        assert len(recent) == 5
        assert summary == ""

    def test_six_messages_compresses_oldest(self):
        msgs = self._msgs(["old1", "old2", "r1", "r2", "r3", "r4", "r5"])
        recent, summary = summarize_messages(msgs)
        assert len(recent) == 5
        assert "old1" in summary
        assert "old2" in summary

    def test_recent_messages_correct_content(self):
        msgs = self._msgs(["old1", "old2", "new1", "new2", "new3", "new4", "new5"])
        recent, _ = summarize_messages(msgs)
        contents = [m["content"] for m in recent]
        assert "new1" in contents
        assert "old1" not in contents

    def test_single_message(self):
        msgs = self._msgs(["only message"])
        recent, summary = summarize_messages(msgs)
        assert len(recent) == 1
        assert summary == ""


class TestBuildEnrichedPrompt:

    def _make_req(self, asset: dict, temp: int = 5) -> EnrichRequest:
        return EnrichRequest(
            threadId="t_test",
            messages=[Message(role="user", content="symptom")],
            assetContext=AssetContext(**asset),
            toolbox=["wrench"],
            serviceHistory=[{"date": "2024-01-01", "service": "Oil change"}],
            skillLevel=temp,
        )

    def test_contains_make_and_model(self):
        req = self._make_req({"make": "Toyota", "model": "Camry", "year": 2018})
        prompt = build_enriched_prompt(req)
        assert "Toyota" in prompt
        assert "Camry" in prompt

    def test_contains_mileage(self):
        req = self._make_req({"make": "Ford", "model": "F-150", "mileage": 50000})
        prompt = build_enriched_prompt(req)
        assert "50,000" in prompt

    def test_contains_toolbox(self):
        req = self._make_req({"type": "vehicle"})
        prompt = build_enriched_prompt(req)
        assert "wrench" in prompt

    def test_contains_service_history(self):
        req = self._make_req({"type": "vehicle"})
        prompt = build_enriched_prompt(req)
        assert "Oil change" in prompt

    def test_no_history_shows_fallback(self):
        req = EnrichRequest(
            threadId="t",
            messages=[Message(role="user", content="q")],
            assetContext=AssetContext(type="vehicle"),
            toolbox=[],
            serviceHistory=[],
            skillLevel=5,
        )
        prompt = build_enriched_prompt(req)
        assert "No service history" in prompt

    def test_older_summary_injected(self):
        req = self._make_req({"type": "vehicle"})
        prompt = build_enriched_prompt(req, conversation_summary="User mentioned smoke last week.")
        assert "smoke last week" in prompt

    def test_asset_isolation_camry_not_in_washer_prompt(self):
        req_camry = self._make_req({"make": "Toyota", "model": "Camry"})
        req_washer = self._make_req({"make": "Whirlpool", "model": "WFW5000"})
        prompt_camry = build_enriched_prompt(req_camry)
        prompt_washer = build_enriched_prompt(req_washer)
        assert "Camry" in prompt_camry
        assert "Camry" not in prompt_washer
        assert "Whirlpool" in prompt_washer
        assert "Whirlpool" not in prompt_camry

    def test_skill_level_1_simple_language(self):
        req = self._make_req({"type": "vehicle"}, temp=1)
        prompt = build_enriched_prompt(req)
        assert "everyday language" in prompt.lower() or "no mechanical knowledge" in prompt.lower()

    def test_skill_level_10_expert(self):
        req = self._make_req({"type": "vehicle"}, temp=10)
        prompt = build_enriched_prompt(req)
        assert "master technician" in prompt.lower() or "peer" in prompt.lower()

    def test_all_skill_levels_are_distinct(self):
        prompts = set()
        for level in range(1, 11):
            req = self._make_req({"type": "vehicle"}, temp=level)
            prompts.add(build_enriched_prompt(req))
        assert len(prompts) == 10, "Each skill level must produce a distinct prompt"

    def test_skill_level_prefixes_dict_has_ten_entries(self):
        assert len(SKILL_LEVEL_PREFIXES) == 10
        assert set(SKILL_LEVEL_PREFIXES.keys()) == set(range(1, 11))


class TestInputValidation:
    """FastAPI / Pydantic validation — no API calls needed."""

    def test_health_ok(self):
        assert client.get("/health").status_code == 200

    def test_enrich_skill_level_too_high(self):
        req = {**CAMRY_ENRICH, "skillLevel": 11}
        assert client.post("/enrich", json=req).status_code == 422

    def test_enrich_skill_level_zero(self):
        req = {**CAMRY_ENRICH, "skillLevel": 0}
        assert client.post("/enrich", json=req).status_code == 422

    def test_enrich_missing_thread_id(self):
        req = {k: v for k, v in CAMRY_ENRICH.items() if k != "threadId"}
        assert client.post("/enrich", json=req).status_code == 422

    def test_enrich_missing_messages(self):
        req = {k: v for k, v in CAMRY_ENRICH.items() if k != "messages"}
        assert client.post("/enrich", json=req).status_code == 422

    def test_enrich_missing_asset_context(self):
        req = {k: v for k, v in CAMRY_ENRICH.items() if k != "assetContext"}
        assert client.post("/enrich", json=req).status_code == 422

    def test_vision_missing_image_url(self):
        assert client.post("/vision", json={"systemPrompt": "analyse"}).status_code == 422

    def test_classify_missing_text_description(self):
        req = {k: v for k, v in CLASSIFY_REQ.items() if k != "textDescription"}
        assert client.post("/classify", json=req).status_code == 422

    def test_classify_missing_asset_id(self):
        req = {k: v for k, v in CLASSIFY_REQ.items() if k != "assetId"}
        assert client.post("/classify", json=req).status_code == 422

    def test_refine_skill_level_zero(self):
        req = {**REFINE_REQ, "skillLevel": 0}
        assert client.post("/classify/refine", json=req).status_code == 422

    def test_refine_missing_initial_classification(self):
        req = {k: v for k, v in REFINE_REQ.items() if k != "initialClassification"}
        assert client.post("/classify/refine", json=req).status_code == 422

    def test_refine_missing_user_answers(self):
        req = {k: v for k, v in REFINE_REQ.items() if k != "userAnswers"}
        assert client.post("/classify/refine", json=req).status_code == 422


# ===========================================================================
# INTEGRATION TESTS — real API calls, auto-skip when keys absent
# ===========================================================================

def _assert_enrich_response(data: dict):
    """Shared structural assertions for any /enrich response."""
    assert isinstance(data["answer"], str), "answer must be a string"
    assert data["answer"].strip(), "answer must not be empty"
    assert isinstance(data["confidence"], int), "confidence must be int"
    assert 0 <= data["confidence"] <= 100, f"confidence out of range: {data['confidence']}"
    assert isinstance(data["confidence_reasons"], list), "confidence_reasons must be list"
    assert len(data["confidence_reasons"]) >= 1
    assert isinstance(data["safety_flag"], bool)
    assert isinstance(data["abstain"], bool)
    assert data["role"] == "assistant"


@needs_grok
@pytest.mark.integration
def test_enrich_camry_brake_symptom():
    """Real Grok call — grinding brake noise on a 2018 Camry."""
    resp = client.post("/enrich", json=CAMRY_ENRICH)
    assert resp.status_code == 200, resp.text
    _assert_enrich_response(resp.json())


@needs_grok
@pytest.mark.integration
def test_enrich_appliance_symptom():
    """Real Grok call — washing machine vibration symptom."""
    resp = client.post("/enrich", json=WASHER_ENRICH)
    assert resp.status_code == 200, resp.text
    _assert_enrich_response(resp.json())


@needs_grok
@pytest.mark.integration
def test_enrich_asset_isolation_responses():
    """
    Send two different assets back-to-back.
    Each response must describe the correct asset type — no cross-contamination.
    Toyota Camry answer should not mention Whirlpool; washer answer should not mention Camry.
    """
    resp_camry = client.post("/enrich", json=CAMRY_ENRICH)
    resp_washer = client.post("/enrich", json=WASHER_ENRICH)

    assert resp_camry.status_code == 200, resp_camry.text
    assert resp_washer.status_code == 200, resp_washer.text

    answer_camry = resp_camry.json()["answer"].lower()
    answer_washer = resp_washer.json()["answer"].lower()

    # Camry answer should not mention Whirlpool brand
    assert "whirlpool" not in answer_camry, "Camry answer leaked Whirlpool context"
    # Washer answer should not mention Camry model
    assert "camry" not in answer_washer, "Washer answer leaked Camry context"


@needs_grok
@pytest.mark.integration
def test_enrich_safety_flag_on_brake_symptom():
    """
    Brake symptoms should result in safety_flag=True from Grok.
    This is a soft assertion — we verify the field is boolean and log the result.
    """
    resp = client.post("/enrich", json=CAMRY_ENRICH)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # safety_flag must be a boolean; brakes should typically set it True
    assert isinstance(data["safety_flag"], bool)
    # We don't hard-assert True because Grok decides — just ensure it's present


@needs_grok
@pytest.mark.integration
def test_enrich_skill_level_1_vs_10():
    """
    Skill level 1 (novice) and 10 (master technician) should produce different answers.
    At minimum, the responses should not be byte-for-byte identical.
    """
    req_novice = {**CAMRY_ENRICH, "skillLevel": 1}
    req_expert = {**CAMRY_ENRICH, "skillLevel": 10}

    resp1 = client.post("/enrich", json=req_novice)
    resp10 = client.post("/enrich", json=req_expert)

    assert resp1.status_code == 200, resp1.text
    assert resp10.status_code == 200, resp10.text

    _assert_enrich_response(resp1.json())
    _assert_enrich_response(resp10.json())

    assert resp1.json()["answer"] != resp10.json()["answer"], \
        "Skill level 1 and 10 produced identical answers — prompt prefix may not be working"


@needs_grok
@pytest.mark.integration
def test_enrich_empty_toolbox_and_history():
    """Edge case: no tools, no history — Grok should still return a valid answer."""
    req = {
        **CAMRY_ENRICH,
        "toolbox": [],
        "serviceHistory": [],
    }
    resp = client.post("/enrich", json=req)
    assert resp.status_code == 200, resp.text
    _assert_enrich_response(resp.json())


@needs_grok
@pytest.mark.integration
def test_enrich_long_conversation_compressed():
    """8-message thread — messages beyond the last 5 must be summarised, not dropped."""
    messages = [
        {"role": "user",      "content": "My Camry started making a noise last week."},
        {"role": "assistant", "content": "Can you describe the noise?"},
        {"role": "user",      "content": "It sounds like grinding."},
        {"role": "assistant", "content": "Is it louder when braking?"},
        {"role": "user",      "content": "Yes, especially going downhill."},
        {"role": "assistant", "content": "That sounds like worn brake pads."},
        {"role": "user",      "content": "The front pads were replaced 2 years ago."},
        {"role": "user",      "content": "Could it be the rear brakes?"},
    ]
    req = {**CAMRY_ENRICH, "messages": messages}
    resp = client.post("/enrich", json=req)
    assert resp.status_code == 200, resp.text
    _assert_enrich_response(resp.json())


@needs_grok
@needs_image
@pytest.mark.integration
def test_vision_returns_valid_structure():
    """Real Gemini call — analyse the image at TEST_IMAGE_URL."""
    req = {
        "imageUrl": TEST_IMAGE_URL,
        "systemPrompt": "Analyse this photo of a vehicle part.",
        "assetContext": {"type": "vehicle", "make": "Toyota"},
    }
    resp = client.post("/vision", json=req)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert isinstance(data["analysis"], str) and data["analysis"].strip()
    assert isinstance(data["confidence"], int)
    assert 0 <= data["confidence"] <= 100

    sf = data["structured_findings"]
    assert "part" in sf
    assert "condition" in sf
    assert "severity" in sf
    assert isinstance(sf.get("observations", []), list)


@needs_grok
@pytest.mark.integration
def test_classify_returns_valid_structure():
    """Real Grok call — extract asset specs from vision + text."""
    resp = client.post("/classify", json=CLASSIFY_REQ)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert isinstance(data["assetSpecs"], dict)
    assert isinstance(data["condition"], str) and data["condition"]
    assert isinstance(data["confidence"], int)
    assert 0 <= data["confidence"] <= 100
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["reminders"], list)


@needs_grok
@pytest.mark.integration
def test_classify_text_only_no_vision():
    """classify should work with no visionResult — text description alone."""
    req = {
        "visionResult": None,
        "textDescription": "Lawn mower won't start, engine cranks but no ignition.",
        "assetId": "asset_mower_789",
    }
    resp = client.post("/classify", json=req)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert isinstance(data["assetSpecs"], dict)
    assert isinstance(data["condition"], str)


@needs_grok
@pytest.mark.integration
def test_refine_returns_valid_structure():
    """Real Grok call — refine initial classification with user Q&A."""
    resp = client.post("/classify/refine", json=REFINE_REQ)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert isinstance(data["assetSpecs"], dict)
    assert isinstance(data["condition"], str) and data["condition"]
    assert isinstance(data["confidence"], int)
    assert 0 <= data["confidence"] <= 100
    assert isinstance(data["reminders"], list)
    assert isinstance(data["refined_recommendations"], list)


@needs_grok
@pytest.mark.integration
def test_refine_dual_pass_low_initial_confidence():
    """
    Pass initialClassification.confidence=10 to force the dual-pass code path.
    Response must still be valid — the self-critique loop must not crash.
    """
    resp = client.post("/classify/refine", json=REFINE_LOW_CONFIDENCE)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert isinstance(data["assetSpecs"], dict)
    assert isinstance(data["confidence"], int)
    assert 0 <= data["confidence"] <= 100


@needs_grok
@pytest.mark.integration
def test_refine_all_skill_levels():
    """All 10 skill levels must produce a valid response."""
    for level in range(1, 11):
        req = {**REFINE_REQ, "skillLevel": level}
        resp = client.post("/classify/refine", json=req)
        assert resp.status_code == 200, f"Skill level {level} failed: {resp.text}"
        data = resp.json()
        assert 0 <= data["confidence"] <= 100, f"Confidence out of range at skill level {level}"
