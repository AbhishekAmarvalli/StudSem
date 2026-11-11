"""Fallback chain: Gemini -> OpenRouter -> Groq."""
import time

import pytest

import llm_client as L


@pytest.fixture(autouse=True)
def _clean_cooldowns(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(L, "_provider_down_until", {})
    monkeypatch.setattr(L, "_provider_last_success", {})
    monkeypatch.setattr(L, "_state_loaded", False)
    monkeypatch.setattr(L, "_probe_history", {})
    monkeypatch.setattr(L, "_provider_state_file", lambda: tmp_path / "provider_state.json")


def _rate_limited(*args, **kwargs):
    raise RuntimeError("429 Too Many Requests")


def test_groq_is_used_when_both_free_tiers_are_exhausted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(L, "_exponential_backoff", lambda attempt: None)
    monkeypatch.setattr(L, "_call_gemini", _rate_limited)
    monkeypatch.setattr(L, "_call_openrouter", _rate_limited)
    monkeypatch.setattr(L, "_call_groq", lambda *args, **kwargs: "from groq")
    assert L.call_llm("system", "user") == "from groq"


def test_missing_openrouter_key_still_reaches_groq(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(L, "_exponential_backoff", lambda attempt: None)
    monkeypatch.setattr(L, "_call_gemini", _rate_limited)
    monkeypatch.setattr(L, "_call_groq", lambda *args, **kwargs: "from groq")
    assert L.call_llm("system", "user") == "from groq"


def test_rate_limited_providers_are_skipped_on_later_calls(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(L, "_exponential_backoff", lambda attempt: None)
    gemini_calls, openrouter_calls, groq_calls = [], [], []

    def gemini(*args, **kwargs):
        gemini_calls.append(1)
        raise RuntimeError("429 Too Many Requests")

    def openrouter(*args, **kwargs):
        openrouter_calls.append(1)
        raise RuntimeError("429 Too Many Requests")

    def groq(*args, **kwargs):
        groq_calls.append(1)
        return "from groq"

    monkeypatch.setattr(L, "_call_gemini", gemini)
    monkeypatch.setattr(L, "_call_openrouter", openrouter)
    monkeypatch.setattr(L, "_call_groq", groq)

    assert L.call_llm("system", "user") == "from groq"
    assert L.call_llm("system", "user") == "from groq"
    assert len(gemini_calls) == L.MAX_RETRIES, "second call should not retry Gemini"
    assert len(openrouter_calls) == L.MAX_RETRIES, "second call should not retry OpenRouter"
    assert len(groq_calls) == 2


def test_successful_provider_is_preferred_on_the_next_call(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(L, "_exponential_backoff", lambda attempt: None)
    gemini_calls, openrouter_calls = [], []
    healthy = {"gemini": False}

    def gemini(*args, **kwargs):
        gemini_calls.append(1)
        if healthy["gemini"]:
            return "from gemini"
        raise RuntimeError("429 Too Many Requests")

    def openrouter(*args, **kwargs):
        openrouter_calls.append(1)
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(L, "_call_gemini", gemini)
    monkeypatch.setattr(L, "_call_openrouter", openrouter)
    monkeypatch.setattr(L, "_call_groq", lambda *args, **kwargs: "from groq")

    assert L.call_llm("system", "user") == "from groq"
    healthy["gemini"] = True
    L._provider_down_until.clear()  # cooldown over; stickiness should still apply
    assert L.call_llm("system", "user") == "from groq"
    assert len(gemini_calls) == L.MAX_RETRIES
    assert len(openrouter_calls) == L.MAX_RETRIES


def test_provider_state_survives_a_reload(monkeypatch):
    L._provider_down_until["gemini"] = time.time() + 120
    L._provider_last_success["groq"] = time.time()
    L._save_provider_state()

    L._provider_down_until.clear()
    L._provider_last_success.clear()
    L._state_loaded = False
    L._ensure_provider_state()

    assert L._provider_down_until["gemini"] > time.time()
    assert "groq" in L._provider_last_success


def test_provider_status_reports_paused_and_preferred(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    L._provider_down_until["gemini"] = time.time() + 120
    L._provider_last_success["groq"] = time.time()
    statuses = {item["name"]: item["state"] for item in L.provider_status()}
    assert statuses == {"Gemini": "paused", "OpenRouter": "off", "Groq": "preferred"}


def test_probe_reports_each_configured_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(L, "_call_gemini", lambda *args, **kwargs: "OK")
    monkeypatch.setattr(L, "_call_groq", lambda *args, **kwargs: "OK")

    results = {item["name"]: item for item in L.probe_providers()}
    assert results["Gemini"]["ok"] is True
    assert results["Groq"]["ok"] is True
    assert results["OpenRouter"]["ok"] is False
    assert "OPENROUTER_API_KEY not set" in results["OpenRouter"]["error"]


def test_probe_keeps_latency_history(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(L, "_call_gemini", lambda *args, **kwargs: "OK")
    monkeypatch.setattr(L, "_call_groq", lambda *args, **kwargs: "OK")

    L.probe_providers()
    results = {item["name"]: item for item in L.probe_providers()}
    assert len(results["Gemini"]["history"]) == 2
    assert len(results["Groq"]["history"]) == 2
    assert all("latency_ms" in entry for entry in results["Gemini"]["history"])
    assert results["Gemini"]["history"][-1]["ok"] is True


def test_probe_pauses_a_rate_limited_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def limited(*args, **kwargs):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(L, "_call_gemini", limited)
    results = L.probe_providers()
    assert results[0]["name"] == "Gemini"
    assert results[0]["ok"] is False
    assert L._provider_down_until["gemini"] > time.time()


def test_all_providers_failing_names_each_one(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(L, "_exponential_backoff", lambda attempt: None)
    monkeypatch.setattr(L, "_call_gemini", _rate_limited)
    monkeypatch.setattr(L, "_call_openrouter", _rate_limited)
    monkeypatch.setattr(L, "_call_groq", _rate_limited)
    try:
        L.call_llm("system", "user")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected every provider to fail")
    assert "All LLM providers failed" in message
    assert "OpenRouter: 429" in message
    assert "Groq: 429" in message
