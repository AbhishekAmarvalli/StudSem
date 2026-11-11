"""Fallback chain: Gemini -> OpenRouter -> Groq."""
import pytest

import llm_client as L


@pytest.fixture(autouse=True)
def _clean_cooldowns(monkeypatch):
    monkeypatch.setattr(L, "_provider_down_until", {})
    monkeypatch.setattr(L, "_provider_last_success", {})


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
