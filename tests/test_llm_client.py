"""Fallback chain: Gemini -> OpenRouter -> Groq."""
import llm_client as L


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
