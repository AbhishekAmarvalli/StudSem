"""
llm_client.py – Unified LLM caller with retry + fallback.

Priority: Gemini -> OpenRouter -> Groq (fallback on rate-limit)
Examiner calls always go to Groq first.
"""
from __future__ import annotations

import os
import re
import time
import json
import logging
from contextvars import ContextVar
from typing import Any, Dict, Optional

from google import genai as genai_new
from google.genai import types as genai_types
from groq import Groq
import requests

logger = logging.getLogger(__name__)

# ── Models ─────────────────────────────────────────────────────────────────────
# These must be current IDs: the previous defaults (gemini-2.0-flash,
# llama3-8b-8192, mistralai/mistral-7b-instruct:free) were retired by their
# providers, so every call with a real key failed with 404 / model_not_found.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"

RETIRED_MODELS = {
    "gemini": {
        "gemini-2.0-flash", "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite", "gemini-2.0-flash-lite-001",
        "gemini-1.5-flash", "gemini-1.5-pro",
    },
    "groq": {"llama3-8b-8192", "llama3-70b-8192"},
    "openrouter": {"mistralai/mistral-7b-instruct:free"},
}

# A hung HTTP call must not freeze a whole run in the Streamlit UI.
REQUEST_TIMEOUT_MS = 180_000

# A long run makes dozens of calls. Once a provider has exhausted its retries
# on a rate limit, skip it for a while instead of paying its backoff every time.
PROVIDER_COOLDOWN_SECONDS = int(os.getenv("LLM_PROVIDER_COOLDOWN", "600"))
_provider_down_until: Dict[str, float] = {}

# Once a provider answers, prefer it for the next calls instead of probing the
# preferred provider again on every chunk of the run.
PROVIDER_STICKY_SECONDS = int(os.getenv("LLM_PROVIDER_STICKY_SECONDS", "900"))
_provider_last_success: Dict[str, float] = {}

PROVIDER_ORDER = ("gemini", "openrouter", "groq")
PROVIDER_LABELS = {"gemini": "Gemini", "openrouter": "OpenRouter", "groq": "Groq"}
PROVIDER_API_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Which provider answered the latest call, for progress messages.
_last_provider: ContextVar[str] = ContextVar("last_provider", default="")


def last_provider() -> str:
    return _last_provider.get()


def reset_provider_state() -> None:
    _provider_down_until.clear()
    _provider_last_success.clear()
    _last_provider.set("")


def _provider_available(name: str) -> bool:
    return time.time() >= _provider_down_until.get(name, 0.0)


def _provider_rank(name: str) -> int:
    last = _provider_last_success.get(name)
    if last and time.time() - last < PROVIDER_STICKY_SECONDS:
        return 0
    return 1


def _ordered_providers() -> list:
    return sorted(
        PROVIDER_ORDER, key=lambda name: (_provider_rank(name), PROVIDER_ORDER.index(name))
    )


def _mark_provider_success(name: str) -> None:
    _provider_last_success[name] = time.time()
    _last_provider.set(PROVIDER_LABELS[name])


def _is_transient(error: str) -> bool:
    err = error.lower()
    return any(
        token in err
        for token in ("rate", "quota", "429", "503", "unavailable", "overloaded", "temporarily")
    )


def _mark_provider_down(name: str, error: Exception) -> None:
    _provider_down_until[name] = time.time() + PROVIDER_COOLDOWN_SECONDS
    logger.warning(
        "%s paused for %ss after rate limiting: %s",
        name, PROVIDER_COOLDOWN_SECONDS, error,
    )


def _mark_provider_up(name: str) -> None:
    _provider_down_until.pop(name, None)


def _active_model(kind: str, env_var: str, default: str) -> str:
    """Read a model ID at call time and skip IDs the provider has retired.

    Values are read per call (not at import) so a key or model set in the
    sidebar mid-session takes effect, and an old .env keeps working instead of
    failing with a 404.
    """
    name = (os.getenv(env_var) or default).strip()
    if name in RETIRED_MODELS[kind]:
        logger.warning("%s=%s was retired by the provider; using %s.", env_var, name, default)
        return default
    return name


def gemini_model() -> str:
    return _active_model("gemini", "GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def groq_model() -> str:
    return _active_model("groq", "GROQ_MODEL", DEFAULT_GROQ_MODEL)


def openrouter_model() -> str:
    return _active_model("openrouter", "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)

_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    # Always re-read key from env so sidebar input is picked up at runtime
    key = os.getenv("GEMINI_API_KEY", "")
    if key and (_gemini_client is None):
        _gemini_client = genai_new.Client(api_key=key)
    elif key and _gemini_client is not None:
        # Reinitialize if key changed
        try:
            if _gemini_client._api_client._api_key != key:  # type: ignore
                _gemini_client = genai_new.Client(api_key=key)
        except Exception:
            pass
    return _gemini_client

MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds


def _exponential_backoff(attempt: int) -> None:
    wait = BACKOFF_BASE ** attempt
    logger.info(f"Rate-limit hit. Waiting {wait}s before retry {attempt+1}…")
    time.sleep(wait)


def _call_gemini(system_prompt: str, user_prompt: str, **kwargs) -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it in the sidebar or .env file.")
    client = genai_new.Client(
        api_key=key,
        http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    model_name = gemini_model()
    contents = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}" if system_prompt else user_prompt

    # JSON mode stops the "Here is the JSON:" / fenced-block drift that made
    # parsing fail; temperature 0.2 keeps factor values and marks stable.
    config = genai_types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=model_name, contents=contents, config=config,
    )

    try:
        text = response.text
    except Exception:  # no text parts at all (blocked prompt, empty candidate)
        text = None
    if not text:
        finish = "unknown"
        try:
            finish = str(response.candidates[0].finish_reason)
        except Exception:
            pass
        if "MAX_TOKENS" in finish:
            raise RuntimeError(
                f"Gemini stopped at its output limit ({model_name}). "
                "Generate less content in one call or pick a model with a larger limit."
            )
        raise RuntimeError(
            f"Gemini returned no text for {model_name} (finish_reason={finish}). "
            "The prompt may have been blocked; try a cleaner or shorter document."
        )
    return text


def _call_groq(system_prompt: str, user_prompt: str, **kwargs) -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set.")
    client = Groq(api_key=key, timeout=REQUEST_TIMEOUT_MS / 1000)
    chat = client.chat.completions.create(
        model=groq_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = chat.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")
    return content


def _call_openrouter(system_prompt: str, user_prompt: str, **kwargs) -> str:
    # Read the key here: the sidebar sets it after this module is imported.
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to .env, Streamlit secrets or the sidebar."
        )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://exam-intelligence-workflow.streamlit.app",
        "X-Title": "Exam Intelligence Workflow",
    }
    payload = {
        "model": openrouter_model(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_llm(
    system_prompt: str,
    user_prompt: str,
    use_groq: bool = False,
    **kwargs,
) -> str:
    """
    Call LLM with automatic retry + fallback.
    use_groq=True forces the Groq examiner path.
    """
    if use_groq:
        for attempt in range(MAX_RETRIES):
            try:
                result = _call_groq(system_prompt, user_prompt, **kwargs)
                _mark_provider_success("groq")
                return result
            except Exception as e:
                err = str(e).lower()
                if "rate" in err and attempt < MAX_RETRIES - 1:
                    _exponential_backoff(attempt)
                else:
                    raise RuntimeError(f"Groq failed: {e}") from e

    errors: List[str] = []
    for name in _ordered_providers():
        label = PROVIDER_LABELS[name]
        key_var = PROVIDER_API_KEYS[name]
        if not os.getenv(key_var):
            errors.append(f"{label}: {key_var} not set")
            continue
        if not _provider_available(name):
            errors.append(f"{label}: paused after a recent rate limit")
            continue
        try:
            result = _attempt_provider(name, system_prompt, user_prompt, **kwargs)
        except Exception as e:
            errors.append(f"{label}: {e}")
            if _is_transient(str(e)):
                _mark_provider_down(name, e)
            continue
        _mark_provider_up(name)
        _mark_provider_success(name)
        return result

    raise RuntimeError(
        "All LLM providers failed.\n"
        + "\n".join(f"- {item}" for item in errors)
        + "\nCheck your API keys and free-tier quotas."
    )


def _attempt_provider(
    name: str, system_prompt: str, user_prompt: str, **kwargs: Any
) -> str:
    """Call one provider, retrying only temporary failures."""
    caller = {
        "gemini": _call_gemini,
        "openrouter": _call_openrouter,
        "groq": _call_groq,
    }[name]
    for attempt in range(MAX_RETRIES):
        try:
            return caller(system_prompt, user_prompt, **kwargs)
        except Exception as e:
            if not _is_transient(str(e)):
                raise
            if attempt < MAX_RETRIES - 1:
                _exponential_backoff(attempt)
                continue
            raise


def _lenient_json(text: str) -> Any:
    """Load JSON that a model may have written loosely.

    Models occasionally emit raw newlines/tabs inside strings or a backslash
    that is not a legal JSON escape (e.g. ``\\(a)``). Plain ``json.loads``
    rejects both, which loses an otherwise usable answer.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
    return json.loads(repaired, strict=False)


def _json_slice(text: str) -> str:
    """Return the outermost JSON value from text.

    Models still wrap JSON in fences or add a closing remark even in JSON mode,
    so take the span from the first ``{``/``[`` to the last ``}``/``]`` instead
    of trusting the first and last lines.
    """
    stripped = (text or "").strip()
    starts = [i for i in (stripped.find("{"), stripped.find("[")) if i != -1]
    if not starts:
        return stripped
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    return stripped[start:end + 1] if end > start else stripped[start:]


def _load_into(text: str, schema_cls):
    return schema_cls(**_lenient_json(_json_slice(text)))


def parse_json_response(raw: str, schema_cls, retry_prompt: str = "") -> Any:
    """
    Parse raw LLM text as JSON into schema_cls (Pydantic model).
    Retries once with the error message if parsing fails.
    """
    if raw is None or not str(raw).strip():
        raise ValueError("The model returned an empty response; nothing to parse.")

    try:
        return _load_into(str(raw), schema_cls)
    except Exception as first_err:
        if not retry_prompt:
            raise ValueError(f"JSON parse failed: {first_err}\nRaw: {str(raw)[:500]}") from first_err

        # Retry with error context
        fixed_raw = call_llm(
            system_prompt="Fix the JSON so it matches the required schema exactly. Return only valid JSON.",
            user_prompt=(
                f"Original prompt: {retry_prompt}\n\n"
                f"Your previous output caused this error: {first_err}\n\n"
                f"Previous output: {raw}\n\n"
                "Return corrected JSON only."
            ),
        )
        try:
            return _load_into(str(fixed_raw), schema_cls)
        except Exception as second_err:
            raise ValueError(
                f"JSON parse failed even after retry.\n"
                f"First error: {first_err}\nSecond error: {second_err}\nRaw: {str(raw)[:300]}"
            ) from second_err
