"""
llm_client.py – Unified LLM caller with retry + fallback.

Priority: Gemini -> OpenRouter (fallback on rate-limit)
Examiner calls always go to Groq.
"""
from __future__ import annotations

import os
import re
import time
import json
import logging
from typing import Any, Dict, Optional

from google import genai as genai_new
from google.genai import types as genai_types
from groq import Groq
import requests

logger = logging.getLogger(__name__)

# ── Load from env ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")

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
    client = genai_new.Client(api_key=key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    contents = []
    if system_prompt:
        contents.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}")]
        ))
    else:
        contents.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_prompt)]
        ))
    response = client.models.generate_content(model=model_name, contents=contents)
    return response.text


def _call_groq(system_prompt: str, user_prompt: str, **kwargs) -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set.")
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    client = Groq(api_key=key)
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return chat.choices[0].message.content


def _call_openrouter(system_prompt: str, user_prompt: str, **kwargs) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://exam-intelligence-workflow.streamlit.app",
        "X-Title": "Exam Intelligence Workflow",
    }
    payload = {
        "model": OPENROUTER_MODEL,
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
                return _call_groq(system_prompt, user_prompt, **kwargs)
            except Exception as e:
                err = str(e).lower()
                if "rate" in err and attempt < MAX_RETRIES - 1:
                    _exponential_backoff(attempt)
                else:
                    raise RuntimeError(f"Groq failed: {e}") from e

    # Gemini first
    for attempt in range(MAX_RETRIES):
        try:
            return _call_gemini(system_prompt, user_prompt, **kwargs)
        except Exception as e:
            err = str(e).lower()
            # 503 / UNAVAILABLE / overloaded are temporary like a quota hit, so
            # retry and then fall back instead of failing the whole run.
            transient = (
                "rate" in err
                or "quota" in err
                or "503" in err
                or "unavailable" in err
                or "overloaded" in err
            )
            if transient:
                if attempt < MAX_RETRIES - 1:
                    _exponential_backoff(attempt)
                    continue
                else:
                    logger.warning("Gemini quota hit. Falling back to OpenRouter…")
                    break
            else:
                raise RuntimeError(f"Gemini failed: {e}") from e

    # OpenRouter fallback
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Gemini rate-limited and OPENROUTER_API_KEY not set. "
            "Add it to .env or Streamlit secrets."
        )
    for attempt in range(MAX_RETRIES):
        try:
            return _call_openrouter(system_prompt, user_prompt, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err and attempt < MAX_RETRIES - 1:
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(
                    f"All LLM providers failed. Last error: {e}\n"
                    "Check your API keys and free-tier quotas."
                ) from e

    raise RuntimeError("LLM call failed after all retries.")


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


def parse_json_response(raw: str, schema_cls, retry_prompt: str = "") -> Any:
    """
    Parse raw LLM text as JSON into schema_cls (Pydantic model).
    Retries once with the error message if parsing fails.
    """
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        data = _lenient_json(text)
        return schema_cls(**data)
    except Exception as first_err:
        if not retry_prompt:
            raise ValueError(f"JSON parse failed: {first_err}\nRaw: {raw[:500]}") from first_err

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
            fixed_text = fixed_raw.strip()
            if fixed_text.startswith("```"):
                lines = fixed_text.split("\n")
                fixed_text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            data = _lenient_json(fixed_text)
            return schema_cls(**data)
        except Exception as second_err:
            raise ValueError(
                f"JSON parse failed even after retry.\n"
                f"First error: {first_err}\nSecond error: {second_err}\nRaw: {raw[:300]}"
            ) from second_err
