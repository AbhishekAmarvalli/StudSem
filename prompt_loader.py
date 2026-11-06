"""
prompt_loader.py – Loads prompts from prompts/ folder.
P0 is returned as system prompt; others as user prompt templates.
"""
from __future__ import annotations

import os
from functools import lru_cache

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load prompt file content (cached). name e.g. 'P0', 'P1', 'P6a'."""
    path = os.path.join(PROMPTS_DIR, f"{name}.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def system_prompt() -> str:
    """Return P0 – shared system rules."""
    return load_prompt("P0")


def build_prompt(key: str, **context) -> str:
    """
    Return P{key} content, optionally formatted with context kwargs.
    E.g. build_prompt('P2', syllabus_text='...')
    """
    template = load_prompt(key)
    if context:
        try:
            return template.format(**context)
        except KeyError:
            return template  # Return raw if keys don't match
    return template
