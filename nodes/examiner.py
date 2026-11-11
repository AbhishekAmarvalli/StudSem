"""
nodes/examiner.py – Groq-powered artifact validator with retry logic.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import load_prompt
from schemas import ExaminerOutput

logger = logging.getLogger(__name__)

MAX_EXAMINER_RETRIES = 2


def _build_exam_prompt(artifact: Any, syllabus: Any, pattern: Dict, mode: str, diff_split: Dict) -> str:
    return (
        load_prompt("P8")
        + f"\n\nMode: {mode}"
        + f"\n\nDifficulty split target: {diff_split}"
        + f"\n\nPattern: {json.dumps(pattern, indent=2)}"
        + f"\n\nSyllabus (abbreviated): {str(syllabus)[:3000]}"
        + f"\n\nArtifact to check:\n{json.dumps(artifact if isinstance(artifact, dict) else artifact.dict() if hasattr(artifact, 'dict') else str(artifact), indent=2)[:8000]}"
        + "\n\nReturn JSON with keys 'verdict' and 'issues'."
    )


def examine_artifact(
    artifact: Any,
    state: Dict[str, Any],
    generator_fn,
    generator_state: Dict[str, Any],
) -> tuple[Any, ExaminerOutput]:
    """
    Examine artifact. If fail, call generator_fn to regenerate (max 2 retries).
    Returns (final_artifact, examiner_result).
    """
    syllabus = state.get("syllabus")
    pattern = state.get("exam_pattern", {})
    mode = state.get("mode", "Creative")
    diff_split = state.get("difficulty_split", {})

    for attempt in range(MAX_EXAMINER_RETRIES + 1):
        prompt = _build_exam_prompt(artifact, syllabus, pattern, mode, diff_split)
        try:
            raw = call_llm(system_prompt="", user_prompt=prompt, use_groq=True)
        except Exception as e:  # noqa: BLE001 - Groq is optional; never fail the run
            logger.warning(f"Examiner unavailable ({e}); keeping artifact unchecked.")
            return artifact, ExaminerOutput(verdict="skipped", issues=[])
        try:
            exam_result: ExaminerOutput = parse_json_response(raw, ExaminerOutput)
        except Exception as e:
            logger.error(f"Examiner parse failed: {e}")
            exam_result = ExaminerOutput(verdict="pass", issues=[])

        if exam_result.verdict == "pass" or attempt == MAX_EXAMINER_RETRIES:
            return artifact, exam_result

        # Regenerate with feedback
        issues_text = "\n".join(
            f"- [{i.check}] {i.problem} → Fix: {i.fix}" for i in exam_result.issues
        )
        logger.info(f"Examiner failed (attempt {attempt+1}). Regenerating with issues:\n{issues_text}")
        generator_state["_examiner_feedback"] = issues_text
        try:
            artifact = generator_fn(generator_state)
        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            break

    return artifact, exam_result


def examiner_node(state: dict) -> dict:
    """Check mock paper if present."""
    from nodes.generators import generate_mock_paper

    mock_paper = state.get("mock_paper")
    if mock_paper is None:
        return {}

    final, result = examine_artifact(
        artifact=mock_paper,
        state=state,
        generator_fn=generate_mock_paper,
        generator_state=dict(state),
    )
    updates: Dict[str, Any] = {
        "progress": {**state.get("progress", {}), "examiner": "done"},
    }
    if final is not mock_paper:
        updates["mock_paper"] = final
    if result.verdict == "fail":
        warnings = state.get("warnings", [])
        for issue in result.issues:
            warnings.append(f"[Examiner warning] {issue.check}: {issue.problem}")
        updates["warnings"] = warnings
    elif result.verdict == "skipped":
        updates["warnings"] = state.get("warnings", []) + [
            "Examiner skipped — Groq key missing or the examiner call failed. "
            "The mock paper was generated but not cross-checked."
        ]
    return updates
