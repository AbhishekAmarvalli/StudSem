"""
nodes/answer_grader.py – Grade a student's submitted answer.
"""
from __future__ import annotations

import logging
from typing import Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import GraderOutput

logger = logging.getLogger(__name__)


def grade_answer(
    question_text: str,
    student_answer: str,
    marks: int,
    model_answer: str = "",
    marking_scheme: list = None,
    syllabus_context: str = "",
) -> GraderOutput:
    marking_scheme = marking_scheme or []
    user_prompt = (
        build_prompt("P9")
        + f"\n\nQuestion: {question_text}"
        + f"\n\nTotal marks: {marks}"
        + (f"\n\nModel answer: {model_answer}" if model_answer else "")
        + (f"\n\nMarking scheme:\n" + "\n".join(f"- {s}" for s in marking_scheme) if marking_scheme else "")
        + (f"\n\nSyllabus context: {syllabus_context[:2000]}" if syllabus_context else "")
        + f"\n\nStudent's answer:\n{student_answer}"
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, GraderOutput, retry_prompt=user_prompt)
