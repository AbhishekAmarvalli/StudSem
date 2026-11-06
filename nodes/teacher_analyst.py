"""
nodes/teacher_analyst.py – Extract teacher priorities from files and pasted text.
"""
from __future__ import annotations

import logging
from typing import Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import TeacherOutput, ClassifierOutput, SyllabusOutput

logger = logging.getLogger(__name__)


def analyse_teacher(
    classified: ClassifierOutput,
    pasted_text: str,
    syllabus: Optional[SyllabusOutput],
) -> Optional[TeacherOutput]:
    teacher_docs = [d for d in classified.documents if d.role == "teacher_priority"]
    combined_text = "\n\n".join(d.clean_text for d in teacher_docs)
    if pasted_text:
        combined_text += f"\n\nPASTED TEXT (WhatsApp/typed):\n{pasted_text}"
    if not combined_text.strip():
        return None

    syllabus_json = syllabus.json() if syllabus else "{}"
    user_prompt = (
        build_prompt("P4")
        + f"\n\nSyllabus:\n{syllabus_json}\n\nTeacher inputs:\n{combined_text[:15000]}"
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, TeacherOutput, retry_prompt=user_prompt)


def teacher_node(state: dict) -> dict:
    classified = state.get("classified_docs")
    if classified is None:
        return {}
    result = analyse_teacher(
        classified,
        state.get("pasted_text", ""),
        state.get("syllabus"),
    )
    updates = {"progress": {**state.get("progress", {}), "teacher_analyst": "done"}}
    if result:
        updates["teacher_analysis"] = result
    return updates
