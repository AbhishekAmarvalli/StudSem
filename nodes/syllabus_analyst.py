"""
nodes/syllabus_analyst.py – Parse syllabus into structured JSON.
"""
from __future__ import annotations

import logging
from typing import Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import SyllabusOutput, ClassifierOutput

logger = logging.getLogger(__name__)


def analyse_syllabus(classified: ClassifierOutput, subject: str) -> Optional[SyllabusOutput]:
    syllabus_docs = [d for d in classified.documents if d.role == "syllabus"]
    if not syllabus_docs:
        logger.warning("No syllabus document found.")
        return None

    combined = "\n\n".join(d.clean_text for d in syllabus_docs)[:20000]
    user_prompt = (
        build_prompt("P2")
        + f"\n\nSubject: {subject}\n\nSyllabus text:\n{combined}"
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, SyllabusOutput, retry_prompt=user_prompt)


def syllabus_node(state: dict) -> dict:
    classified = state.get("classified_docs")
    if classified is None:
        return {"warnings": state.get("warnings", []) + ["Classifier output missing."]}
    subject = state.get("subject", "Unknown")
    result = analyse_syllabus(classified, subject)
    updates = {"progress": {**state.get("progress", {}), "syllabus_analyst": "done"}}
    if result is None:
        updates["warnings"] = state.get("warnings", []) + ["No syllabus found – prediction quality will be low."]
    else:
        updates["syllabus"] = result
    return updates
