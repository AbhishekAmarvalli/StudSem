"""
nodes/paper_analyst.py – Extract questions from past papers.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import PaperAnalystOutput, ClassifierOutput, SyllabusOutput

logger = logging.getLogger(__name__)


def analyse_papers(
    classified: ClassifierOutput,
    syllabus: Optional[SyllabusOutput],
) -> Optional[PaperAnalystOutput]:
    paper_docs = [d for d in classified.documents if d.role == "past_paper"]
    if not paper_docs:
        logger.warning("No past paper documents found.")
        return None

    syllabus_json = syllabus.json() if syllabus else "{}"
    combined = "\n\n---\n\n".join(
        f"Year: {d.year or 'unknown'}\n{d.clean_text}" for d in paper_docs
    )[:25000]

    user_prompt = (
        build_prompt("P3")
        + f"\n\nSyllabus (for topic mapping):\n{syllabus_json}\n\nPast papers:\n{combined}"
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, PaperAnalystOutput, retry_prompt=user_prompt)


def paper_node(state: dict) -> dict:
    classified = state.get("classified_docs")
    if classified is None:
        return {}
    result = analyse_papers(classified, state.get("syllabus"))
    updates = {"progress": {**state.get("progress", {}), "paper_analyst": "done"}}
    if result:
        updates["paper_analysis"] = result
    return updates
