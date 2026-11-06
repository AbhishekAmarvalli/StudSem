"""
nodes/predictor.py – Runs LLM predictor (factor extraction) + scoring.py for final scores.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import PredictorRawOutput, TopicPrediction, WorkflowState
from scoring import score_topics

logger = logging.getLogger(__name__)


def run_predictor(state_dict: Dict[str, Any]) -> List[TopicPrediction]:
    syllabus = state_dict.get("syllabus")
    paper_analysis = state_dict.get("paper_analysis")
    teacher_analysis = state_dict.get("teacher_analysis")
    unit_focus = state_dict.get("unit_focus", [])
    include_topics = state_dict.get("include_topics", [])
    exclude_topics = state_dict.get("exclude_topics", [])
    exam_type = state_dict.get("exam_type", "End-sem")

    if syllabus is None:
        logger.warning("No syllabus – cannot predict.")
        return []

    syllabus_json = syllabus.json() if hasattr(syllabus, "json") else json.dumps(syllabus)
    paper_json = paper_analysis.json() if paper_analysis and hasattr(paper_analysis, "json") else "{}"
    teacher_json = teacher_analysis.json() if teacher_analysis and hasattr(teacher_analysis, "json") else "{}"

    filters = {
        "unit_focus": unit_focus,
        "include_topics": include_topics,
        "exclude_topics": exclude_topics,
        "exam_type": exam_type,
    }

    user_prompt = (
        build_prompt("P5")
        + f"\n\nSyllabus:\n{syllabus_json}"
        + f"\n\nPast paper stats:\n{paper_json}"
        + f"\n\nTeacher analysis:\n{teacher_json}"
        + f"\n\nFilters:\n{json.dumps(filters, indent=2)}"
        + "\n\nReturn a JSON object with key 'topics' containing the array."
    )

    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    result: PredictorRawOutput = parse_json_response(raw, PredictorRawOutput, retry_prompt=user_prompt)

    # Determine must/dep topic names for scoring floor/ceiling
    must_topics: List[str] = []
    dep_topics: List[str] = []
    if teacher_analysis:
        ta = teacher_analysis if hasattr(teacher_analysis, "must_study") else None
        if ta:
            must_topics = list(ta.must_study)
            dep_topics = list(ta.deprioritized)

    scored = score_topics(result.topics, must_topics, dep_topics)
    return scored


def predictor_node(state: dict) -> dict:
    topics = run_predictor(state)
    return {
        "predictions": topics,
        "priority_list": topics,
        "progress": {**state.get("progress", {}), "predictor": "done"},
    }
