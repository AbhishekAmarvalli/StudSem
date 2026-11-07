"""
nodes/generators.py – All content generation nodes.
Only the ones selected by the user are run.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import (
    MockPaperOutput, QuestionBankOutput, ModelAnswersOutput,
    FlashcardsOutput, StudyPlanOutput, TopicPrediction, VivaOutput,
)

logger = logging.getLogger(__name__)


def _priority_summary(topics: List[TopicPrediction]) -> str:
    lines = []
    for t in topics:
        lines.append(f"- [{t.label}] {t.topic} (unit {t.unit}) – {t.percentage}% – {t.evidence}")
    return "\n".join(lines)


def _exam_context(state: Dict[str, Any]) -> str:
    """Describe the course/exam so every generator adapts to the subject."""
    bits = [f"Subject/course: {state.get('subject') or 'not specified'}"]
    if state.get("course_level"):
        bits.append(f"Level: {state['course_level']}")
    if state.get("discipline"):
        bits.append(f"Discipline: {state['discipline']}")
    if state.get("exam_type"):
        bits.append(f"Exam type: {state['exam_type']}")
    return "\n\n" + " | ".join(bits)


# ── Mock Paper ─────────────────────────────────────────────────────────────────

def generate_mock_paper(state: Dict[str, Any]) -> MockPaperOutput:
    topics = state.get("predictions", [])
    pattern = state.get("exam_pattern", {})
    mode = state.get("mode", "Creative")
    syllabus = state.get("syllabus")
    syllabus_json = syllabus.json() if syllabus and hasattr(syllabus, "json") else "{}"

    user_prompt = (
        build_prompt("P6a")
        + _exam_context(state)
        + f"\n\nMode: {mode}"
        + f"\n\nExam pattern:\n{json.dumps(pattern, indent=2)}"
        + f"\n\nSyllabus:\n{syllabus_json}"
        + f"\n\nPriority topics:\n{_priority_summary(topics)}"
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, MockPaperOutput, retry_prompt=user_prompt)


def mock_paper_node(state: dict) -> dict:
    result = generate_mock_paper(state)
    return {"mock_paper": result, "progress": {**state.get("progress", {}), "mock_paper": "done"}}


# ── Question Bank ──────────────────────────────────────────────────────────────

def generate_question_bank(state: Dict[str, Any]) -> QuestionBankOutput:
    topics = state.get("predictions", [])
    pattern = state.get("exam_pattern", {})
    diff_split = state.get("difficulty_split", {})
    syllabus = state.get("syllabus")
    syllabus_json = syllabus.json() if syllabus and hasattr(syllabus, "json") else "{}"

    user_prompt = (
        build_prompt("P6b")
        + _exam_context(state)
        + f"\n\nDifficulty split: {diff_split}"
        + f"\n\nQuestion types from pattern: {pattern.get('question_types', [])}"
        + f"\n\nSyllabus:\n{syllabus_json}"
        + f"\n\nPriority topics:\n{_priority_summary(topics)}"
        + "\n\nReturn JSON with key 'units' containing array of unit objects."
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, QuestionBankOutput, retry_prompt=user_prompt)


def question_bank_node(state: dict) -> dict:
    result = generate_question_bank(state)
    return {"question_bank": result, "progress": {**state.get("progress", {}), "question_bank": "done"}}


# ── Model Answers ──────────────────────────────────────────────────────────────

def generate_model_answers(state: Dict[str, Any]) -> ModelAnswersOutput:
    mock_paper = state.get("mock_paper")
    answer_length = state.get("answer_length", {})

    if mock_paper is None:
        return ModelAnswersOutput(answers=[])

    questions_text = json.dumps(
        [q.dict() for s in mock_paper.sections for q in s.questions],
        indent=2,
    )[:20000]

    user_prompt = (
        build_prompt("P6c")
        + _exam_context(state)
        + f"\n\nAnswer length guide: {answer_length}"
        + f"\n\nQuestions:\n{questions_text}"
        + "\n\nReturn JSON with key 'answers'."
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, ModelAnswersOutput, retry_prompt=user_prompt)


def model_answers_node(state: dict) -> dict:
    result = generate_model_answers(state)
    return {"model_answers": result, "progress": {**state.get("progress", {}), "model_answers": "done"}}


# ── Flashcards + Formula Sheet ─────────────────────────────────────────────────

def generate_flashcards(state: Dict[str, Any]) -> FlashcardsOutput:
    topics = state.get("predictions", [])
    must_should = [t for t in topics if t.label in ("MUST STUDY", "SHOULD STUDY")]
    syllabus = state.get("syllabus")
    syllabus_json = syllabus.json() if syllabus and hasattr(syllabus, "json") else "{}"

    user_prompt = (
        build_prompt("P6d")
        + _exam_context(state)
        + f"\n\nSyllabus:\n{syllabus_json}"
        + f"\n\nMUST+SHOULD topics:\n{_priority_summary(must_should)}"
        + "\n\nReturn JSON with keys 'flashcards' and 'formula_units'."
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, FlashcardsOutput, retry_prompt=user_prompt)


def flashcards_node(state: dict) -> dict:
    result = generate_flashcards(state)
    return {"flashcards_output": result, "progress": {**state.get("progress", {}), "flashcards": "done"}}


# ── Study Plan ─────────────────────────────────────────────────────────────────

def generate_study_plan(state: Dict[str, Any]) -> StudyPlanOutput:
    topics = state.get("predictions", [])
    exam_date = state.get("exam_date", "")
    hours_per_day = state.get("hours_per_day", 4)
    import datetime
    today = datetime.date.today().isoformat()

    user_prompt = (
        build_prompt("P7")
        + _exam_context(state)
        + f"\n\nToday: {today}"
        + f"\n\nExam date: {exam_date}"
        + f"\n\nHours per day: {hours_per_day}"
        + f"\n\nTopics:\n{_priority_summary(topics)}"
        + "\n\nReturn JSON with key 'days'."
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, StudyPlanOutput, retry_prompt=user_prompt)


def study_plan_node(state: dict) -> dict:
    result = generate_study_plan(state)
    return {"study_plan": result, "progress": {**state.get("progress", {}), "study_plan": "done"}}


# ── Practical / Viva Questions ────────────────────────────────────────────────

def generate_viva(state: Dict[str, Any]) -> VivaOutput:
    topics = state.get("predictions", [])
    syllabus = state.get("syllabus")
    syllabus_json = syllabus.json() if syllabus and hasattr(syllabus, "json") else "{}"
    count = state.get("viva_count") or 15
    lab = (state.get("practical_focus") or "").strip()

    user_prompt = (
        build_prompt("P6e")
        + _exam_context(state)
        + f"\n\nNumber of questions wanted: about {count}"
        + (f"\n\nPractical/lab focus: {lab}" if lab else "")
        + f"\n\nSyllabus:\n{syllabus_json}"
        + f"\n\nPriority topics:\n{_priority_summary(topics)}"
        + "\n\nReturn JSON with keys 'questions' and 'practical_tasks'."
    )
    raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
    return parse_json_response(raw, VivaOutput, retry_prompt=user_prompt)


def viva_node(state: dict) -> dict:
    result = generate_viva(state)
    return {"viva_output": result, "progress": {**state.get("progress", {}), "viva": "done"}}
