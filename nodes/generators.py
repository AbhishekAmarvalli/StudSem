"""
nodes/generators.py – All content generation nodes.
Only the ones selected by the user are run.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import (
    MockPaperOutput, MockSection, QuestionBankOutput, QuestionBankUnit,
    ModelAnswersOutput, FlashcardsOutput, StudyPlanOutput, TopicPrediction, VivaOutput,
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

# One reply has a finite budget. Above these sizes the paper and the bank are
# built in pieces, so a large subject can't lose its tail to the output limit.
MOCK_PAPER_QUESTION_LIMIT = 12
BANK_UNITS_PER_CALL = 3


def _planned_question_count(pattern: Dict[str, Any]) -> int:
    try:
        return int(pattern.get("num_questions") or 0)
    except (TypeError, ValueError):
        return 0


def _pattern_sections(pattern: Dict[str, Any]) -> List[str]:
    raw = pattern.get("sections") or []
    if isinstance(raw, str):
        raw = raw.split(",")
    return [str(item).strip() for item in raw if str(item).strip()]


def _syllabus_json(syllabus: Any, units: Optional[set] = None) -> str:
    if syllabus is None or not hasattr(syllabus, "json"):
        return "{}"
    if units is None:
        return syllabus.json()
    kept = [
        unit.dict() for unit in getattr(syllabus, "units", [])
        if int(getattr(unit, "unit", 0) or 0) in units
    ]
    return json.dumps({"subject": getattr(syllabus, "subject", ""), "units": kept}, indent=2)


def _mock_paper_prompt(state: Dict[str, Any]) -> str:
    topics = state.get("predictions", [])
    pattern = state.get("exam_pattern", {})
    mode = state.get("mode", "Creative")
    syllabus = state.get("syllabus")
    syllabus_json = syllabus.json() if syllabus and hasattr(syllabus, "json") else "{}"

    # The examiner's regeneration loop passes its issues here; without this the
    # retry sent an identical prompt and produced the same rejected paper.
    feedback = (state.get("_examiner_feedback") or "").strip()
    return (
        build_prompt("P6a")
        + _exam_context(state)
        + f"\n\nMode: {mode}"
        + f"\n\nExam pattern:\n{json.dumps(pattern, indent=2)}"
        + f"\n\nSyllabus:\n{syllabus_json}"
        + f"\n\nPriority topics:\n{_priority_summary(topics)}"
        + (f"\n\nA previous attempt was rejected. Fix these problems:\n{feedback}" if feedback else "")
    )


def generate_mock_paper(
    state: Dict[str, Any], warnings: Optional[List[str]] = None
) -> MockPaperOutput:
    base_prompt = _mock_paper_prompt(state)
    pattern = state.get("exam_pattern", {}) or {}
    sections = _pattern_sections(pattern)
    if len(sections) > 1 and _planned_question_count(pattern) > MOCK_PAPER_QUESTION_LIMIT:
        return _generate_mock_paper_by_section(state, base_prompt, sections, warnings)
    raw = call_llm(system_prompt=system_prompt(), user_prompt=base_prompt)
    return parse_json_response(raw, MockPaperOutput, retry_prompt=base_prompt)


def _generate_mock_paper_by_section(
    state: Dict[str, Any],
    base_prompt: str,
    sections: List[str],
    warnings: Optional[List[str]],
) -> MockPaperOutput:
    """Build a long paper one section per call so no single reply is truncated."""
    notes = warnings if warnings is not None else []
    merged: List[MockSection] = []
    header: Dict[str, Any] = {}
    failures: List[str] = []

    for index, section in enumerate(sections, 1):
        prompt = (
            base_prompt
            + f"\n\nThis paper is long, so write ONLY section {index} of {len(sections)}: {section}"
            + "\nRespect that section's question count, marks and choice rules exactly."
            + "\nReturn the usual JSON shape with just this one section in 'sections'; header may be empty."
        )
        try:
            raw = call_llm(system_prompt=system_prompt(), user_prompt=prompt)
            part = parse_json_response(raw, MockPaperOutput, retry_prompt=prompt)
        except Exception as exc:
            failures.append(f"{section}: {exc}")
            logger.warning("Mock paper section %s failed: %s", section, exc)
            continue
        if not header:
            header = dict(part.header or {})
        merged.extend(part.sections)

    if not merged:
        raise RuntimeError("Mock paper generation failed for every section: " + " | ".join(failures))
    if failures:
        notes.append(
            "Mock paper: " + "; ".join(failures) + " — those section(s) are missing."
        )

    pattern = state.get("exam_pattern", {}) or {}
    total = sum(int(getattr(q, "marks", 0) or 0) for s in merged for q in s.questions)
    marks_check: Dict[str, Any] = {"total": total, "computed": "per-section"}
    try:
        target = int(pattern.get("total_marks"))
    except (TypeError, ValueError):
        target = None
    if target is not None:
        marks_check["target"] = target
        # Internal choice ("any 5 of 7") means printed totals need not equal the
        # paper's target, so only claim a match when there is no choice.
        if not any(re.search(r"\bany\b[^,]*\bof\b", text, re.I) for text in sections):
            marks_check["match"] = total == target
        header.setdefault("total_marks", target)
    if pattern.get("duration") and not header.get("duration"):
        header["duration"] = pattern["duration"]

    return MockPaperOutput(header=header, sections=merged, marks_check=marks_check)


def mock_paper_node(state: dict) -> dict:
    notes: List[str] = []
    result = generate_mock_paper(state, notes)
    update = {"mock_paper": result, "progress": {**state.get("progress", {}), "mock_paper": "done"}}
    if notes:
        update["warnings"] = [*(state.get("warnings") or []), *notes]
    return update


# ── Question Bank ──────────────────────────────────────────────────────────────

def _bank_prompt(state: Dict[str, Any], syllabus_json: str, topics: List[TopicPrediction]) -> str:
    pattern = state.get("exam_pattern", {})
    return (
        build_prompt("P6b")
        + _exam_context(state)
        + f"\n\nDifficulty split: {state.get('difficulty_split', {})}"
        + f"\n\nQuestion types from pattern: {pattern.get('question_types', [])}"
        + f"\n\nSyllabus:\n{syllabus_json}"
        + f"\n\nPriority topics:\n{_priority_summary(topics)}"
    )


def _bank_unit_groups(state: Dict[str, Any]) -> List[List[int]]:
    """Group the subject's units into batches small enough for one reply."""
    units: set = set()
    for topic in state.get("predictions") or []:
        try:
            units.add(int(topic.unit))
        except (TypeError, ValueError):
            continue
    for unit in getattr(state.get("syllabus"), "units", []) or []:
        try:
            units.add(int(unit.unit))
        except (TypeError, ValueError):
            continue
    ordered = sorted(unit for unit in units if unit > 0)
    return [ordered[i:i + BANK_UNITS_PER_CALL] for i in range(0, len(ordered), BANK_UNITS_PER_CALL)]


def _merge_bank_units(units: List[QuestionBankUnit]) -> List[QuestionBankUnit]:
    by_unit: Dict[int, QuestionBankUnit] = {}
    for unit in units:
        merged = by_unit.setdefault(int(unit.unit), QuestionBankUnit(unit=int(unit.unit)))
        for bucket in ("must", "should", "optional"):
            existing = getattr(merged, bucket)
            seen = {q.text.strip().lower() for q in existing}
            for question in getattr(unit, bucket):
                key = question.text.strip().lower()
                if key not in seen:
                    seen.add(key)
                    existing.append(question)
    return [by_unit[key] for key in sorted(by_unit)]


def generate_question_bank(
    state: Dict[str, Any], warnings: Optional[List[str]] = None
) -> QuestionBankOutput:
    topics = state.get("predictions", [])
    syllabus = state.get("syllabus")
    suffix = "\n\nReturn JSON with key 'units' containing array of unit objects."
    groups = _bank_unit_groups(state)

    if len(groups) <= 1:
        user_prompt = _bank_prompt(state, _syllabus_json(syllabus), topics) + suffix
        raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
        return parse_json_response(raw, QuestionBankOutput, retry_prompt=user_prompt)

    notes = warnings if warnings is not None else []
    merged: List[QuestionBankUnit] = []
    failures: List[str] = []
    for group in groups:
        group_set = set(group)
        subset = [t for t in topics if getattr(t, "unit", None) in group_set]
        user_prompt = (
            _bank_prompt(state, _syllabus_json(syllabus, group_set), subset)
            + f"\n\nCover ONLY units {group}; do not add other units."
            + suffix
        )
        try:
            raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
            part = parse_json_response(raw, QuestionBankOutput, retry_prompt=user_prompt)
        except Exception as exc:
            failures.append(f"units {group}: {exc}")
            logger.warning("Question bank chunk %s failed: %s", group, exc)
            continue
        merged.extend(part.units)

    if not merged:
        raise RuntimeError(
            "Question bank generation failed for every unit chunk: " + " | ".join(failures)
        )
    if failures:
        notes.append("Question bank: " + "; ".join(failures) + " — those units are missing.")
    return QuestionBankOutput(units=_merge_bank_units(merged))


def question_bank_node(state: dict) -> dict:
    notes: List[str] = []
    result = generate_question_bank(state, notes)
    update = {"question_bank": result, "progress": {**state.get("progress", {}), "question_bank": "done"}}
    if notes:
        update["warnings"] = [*(state.get("warnings") or []), *notes]
    return update


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
