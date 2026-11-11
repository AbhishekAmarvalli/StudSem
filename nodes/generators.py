"""
nodes/generators.py – All content generation nodes.
Only the ones selected by the user are run.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import (
    MockPaperOutput, MockSection, QuestionBankOutput, QuestionBankUnit,
    ModelAnswersOutput, FlashcardsOutput, StudyPlanOutput, TopicPrediction, VivaOutput,
)

logger = logging.getLogger(__name__)

# Live chunk progress for the UI. A context variable keeps concurrent Streamlit
# sessions from seeing each other's messages.
_progress_reporter: ContextVar = ContextVar("generation_progress", default=None)


def set_progress_reporter(reporter: Optional[Callable[[str], None]]) -> object:
    """Route chunk messages to the UI; returns a token for reset_progress_reporter."""
    return _progress_reporter.set(reporter)


def reset_progress_reporter(token: object) -> None:
    _progress_reporter.reset(token)


def _report(message: str) -> None:
    reporter = _progress_reporter.get()
    if reporter is not None:
        reporter(message)


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


# "A: 7 x 2 marks" and "Section C: 7 questions" both ask for seven questions.
_SECTION_COUNT_PATTERNS = (
    re.compile(r"(\d+)\s*[x×*]\s*\d+\b", re.I),
    re.compile(r"(\d+)\s*questions?\b", re.I),
)


def _expected_section_questions(section: str) -> int:
    for pattern in _SECTION_COUNT_PATTERNS:
        for match in pattern.finditer(str(section)):
            try:
                count = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count
    return 0


def _section_label(spec: str) -> str:
    """'A: 7 x 2 marks' and 'Section A' both name section A."""
    head = str(spec).split(":", 1)[0].strip()
    match = re.search(r"([A-Za-z]+)\s*$", head)
    return (match.group(1) if match else head).strip()


def _display_section_name(label: str) -> Optional[str]:
    return f"Section {label.upper()}" if re.fullmatch(r"[A-Za-z]", label) else None


def _normalise_section_names(
    paper: MockPaperOutput, specs: List[str]
) -> MockPaperOutput:
    """Use one naming style ('Section A') whatever the model calls its sections."""
    labels = [_section_label(spec) for spec in specs]
    if not labels or len(paper.sections) != len(labels):
        return paper
    for section, label in zip(paper.sections, labels):
        display = _display_section_name(label)
        if display:
            section.name = display
    return paper


def _sections_named_in_feedback(feedback: str, sections: List[str]) -> set:
    named = set()
    for spec in sections:
        label = _section_label(spec)
        if not label:
            continue
        if re.search(rf"\bSection\s+{re.escape(label)}\b", feedback, re.I):
            named.add(label.lower())
        elif re.search(rf"\b{re.escape(label)}\d+\b", feedback):
            named.add(label.lower())
    return named


def _reusable_sections(
    state: Dict[str, Any], sections: List[str], previous: Any
) -> Dict[str, MockSection]:
    """Keep sections the examiner did not complain about instead of rebuilding them."""
    feedback = (state.get("_examiner_feedback") or "").strip()
    if not feedback or not previous or not getattr(previous, "sections", None):
        return {}
    named = _sections_named_in_feedback(feedback, sections)
    if not named:
        return {}
    reusable = {}
    for section in previous.sections:
        # Models name sections "Section B", patterns say "B: 7 x 5 marks";
        # grade both to the same letter before comparing.
        key = _section_label(section.name).lower()
        if key not in named:
            reusable[key] = copy.deepcopy(section)
    logger.info(
        "Reusing mock paper sections %s; regenerating %s",
        sorted(reusable), sorted(named),
    )
    return reusable


def _short_paper_warnings(
    paper: MockPaperOutput, pattern: Dict[str, Any], sections: List[str]
) -> List[str]:
    expected = sum(_expected_section_questions(item) for item in sections)
    expected = expected or _planned_question_count(pattern)
    actual = sum(len(item.questions) for item in paper.sections)
    if expected and actual < expected:
        return [f"Mock paper: only {actual} of {expected} questions were generated."]
    return []


def _next_question_id(current: str, used: set) -> str:
    match = re.match(r"^(.*?)(\d+)$", current or "")
    prefix = match.group(1) if match else (current or "Q")
    numbers = [
        int(found.group(1))
        for existing in used
        if (found := re.match(rf"^{re.escape(prefix)}(\d+)$", str(existing)))
    ]
    return f"{prefix}{max(numbers, default=0) + 1}"


def _merge_section_questions(
    sections: List[MockSection], extra: List[MockSection]
) -> None:
    by_name = {section.name.strip().lower(): section for section in sections}
    for incoming in extra:
        target = by_name.get((incoming.name or "").strip().lower())
        if target is None and len(sections) == 1:
            target = sections[0]
        if target is None:
            sections.append(incoming)
            by_name[(incoming.name or "").strip().lower()] = incoming
            continue
        used = {question.id for question in target.questions}
        for question in incoming.questions:
            if question.id in used:
                question.id = _next_question_id(question.id, used)
            used.add(question.id)
            target.questions.append(question)


def _top_up_section(
    section: str, prompt: str, part: MockPaperOutput
) -> MockPaperOutput:
    """Ask for the questions a section skipped instead of trusting the count."""
    expected = _expected_section_questions(section)
    written = sum(len(item.questions) for item in part.sections)
    missing = expected - written
    if missing <= 0:
        return part
    ids = ", ".join(
        question.id for item in part.sections for question in item.questions
    ) or "none"
    _report(
        f"Finishing mock paper section: {written}/{expected} written, adding {missing}"
    )
    continuation = (
        prompt
        + f"\n\nContinue the same section: you wrote {written} of the {expected} questions"
        + f" required by '{section}' (ids: {ids})."
        + f"\nWrite ONLY the remaining {missing} questions for that section, continuing the numbering."
        + "\nReturn the usual JSON shape with just those new questions in one section entry."
    )
    try:
        raw = call_llm(system_prompt=system_prompt(), user_prompt=continuation)
        extra = parse_json_response(raw, MockPaperOutput, retry_prompt=continuation)
    except Exception as exc:
        logger.warning("Mock paper top-up for %s failed: %s", section, exc)
        return part
    _merge_section_questions(part.sections, extra.sections)
    return part


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
    _report("Writing mock paper")
    raw = call_llm(system_prompt=system_prompt(), user_prompt=base_prompt)
    paper = parse_json_response(raw, MockPaperOutput, retry_prompt=base_prompt)
    paper = _normalise_section_names(paper, sections)
    if warnings is not None:
        warnings.extend(_short_paper_warnings(paper, pattern, sections))
    return paper


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
    previous = state.get("mock_paper")
    reusable = _reusable_sections(state, sections, previous)
    if reusable and previous is not None:
        header = dict(getattr(previous, "header", {}) or {})

    for index, section in enumerate(sections, 1):
        kept = reusable.get(_section_label(section).lower())
        if kept is not None:
            _report(f"Keeping mock paper section {index}/{len(sections)}: {section}")
            display = _display_section_name(_section_label(section))
            if display:
                kept.name = display
            merged.append(kept)
            continue
        _report(f"Writing mock paper section {index}/{len(sections)}: {section}")
        prompt = (
            base_prompt
            + f"\n\nThis paper is long, so write ONLY section {index} of {len(sections)}: {section}"
            + "\nRespect that section's question count, marks and choice rules exactly."
            + "\nReturn the usual JSON shape with just this one section in 'sections'; header may be empty."
        )
        try:
            raw = call_llm(system_prompt=system_prompt(), user_prompt=prompt)
            part = parse_json_response(raw, MockPaperOutput, retry_prompt=prompt)
            part = _top_up_section(section, prompt, part)
            part = _normalise_section_names(part, [section])
            if not any(item.questions for item in part.sections):
                raise RuntimeError("the model returned no questions for this section")
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

    if not header and previous is not None:
        header = dict(getattr(previous, "header", {}) or {})

    pattern = state.get("exam_pattern", {}) or {}
    notes.extend(
        _short_paper_warnings(
            MockPaperOutput(header=header, sections=merged, marks_check={}),
            pattern, sections,
        )
    )
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
        _report("Building question bank")
        user_prompt = _bank_prompt(state, _syllabus_json(syllabus), topics) + suffix
        raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
        return parse_json_response(raw, QuestionBankOutput, retry_prompt=user_prompt)

    notes = warnings if warnings is not None else []
    merged: List[QuestionBankUnit] = []
    failures: List[str] = []
    for position, group in enumerate(groups, 1):
        _report(
            f"Building question bank: units {group} (chunk {position}/{len(groups)})"
        )
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

# Answers are the longest artifact per question, so a section is split again.
# Both caps apply: five questions, and roughly 25 marks of answers per reply, so a
# section of 10-mark questions is split sooner than one of 2-mark questions.
ANSWERS_QUESTIONS_PER_CALL = 5
ANSWERS_MARKS_PER_CALL = 25


def _answer_batches(questions: List[Any]) -> List[List[Any]]:
    batches: List[List[Any]] = []
    current: List[Any] = []
    marks = 0
    for question in questions:
        weight = max(1, int(getattr(question, "marks", 0) or 0))
        if current and (
            len(current) >= ANSWERS_QUESTIONS_PER_CALL
            or marks + weight > ANSWERS_MARKS_PER_CALL
        ):
            batches.append(current)
            current, marks = [], 0
        current.append(question)
        marks += weight
    if current:
        batches.append(current)
    return batches


def _answers_prompt(
    state: Dict[str, Any],
    questions: List[Any],
    section_name: Optional[str] = None,
    batch: Optional[tuple] = None,
) -> str:
    questions_text = json.dumps([q.dict() for q in questions], indent=2)
    scope = ""
    if section_name:
        scope = f"\n\nAnswer ONLY the questions in section '{section_name}'."
        if batch:
            index, count = batch
            scope += (
                f" This is batch {index} of {count} for that section; answer only the"
                " questions listed below."
            )
        else:
            scope += " Do not write answers for any other section."
    return (
        build_prompt("P6c")
        + _exam_context(state)
        + f"\n\nAnswer length guide: {state.get('answer_length', {})}"
        + scope
        + f"\n\nQuestions:\n{questions_text}"
        + "\n\nReturn JSON with key 'answers'."
    )


def generate_model_answers(
    state: Dict[str, Any], warnings: Optional[List[str]] = None
) -> ModelAnswersOutput:
    mock_paper = state.get("mock_paper")

    if mock_paper is None:
        return ModelAnswersOutput(answers=[])

    sections = [section for section in mock_paper.sections if section.questions]
    if not sections:
        return ModelAnswersOutput(answers=[])

    # Answers are long, so long papers are written one section per call and a
    # section with many questions is split again into batches. A paper that fits
    # in one batch keeps the original single-call path.
    work = []
    for section in sections:
        batches = _answer_batches(section.questions)
        for index, batch in enumerate(batches, 1):
            work.append((section, batch, index, len(batches)))

    if len(work) == 1:
        _report("Writing model answers")
        user_prompt = _answers_prompt(state, work[0][1])
        raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
        return parse_json_response(raw, ModelAnswersOutput, retry_prompt=user_prompt)

    notes = warnings if warnings is not None else []
    answers: List[Any] = []
    failures: List[str] = []
    seen = set()

    for section, batch, index, count in work:
        batched = count > 1
        label = f"{section.name} (batch {index}/{count})" if batched else section.name
        _report(f"Writing model answers: {label}")
        user_prompt = _answers_prompt(
            state, batch, section.name, (index, count) if batched else None
        )
        try:
            raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
            part = parse_json_response(raw, ModelAnswersOutput, retry_prompt=user_prompt)
        except Exception as exc:
            logger.warning("Model answers for %s failed: %s", label, exc)
            failures.append(f"{label}: {exc}")
            continue
        for answer in part.answers:
            # Real replies sometimes omit ids/text, so include the body in the key
            # to avoid collapsing distinct answers into one.
            key = (answer.question_id, answer.question_text, answer.answer[:80])
            if key in seen:
                continue
            seen.add(key)
            answers.append(answer)

    if not answers:
        raise RuntimeError(
            "Model answer generation failed for every section: " + " | ".join(failures)
        )
    if failures:
        notes.append("Model answers: " + "; ".join(failures) + " — those sections are missing.")
    return ModelAnswersOutput(answers=answers)


def model_answers_node(state: dict) -> dict:
    notes: List[str] = []
    result = generate_model_answers(state, notes)
    update = {"model_answers": result, "progress": {**state.get("progress", {}), "model_answers": "done"}}
    if notes:
        update["warnings"] = [*(state.get("warnings") or []), *notes]
    return update


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
