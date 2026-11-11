"""
schemas.py – Pydantic models for every LangGraph node's input/output.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, BeforeValidator, Field


# ── Lenient scalar types ───────────────────────────────────────────────────────
# LLMs answer in human units ("5 marks", "2023-24", "80%", "Unit 3"). Rejecting
# those burned a refresh and produced the same answer again, so parse the number
# out of whatever the model wrote instead of failing validation.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"expected a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _NUMBER_RE.search(value.replace(",", ""))
        if match:
            return float(match.group(0))
    raise ValueError(f"expected a number, got {value!r}")


def _to_int(value: Any) -> int:
    return int(round(_number(value)))


def _to_float(value: Any) -> float:
    return _number(value)


def _to_ratio(value: Any) -> float:
    """Scores/confidences the model wrote as 0-1, or loosely as 0-100 or "80%"."""
    if isinstance(value, str):
        fraction = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*", value)
        if fraction and float(fraction.group(2)):
            return float(fraction.group(1)) / float(fraction.group(2))
    number = _number(value)
    text = value if isinstance(value, str) else ""
    if "%" in text or (1 < number <= 100 and number.is_integer()):
        number /= 100
    return number


def _to_year(value: Any) -> int:
    """"2023-24", "May 2024" and "2024" all mean 2024; "2023-24" means 2023."""
    if isinstance(value, str):
        match = _YEAR_RE.search(value)
        if match:
            return int(match.group(0))
    return _to_int(value)


def _to_text(value: Any) -> str:
    return "" if value is None else str(value)


LenientInt = Annotated[int, BeforeValidator(_to_int)]
LenientFloat = Annotated[float, BeforeValidator(_to_float)]
LenientRatio = Annotated[float, BeforeValidator(_to_ratio)]
LenientYear = Annotated[int, BeforeValidator(_to_year)]
LenientStr = Annotated[str, BeforeValidator(_to_text)]


# ── Document / Classifier ──────────────────────────────────────────────────────

class ClassifiedDocument(BaseModel):
    id: str
    role: str  # syllabus | past_paper | teacher_priority | notes | pattern | unknown
    year: Optional[LenientYear] = None
    confidence: LenientRatio = Field(ge=0, le=1)
    clean_text: str


class ClassifierOutput(BaseModel):
    documents: List[ClassifiedDocument]


# ── Syllabus ───────────────────────────────────────────────────────────────────

class SyllabusConcept(BaseModel):
    topic: str
    concepts: List[str] = []


class SyllabusUnit(BaseModel):
    unit: LenientInt
    title: str
    hours: Optional[LenientFloat] = None
    weightage_marks: Optional[LenientFloat] = None
    topics: List[SyllabusConcept] = []


class SyllabusOutput(BaseModel):
    subject: str
    units: List[SyllabusUnit]
    warnings: List[str] = []


# ── Past Papers ────────────────────────────────────────────────────────────────

class PastQuestion(BaseModel):
    year: LenientYear
    exam_type: str = ""
    q_no: LenientStr = ""
    text: str
    marks: LenientInt
    type: str  # mcq | short | long | numerical | coding | viva
    unit: LenientInt
    topic: str
    difficulty: str  # easy | medium | hard


class TopicStat(BaseModel):
    topic: str
    count: LenientInt
    years: List[LenientYear] = []
    total_marks: LenientInt
    last_seen: LenientYear


class PaperAnalystOutput(BaseModel):
    questions: List[PastQuestion]
    topic_stats: List[TopicStat]
    repeated_concepts: List[str] = []
    warnings: List[str] = []


# ── Teacher priority ───────────────────────────────────────────────────────────

class QuestionHint(BaseModel):
    topic: str
    hint: str


class TeacherOutput(BaseModel):
    must_study: List[str] = []
    important: List[str] = []
    deprioritized: List[str] = []
    question_hints: List[QuestionHint] = []
    exam_format_hints: List[str] = []
    warnings: List[str] = []


# ── Predictor factors ──────────────────────────────────────────────────────────

class TopicFactors(BaseModel):
    teacher: LenientRatio = Field(ge=0, le=1)
    frequency: LenientRatio = Field(ge=0, le=1)
    syllabus_weight: LenientRatio = Field(ge=0, le=1)
    recency: LenientRatio = Field(ge=0, le=1)
    type_fit: LenientRatio = Field(ge=0, le=1)


class TopicPrediction(BaseModel):
    topic: str
    unit: LenientInt
    factors: TopicFactors
    likely_question_types: List[str] = []
    evidence: str = ""
    confidence_note: str = ""
    # Computed by scoring.py:
    score: Optional[LenientFloat] = None
    label: Optional[str] = None
    percentage: Optional[LenientFloat] = None


class PredictorRawOutput(BaseModel):
    topics: List[TopicPrediction]


# ── Mock Paper ─────────────────────────────────────────────────────────────────

class MockQuestion(BaseModel):
    id: LenientStr
    section: str
    text: str
    marks: LenientInt
    type: str
    unit: LenientInt
    topic: str
    difficulty: str
    source_tag: str


class MockSection(BaseModel):
    name: str
    instructions: str
    questions: List[MockQuestion]


class MockPaperOutput(BaseModel):
    header: Dict[str, Any]
    sections: List[MockSection]
    marks_check: Dict[str, Any]


# ── Question Bank ──────────────────────────────────────────────────────────────

class BankQuestion(BaseModel):
    text: str
    marks: LenientInt
    type: str
    difficulty: str
    topic: str
    source_tag: str
    options: Optional[List[str]] = None
    correct_option: Optional[str] = None
    reason: Optional[str] = None


class QuestionBankUnit(BaseModel):
    unit: LenientInt
    must: List[BankQuestion] = []
    should: List[BankQuestion] = []
    optional: List[BankQuestion] = []


class QuestionBankOutput(BaseModel):
    units: List[QuestionBankUnit]


# ── Model Answers ──────────────────────────────────────────────────────────────

class ModelAnswer(BaseModel):
    # Models often shorten these to "id"/"question"/"text" — accept both.
    question_id: str = Field(validation_alias=AliasChoices("question_id", "id"))
    question_text: str = Field(validation_alias=AliasChoices("question_text", "question", "text"))
    marks: LenientInt
    answer: str
    marking_scheme: List[str] = []


class ModelAnswersOutput(BaseModel):
    answers: List[ModelAnswer]


# ── Flashcards / Formula Sheet ─────────────────────────────────────────────────

class Flashcard(BaseModel):
    front: str
    back: str
    topic: str


class FormulaUnit(BaseModel):
    unit: LenientInt
    content: str  # markdown


class FlashcardsOutput(BaseModel):
    flashcards: List[Flashcard]
    formula_units: List[FormulaUnit]


# ── Study Plan ─────────────────────────────────────────────────────────────────

class StudyBlock(BaseModel):
    topic: str
    activity: str  # learn | practice | revise | mock
    minutes: LenientInt


class StudyDay(BaseModel):
    date: str
    hours: LenientFloat
    blocks: List[StudyBlock]


class StudyPlanOutput(BaseModel):
    days: List[StudyDay]
    warnings: List[str] = []


# ── Practical / Viva ───────────────────────────────────────────────────────────

class VivaQuestion(BaseModel):
    question: str
    answer_points: List[str] = []
    follow_ups: List[str] = []
    topic: str = ""
    unit: Optional[LenientInt] = None
    kind: str = "theory"  # theory | application | practical | troubleshooting


class VivaOutput(BaseModel):
    questions: List[VivaQuestion] = []
    practical_tasks: List[str] = []
    warnings: List[str] = []


# ── Examiner ───────────────────────────────────────────────────────────────────

class ExaminerIssue(BaseModel):
    item_id: str
    check: str
    problem: str
    fix: str


class ExaminerOutput(BaseModel):
    verdict: str  # pass | fail
    issues: List[ExaminerIssue] = []


# ── Answer Grader ──────────────────────────────────────────────────────────────

class RubricPoint(BaseModel):
    point: str
    awarded: LenientFloat
    max: LenientFloat
    comment: str


class GraderOutput(BaseModel):
    marks_awarded: LenientFloat
    out_of: LenientFloat
    rubric: List[RubricPoint]
    missing_points: List[str] = []
    errors: List[str] = []
    improved_answer: str = ""
    tips: List[str] = []


# ── Shared LangGraph State ─────────────────────────────────────────────────────

class WorkflowState(BaseModel):
    """Typed state passed between LangGraph nodes."""

    # Inputs
    subject: str = ""
    exam_type: str = "End-sem"
    exam_pattern: Dict[str, Any] = Field(default_factory=dict)
    difficulty_split: Dict[str, float] = Field(default_factory=lambda: {"easy": 20, "medium": 50, "hard": 30})
    unit_focus: List[LenientInt] = Field(default_factory=list)
    include_topics: List[str] = Field(default_factory=list)
    exclude_topics: List[str] = Field(default_factory=list)
    exam_date: Optional[str] = None
    hours_per_day: LenientFloat = 4.0
    mode: str = "Creative"
    language: str = "hinglish"
    answer_length: Dict[str, str] = Field(default_factory=lambda: {
        "2": "2-3 lines", "5": "~150 words", "10": "~350 words"
    })
    outputs_wanted: List[str] = Field(default_factory=list)
    pasted_text: str = ""

    # Pipeline intermediates
    course_level: str = ""
    discipline: str = ""
    raw_documents: List[Dict[str, Any]] = Field(default_factory=list)
    classified_docs: Optional[ClassifierOutput] = None
    syllabus: Optional[SyllabusOutput] = None
    paper_analysis: Optional[PaperAnalystOutput] = None
    teacher_analysis: Optional[TeacherOutput] = None
    knowledge_base: Optional[Dict[str, Any]] = None
    predictions: Optional[List[TopicPrediction]] = None

    # Generated artifacts
    priority_list: Optional[List[TopicPrediction]] = None
    study_plan: Optional[StudyPlanOutput] = None
    mock_paper: Optional[MockPaperOutput] = None
    question_bank: Optional[QuestionBankOutput] = None
    model_answers: Optional[ModelAnswersOutput] = None
    flashcards_output: Optional[FlashcardsOutput] = None
    viva_output: Optional[VivaOutput] = None

    # Meta
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    progress: Dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
