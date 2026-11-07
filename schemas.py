"""
schemas.py – Pydantic models for every LangGraph node's input/output.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Document / Classifier ──────────────────────────────────────────────────────

class ClassifiedDocument(BaseModel):
    id: str
    role: str  # syllabus | past_paper | teacher_priority | notes | pattern | unknown
    year: Optional[int] = None
    confidence: float = Field(ge=0, le=1)
    clean_text: str


class ClassifierOutput(BaseModel):
    documents: List[ClassifiedDocument]


# ── Syllabus ───────────────────────────────────────────────────────────────────

class SyllabusConcept(BaseModel):
    topic: str
    concepts: List[str] = []


class SyllabusUnit(BaseModel):
    unit: int
    title: str
    hours: Optional[float] = None
    weightage_marks: Optional[float] = None
    topics: List[SyllabusConcept] = []


class SyllabusOutput(BaseModel):
    subject: str
    units: List[SyllabusUnit]
    warnings: List[str] = []


# ── Past Papers ────────────────────────────────────────────────────────────────

class PastQuestion(BaseModel):
    year: int
    exam_type: str = ""
    q_no: str = ""
    text: str
    marks: int
    type: str  # mcq | short | long | numerical | coding | viva
    unit: int
    topic: str
    difficulty: str  # easy | medium | hard


class TopicStat(BaseModel):
    topic: str
    count: int
    years: List[int] = []
    total_marks: int
    last_seen: int


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
    teacher: float = Field(ge=0, le=1)
    frequency: float = Field(ge=0, le=1)
    syllabus_weight: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    type_fit: float = Field(ge=0, le=1)


class TopicPrediction(BaseModel):
    topic: str
    unit: int
    factors: TopicFactors
    likely_question_types: List[str] = []
    evidence: str = ""
    confidence_note: str = ""
    # Computed by scoring.py:
    score: Optional[float] = None
    label: Optional[str] = None
    percentage: Optional[float] = None


class PredictorRawOutput(BaseModel):
    topics: List[TopicPrediction]


# ── Mock Paper ─────────────────────────────────────────────────────────────────

class MockQuestion(BaseModel):
    id: str
    section: str
    text: str
    marks: int
    type: str
    unit: int
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
    marks: int
    type: str
    difficulty: str
    topic: str
    source_tag: str
    options: Optional[List[str]] = None
    correct_option: Optional[str] = None
    reason: Optional[str] = None


class QuestionBankUnit(BaseModel):
    unit: int
    must: List[BankQuestion] = []
    should: List[BankQuestion] = []
    optional: List[BankQuestion] = []


class QuestionBankOutput(BaseModel):
    units: List[QuestionBankUnit]


# ── Model Answers ──────────────────────────────────────────────────────────────

class ModelAnswer(BaseModel):
    question_id: str
    question_text: str
    marks: int
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
    unit: int
    content: str  # markdown


class FlashcardsOutput(BaseModel):
    flashcards: List[Flashcard]
    formula_units: List[FormulaUnit]


# ── Study Plan ─────────────────────────────────────────────────────────────────

class StudyBlock(BaseModel):
    topic: str
    activity: str  # learn | practice | revise | mock
    minutes: int


class StudyDay(BaseModel):
    date: str
    hours: float
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
    unit: Optional[int] = None
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
    awarded: float
    max: float
    comment: str


class GraderOutput(BaseModel):
    marks_awarded: float
    out_of: float
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
    unit_focus: List[int] = Field(default_factory=list)
    include_topics: List[str] = Field(default_factory=list)
    exclude_topics: List[str] = Field(default_factory=list)
    exam_date: Optional[str] = None
    hours_per_day: float = 4.0
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
