"""
tests/test_schemas.py – Validate Pydantic schema parsing.
"""
import pytest
import json
from schemas import (
    ClassifierOutput, SyllabusOutput, PaperAnalystOutput, TeacherOutput,
    MockPaperOutput, ExaminerOutput, GraderOutput, FlashcardsOutput,
    StudyPlanOutput, WorkflowState,
)


def test_classifier_output_valid():
    data = {
        "documents": [
            {
                "id": "doc_0", "role": "syllabus", "year": None,
                "confidence": 0.95, "clean_text": "Unit 1: Introduction"
            }
        ]
    }
    out = ClassifierOutput(**data)
    assert out.documents[0].role == "syllabus"


def test_classifier_output_confidence_range():
    with pytest.raises(Exception):
        from schemas import ClassifiedDocument
        ClassifiedDocument(id="x", role="syllabus", confidence=1.5, clean_text="")


def test_syllabus_output_valid():
    data = {
        "subject": "Data Structures",
        "units": [
            {
                "unit": 1, "title": "Arrays",
                "topics": [{"topic": "1D Array", "concepts": ["declaration", "traversal"]}]
            }
        ],
        "warnings": []
    }
    out = SyllabusOutput(**data)
    assert out.units[0].unit == 1


def test_examiner_output_pass():
    out = ExaminerOutput(verdict="pass", issues=[])
    assert out.verdict == "pass"
    assert len(out.issues) == 0


def test_examiner_output_fail():
    data = {
        "verdict": "fail",
        "issues": [
            {"item_id": "Q1", "check": "syllabus", "problem": "No mapping", "fix": "Map to unit 2"}
        ]
    }
    out = ExaminerOutput(**data)
    assert out.verdict == "fail"
    assert out.issues[0].check == "syllabus"


def test_grader_output_valid():
    data = {
        "marks_awarded": 4,
        "out_of": 5,
        "rubric": [
            {"point": "Definition", "awarded": 2, "max": 2, "comment": "Correct"}
        ],
        "missing_points": ["Example missing"],
        "errors": [],
        "improved_answer": "Better answer here",
        "tips": ["Add an example"],
    }
    out = GraderOutput(**data)
    assert out.marks_awarded == 4


def test_workflow_state_defaults():
    state = WorkflowState(subject="OS")
    assert state.mode == "Creative"
    assert state.language == "hinglish"
    assert state.hours_per_day == 4.0


def test_past_question_parses_human_units():
    from schemas import PastQuestion
    q = PastQuestion(
        year="2023-24", q_no="Q4", text="Explain", marks="5 marks",
        type="short", unit="Unit 2", topic="Normalization", difficulty="easy",
    )
    assert (q.year, q.marks, q.unit, q.q_no) == (2023, 5, 2, "Q4")


def test_ratios_parse_percent_and_fractions():
    from schemas import TopicFactors
    factors = TopicFactors(
        teacher="80%", frequency=85, syllabus_weight="9/10", recency=0.5, type_fit=1,
    )
    assert factors.teacher == 0.8
    assert factors.frequency == 0.85
    assert factors.syllabus_weight == 0.9
    assert factors.recency == 0.5
    assert factors.type_fit == 1.0


def test_confidence_and_hours_parse_loose_numbers():
    from schemas import ClassifiedDocument, StudyDay
    doc = ClassifiedDocument(id="d", role="syllabus", confidence="95%", clean_text="")
    assert doc.confidence == 0.95
    day = StudyDay(date="2026-11-10", hours="2.5 h", blocks=[])
    assert day.hours == 2.5


def test_model_answer_marks_and_ids_coerce():
    from schemas import MockQuestion, ModelAnswer
    answer = ModelAnswer(
        question_id=3, question_text="Q", marks="7.5 / 10", answer="A",
    )
    assert answer.question_id == "3"
    assert answer.marks == 8
    question = MockQuestion(
        id=2, section="A", text="Q", marks="10", type="long",
        unit="Unit 1", topic="t", difficulty="hard", source_tag="new",
    )
    assert question.id == "2"
    assert question.unit == 1


def test_unparseable_numbers_still_fail():
    from schemas import PastQuestion
    with pytest.raises(Exception):
        PastQuestion(
            year="recent", q_no="1", text="Q", marks="many",
            type="short", unit=1, topic="t", difficulty="easy",
        )


def test_study_plan_output():
    data = {
        "days": [
            {
                "date": "2026-11-10",
                "hours": 4,
                "blocks": [
                    {"topic": "Arrays", "activity": "learn", "minutes": 60}
                ]
            }
        ],
        "warnings": []
    }
    out = StudyPlanOutput(**data)
    assert out.days[0].blocks[0].activity == "learn"
