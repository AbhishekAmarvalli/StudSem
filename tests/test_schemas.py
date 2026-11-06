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
