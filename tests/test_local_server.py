"""
tests/test_local_server.py – integration test for the local web UI backend.

The LangGraph nodes are stubbed so the test exercises the server plumbing
(validation, progress events, result JSON, exports) without calling an LLM.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import graph
import server
from schemas import (
    BankQuestion,
    Flashcard,
    FlashcardsOutput,
    FormulaUnit,
    MockPaperOutput,
    MockQuestion,
    MockSection,
    ModelAnswer,
    ModelAnswersOutput,
    QuestionBankOutput,
    QuestionBankUnit,
    StudyBlock,
    StudyDay,
    StudyPlanOutput,
    TopicFactors,
    TopicPrediction,
    VivaOutput,
    VivaQuestion,
)


def _topic(topic: str, unit: int, label: str, percentage: float) -> TopicPrediction:
    return TopicPrediction(
        topic=topic,
        unit=unit,
        factors=TopicFactors(teacher=0.8, frequency=0.6, syllabus_weight=0.7, recency=0.5, type_fit=0.6),
        likely_question_types=["short", "long"],
        evidence=f"{topic} appeared in 2 of the last 3 papers.",
        score=percentage,
        label=label,
        percentage=percentage,
    )


def _stub_graph(monkeypatch) -> None:
    predictions = [_topic("Normalisation", 3, "MUST STUDY", 82.0), _topic("Indexing", 4, "OPTIONAL", 41.0)]

    def simple(name, payload=None):
        def node(state):
            return {name: payload}
        return node

    monkeypatch.setattr(graph, "classify_node", simple("classified_docs", None))
    monkeypatch.setattr(graph, "syllabus_node", simple("syllabus", None))
    monkeypatch.setattr(graph, "paper_node", simple("paper_analysis", None))
    monkeypatch.setattr(graph, "teacher_node", simple("teacher_analysis", None))
    monkeypatch.setattr(graph, "knowledge_base_node", simple("knowledge_base", {}))
    monkeypatch.setattr(
        graph, "predictor_node", lambda state: {"predictions": predictions, "priority_list": predictions}
    )
    monkeypatch.setattr(
        graph,
        "mock_paper_node",
        lambda state: {
            "mock_paper": MockPaperOutput(
                header={"title": "Mock", "duration": "3 hours", "total_marks": 10},
                sections=[
                    MockSection(
                        name="Section A",
                        instructions="Answer all",
                        questions=[
                            MockQuestion(
                                id="1", section="A", text="Explain 3NF.", marks=5, type="long",
                                unit=3, topic="Normalisation", difficulty="medium", source_tag="past:2024",
                            )
                        ],
                    )
                ],
                marks_check={"total": 5},
            )
        },
    )
    monkeypatch.setattr(graph, "examiner_node", lambda state: {"examiner_output": {"verdict": "pass"}})
    monkeypatch.setattr(
        graph,
        "model_answers_node",
        lambda state: {
            "model_answers": ModelAnswersOutput(
                answers=[ModelAnswer(question_id="1", question_text="Explain 3NF.", marks=5,
                                     answer="Third normal form removes transitive dependencies.",
                                     marking_scheme=["Definition: 2 marks"])]
            )
        },
    )
    monkeypatch.setattr(
        graph,
        "question_bank_node",
        lambda state: {
            "question_bank": QuestionBankOutput(
                units=[QuestionBankUnit(unit=3, must=[BankQuestion(
                    text="Define 3NF.", marks=2, type="short", difficulty="easy",
                    topic="Normalisation", source_tag="new-in-syllabus",
                )])]
            )
        },
    )
    monkeypatch.setattr(
        graph,
        "flashcards_node",
        lambda state: {
            "flashcards_output": FlashcardsOutput(
                flashcards=[Flashcard(front="3NF", back="No transitive dependency", topic="Normalisation")],
                formula_units=[FormulaUnit(unit=3, content="1NF, 2NF, 3NF, BCNF")],
            )
        },
    )
    monkeypatch.setattr(
        graph,
        "study_plan_node",
        lambda state: {
            "study_plan": StudyPlanOutput(
                days=[StudyDay(date="2026-11-10", hours=4, blocks=[StudyBlock(
                    topic="Normalisation", activity="revise", minutes=60)])]
            )
        },
    )
    monkeypatch.setattr(
        graph,
        "viva_node",
        lambda state: {
            "viva_output": VivaOutput(
                questions=[VivaQuestion(
                    question="Why do we need 3NF?",
                    answer_points=["Removes transitive dependency", "Reduces update anomalies"],
                    follow_ups=["Give an example table that violates 3NF."],
                    topic="Normalisation",
                    unit=3,
                    kind="theory",
                )],
                practical_tasks=["Normalise a given table to 3NF on the board."],
            )
        },
    )


def _config(**overrides) -> dict:
    config = {
        "subject": "Database Systems",
        "exam_type": "Practical / Viva",
        "course_level": "Undergraduate",
        "discipline": "Engineering",
        "pasted_text": "Sir said normalisation comes for sure.",
        "outputs_wanted": ["mock_paper", "model_answers", "question_bank", "flashcards", "study_plan", "viva", "formula_sheet"],
    }
    config.update(overrides)
    return config


def _events(response_text: str) -> list[dict]:
    events = []
    for line in response_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_health_and_index(monkeypatch):
    client = TestClient(server.app)
    assert client.get("/api/health").json()["ok"] is True

    page = client.get("/")
    assert page.status_code == 200
    assert "StudSem" in page.text
    assert "viva-count" in page.text


def test_run_requires_subject(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = TestClient(server.app)
    response = client.post("/api/run", data={"config": json.dumps({"subject": "  "})})
    assert response.status_code == 400


def test_full_run_events_result_and_exports(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _stub_graph(monkeypatch)

    client = TestClient(server.app)
    started = client.post("/api/run", data={"config": json.dumps(_config())})
    assert started.status_code == 200
    run_id = started.json()["run_id"]

    events = _events(client.get(f"/api/run/{run_id}/events").text)
    types = [event["type"] for event in events]
    assert "done" in types, events
    stages = [event["node"] for event in events if event["type"] == "stage" and event["status"] == "done"]
    assert stages[:3] == ["classifier", "syllabus_analyst", "paper_analyst"]
    assert stages[-1] == "viva"

    result = client.get(f"/api/run/{run_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert [topic["label"] for topic in payload["topics"]] == ["MUST STUDY", "OPTIONAL"]
    assert payload["topics"][0]["factors"]["teacher"] == 32.0
    assert payload["viva"]["questions"][0]["kind"] == "theory"
    assert payload["mock_paper"]["sections"][0]["questions"][0]["id"] == "1"
    assert payload["study_plan"]["days"][0]["date"] == "2026-11-10"

    for kind in ("priority", "mock", "plan", "flashcards", "viva", "dashboard"):
        download = client.get(f"/api/run/{run_id}/download/{kind}")
        assert download.status_code == 200, (kind, download.text)
        if kind != "dashboard":
            assert download.content.startswith(b"%PDF"), kind

    assert client.get(f"/api/run/{run_id}/download/nope").status_code == 404


def test_generation_chunks_are_streamed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _stub_graph(monkeypatch)

    from nodes import generators

    def fake_call(system_prompt=None, user_prompt="", **kwargs):
        # Return the full section so the count check has no missing questions.
        return json.dumps({
            "header": {},
            "sections": [{"name": "A", "instructions": "", "questions": [{
                "id": str(index), "section": "A", "text": "Q",
                "marks": 10, "type": "long", "unit": 1, "topic": "t",
                "difficulty": "easy", "source_tag": "new",
            } for index in range(1, 6)]}],
            "marks_check": {},
        })

    def paper_node(state):
        return {"mock_paper": generators.generate_mock_paper({
            **state,
            "predictions": [],
            "exam_pattern": {"sections": ["A: 5 x 10", "B: 5 x 10"], "num_questions": 20},
        })}

    monkeypatch.setattr(generators, "call_llm", fake_call)
    monkeypatch.setattr(graph, "mock_paper_node", paper_node)

    client = TestClient(server.app)
    started = client.post("/api/run", data={"config": json.dumps(_config())})
    run_id = started.json()["run_id"]
    events = _events(client.get(f"/api/run/{run_id}/events").text)

    chunks = [event["message"] for event in events if event["type"] == "chunk"]
    assert chunks == [
        "Writing mock paper section 1/2: A: 5 x 10",
        "Finished mock paper section 1/2: A: 5 x 10",
        "Writing mock paper section 2/2: B: 5 x 10",
        "Finished mock paper section 2/2: B: 5 x 10",
    ]
    assert "done" in [event["type"] for event in events]


def test_grade_requires_both_fields(monkeypatch):
    client = TestClient(server.app)
    response = client.post("/api/grade", json={"question": "", "answer": "something"})
    assert response.status_code == 400
