"""
tests/test_chunked_generation.py – Chunked generation behaviour.

The generators switch between one call and one call per chunk; these tests pin
down where the switch happens, how chunks are merged, and what a failed chunk
does to the artifact.
"""
import json
import re

import pytest

import nodes.generators as G
from schemas import (
    MockQuestion, MockPaperOutput, MockSection, SyllabusOutput, SyllabusUnit,
    TopicFactors, TopicPrediction,
)


def _topic(unit):
    return TopicPrediction(
        topic=f"Topic {unit}",
        unit=unit,
        factors=TopicFactors(
            teacher=0.5, frequency=0.5, syllabus_weight=0.5, recency=0.5, type_fit=0.5
        ),
    )


def _syllabus(units):
    return SyllabusOutput(
        subject="DBMS",
        units=[SyllabusUnit(unit=unit, title=f"Unit {unit}", topics=[]) for unit in units],
    )


def _paper_json(count=1, marks="10", name="A", first=1):
    return json.dumps({
        "header": {"title": "DBMS Mock"},
        "sections": [{
            "name": name,
            "instructions": "",
            "questions": [{
                "id": f"{name}{index}", "section": name, "text": "Question?",
                "marks": marks, "type": "long", "unit": 1, "topic": "Topic 1",
                "difficulty": "hard", "source_tag": "new",
            } for index in range(first, first + count)],
        }],
        "marks_check": {},
    })


def _bank_json(units):
    return json.dumps({"units": [{
        "unit": unit,
        "must": [
            {"text": f"Unit {unit} repeated", "marks": "5 marks", "type": "short",
             "difficulty": "easy", "topic": "t", "source_tag": "new"},
            {"text": f"Unit {unit} repeated", "marks": 5, "type": "short",
             "difficulty": "easy", "topic": "t", "source_tag": "new"},
            {"text": f"Unit {unit} unique", "marks": 5, "type": "short",
             "difficulty": "easy", "topic": "t", "source_tag": "new"},
        ],
        "should": [],
        "optional": [],
    } for unit in units]})


def _paper_fake(calls, failing=()):
    def fake(system_prompt=None, user_prompt="", **kwargs):
        calls.append(user_prompt)
        match = re.search(r"write ONLY section \d+ of \d+: (.+)", user_prompt)
        if not match:
            return _paper_json()
        spec = match.group(1)
        if any(name in spec for name in failing):
            raise RuntimeError("model returned prose")
        count = re.search(r"(\d+)\s*[x×]\s*\d+", spec)
        name = re.match(r"([A-Za-z]+)", spec)
        return _paper_json(
            int(count.group(1)) if count else 1,
            name=name.group(1) if name else "A",
        )
    return fake


def _bank_fake(calls, failing=()):
    def fake(system_prompt=None, user_prompt="", **kwargs):
        calls.append(user_prompt)
        match = re.search(r"Cover ONLY units \[(.*?)\]", user_prompt)
        if not match:
            return _bank_json([1, 2])
        units = json.loads("[" + match.group(1) + "]")
        if any(name in units for name in failing):
            raise RuntimeError("model returned prose")
        return _bank_json(units)
    return fake


def _answers_fake(calls, failing=()):
    def fake(system_prompt=None, user_prompt="", **kwargs):
        calls.append(user_prompt)
        match = re.search(r"Answer ONLY the questions in section '([^']+)'", user_prompt)
        name = match.group(1) if match else "A"
        if name in failing:
            raise RuntimeError("output cut off")
        return json.dumps({"answers": [{
            "question_id": f"{name}1", "question_text": f"{name} question",
            "marks": "10 marks", "answer": "Because.", "marking_scheme": ["1 mark"],
        }]})
    return fake


# ── Mock paper ────────────────────────────────────────────────────────────────

def test_small_paper_stays_single_call(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _paper_fake(calls))
    state = {
        "predictions": [_topic(1), _topic(2)],
        "syllabus": _syllabus([1, 2]),
        "exam_pattern": {"sections": ["A: 4 x 5", "B: 5 x 10"], "num_questions": 9},
    }
    G.generate_mock_paper(state)
    assert len(calls) == 1
    assert "write ONLY section" not in calls[0]


def test_large_paper_is_built_section_by_section(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _paper_fake(calls))
    state = {
        "predictions": [_topic(1), _topic(2)],
        "syllabus": _syllabus([1, 2]),
        "exam_pattern": {
            "sections": ["A: 5 x 10", "B: 5 x 10"],
            "num_questions": 20,
            "total_marks": 100,
        },
    }
    paper = G.generate_mock_paper(state)
    assert len(calls) == 2, "full sections need no top-up call"
    assert len(paper.sections) == 2
    assert paper.marks_check == {
        "total": 100, "computed": "per-section", "target": 100, "match": True,
    }
    assert paper.header["total_marks"] == 100


def test_section_shorter_than_planned_is_topped_up(monkeypatch):
    calls = []

    def fake(system_prompt=None, user_prompt="", **kwargs):
        calls.append(user_prompt)
        if not re.search(r"write ONLY section \d+ of \d+: (.+)", user_prompt):
            return _paper_json()
        if "Write ONLY the remaining" in user_prompt:
            assert "you wrote 2 of the 7 questions" in user_prompt
            assert "remaining 5 questions" in user_prompt
            return _paper_json(5, name="C")  # restarts numbering on purpose
        return _paper_json(2, name="C")

    monkeypatch.setattr(G, "call_llm", fake)
    state = {
        "predictions": [_topic(1)],
        "syllabus": _syllabus([1]),
        "exam_pattern": {
            "sections": ["A: 1 x 5", "C: 7 x 10 marks"], "num_questions": 20,
        },
    }
    warnings = []
    paper = G.generate_mock_paper(state, warnings)
    assert len(calls) == 3, "one call per section plus one top-up"
    section_c = paper.sections[1]
    ids = [question.id for question in section_c.questions]
    assert len(ids) == 7
    assert len(set(ids)) == 7, "top-up ids are renumbered instead of colliding"
    assert not warnings


def test_short_section_that_stays_short_warns(monkeypatch):
    calls = []

    def fake(system_prompt=None, user_prompt="", **kwargs):
        calls.append(user_prompt)
        match = re.search(r"write ONLY section \d+ of \d+: (.+)", user_prompt)
        if not match:
            return _paper_json()
        if match.group(1).startswith("A"):
            return _paper_json(1, name="A", marks="5")
        return _paper_json(2, name="C")

    monkeypatch.setattr(G, "call_llm", fake)
    state = {
        "predictions": [_topic(1)],
        "syllabus": _syllabus([1]),
        "exam_pattern": {
            "sections": ["A: 1 x 5", "C: 7 x 10 marks"], "num_questions": 20,
        },
    }
    warnings = []
    paper = G.generate_mock_paper(state, warnings)
    assert len(calls) == 3, "one top-up attempt per section"
    assert len(paper.sections[1].questions) == 4, "topped up, still short"
    assert any("5 of 8 questions" in note for note in warnings)


def _section_obj(name, count):
    return MockSection(name=name, instructions="", questions=[
        MockQuestion(
            id=f"{name}{index}", section=name, text=f"old {name}{index}", marks=10,
            type="long", unit=1, topic="t", difficulty="easy", source_tag="new",
        )
        for index in range(1, count + 1)
    ])


def test_examiner_retry_reuses_untouched_sections(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _paper_fake(calls))
    previous = MockPaperOutput(
        header={"title": "Old paper"}, marks_check={},
        sections=[_section_obj("Section A", 5), _section_obj("Section B", 5)],
    )
    state = {
        "predictions": [_topic(1)],
        "syllabus": _syllabus([1]),
        "mock_paper": previous,
        "_examiner_feedback": "- [pattern] Section B is incomplete: only 2 of 5 questions.",
        "exam_pattern": {
            "sections": ["A: 5 x 10", "B: 5 x 10"], "num_questions": 20,
        },
    }
    paper = G.generate_mock_paper(state)
    assert len(calls) == 1, "section A is kept, so only B is regenerated"
    assert "B: 5 x 10" in calls[0]
    assert paper.sections[0].name == "Section A", "kept section is unchanged"
    assert [question.text for question in paper.sections[0].questions] == [
        f"old Section A{index}" for index in range(1, 6)
    ]
    assert len(paper.sections[1].questions) == 5
    assert paper.header["title"] == "Old paper", "header survives reuse"


def test_examiner_retry_without_section_names_rebuilds_all(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _paper_fake(calls))
    previous = MockPaperOutput(
        header={}, marks_check={},
        sections=[_section_obj("Section A", 5), _section_obj("Section B", 5)],
    )
    state = {
        "predictions": [_topic(1)],
        "syllabus": _syllabus([1]),
        "mock_paper": previous,
        "_examiner_feedback": "- [duplicates] Some questions overlap.",
        "exam_pattern": {
            "sections": ["A: 5 x 10", "B: 5 x 10"], "num_questions": 20,
        },
    }
    G.generate_mock_paper(state)
    assert len(calls) == 2, "no identifiable section means regenerate everything"


def test_section_that_comes_back_empty_is_reported_not_dropped(monkeypatch):
    def fake(system_prompt=None, user_prompt="", **kwargs):
        if "write ONLY section 2 of 2: B" in user_prompt:
            return json.dumps({
                "header": {},
                "sections": [{
                    "name": "Section B", "instructions": "",
                    "questions": json.loads(_paper_json(5, name="B", marks="10"))["sections"][0]["questions"],
                }],
                "marks_check": {},
            })
        return json.dumps({"header": {}, "sections": [], "marks_check": {}})

    monkeypatch.setattr(G, "call_llm", fake)
    warnings = []
    paper = G.generate_mock_paper(
        {
            "predictions": [_topic(1)],
            "syllabus": _syllabus([1]),
            "exam_pattern": {
                "sections": ["A: 5 x 10", "B: 5 x 10"], "num_questions": 20,
            },
        },
        warnings,
    )
    assert [section.name for section in paper.sections] == ["Section B"]
    assert any("A: 5 x 10" in warning and "missing" in warning for warning in warnings)
    assert any("only 5 of 10 questions" in warning for warning in warnings)


def test_examiner_loop_reuses_kept_sections(monkeypatch):
    """Drive the real examiner retry loop: one rejection, one section rebuilt."""
    import nodes.examiner as E

    generator_calls = []
    monkeypatch.setattr(G, "call_llm", _paper_fake(generator_calls))
    previous = MockPaperOutput(
        header={}, marks_check={},
        sections=[
            _section_obj("Section A", 5), _section_obj("Section B", 5),
            _section_obj("Section C", 5),
        ],
    )
    verdicts = [
        {"verdict": "fail", "issues": [{
            "check": "pattern", "item_id": "B-3",
            "problem": "Section B is incomplete: only 3 of 5 questions.",
            "fix": "Write all 5 questions for Section B.",
        }]},
        {"verdict": "pass", "issues": []},
    ]
    examiner_calls = []

    def fake_examiner_call(*args, **kwargs):
        assert kwargs.get("use_groq"), "only the examiner should hit this stub"
        examiner_calls.append(1)
        return json.dumps(verdicts[min(len(examiner_calls) - 1, 1)])

    monkeypatch.setattr(E, "call_llm", fake_examiner_call)
    state = {
        "predictions": [_topic(1)],
        "syllabus": _syllabus([1]),
        "mock_paper": previous,
        "warnings": [],
        "exam_pattern": {
            "sections": ["A: 5 x 10", "B: 5 x 10", "C: 5 x 10"], "num_questions": 15,
        },
    }
    update = E.examiner_node(state)

    assert len(examiner_calls) == 2, "rejection then pass"
    assert len(generator_calls) == 1, "only the named section is rewritten"
    assert "write ONLY section 2 of 3: B: 5 x 10" in generator_calls[0]

    final = update["mock_paper"]
    assert [section.name for section in final.sections] == [
        "Section A", "Section B", "Section C"
    ]
    assert [question.id for question in final.sections[0].questions] == [
        f"Section A{index}" for index in range(1, 6)
    ], "section A is reused"
    assert [question.id for question in final.sections[1].questions] == [
        f"B{index}" for index in range(1, 6)
    ], "section B is regenerated"
    assert [question.id for question in final.sections[2].questions] == [
        f"Section C{index}" for index in range(1, 6)
    ], "section C is reused"
    assert "warnings" not in update, "a passing re-check leaves no warnings"


def test_single_call_paper_warns_when_it_is_short(monkeypatch):
    def fake(system_prompt=None, user_prompt="", **kwargs):
        return json.dumps({
            "header": {},
            "sections": [{"name": "A", "instructions": "", "questions": []}],
            "marks_check": {},
        })

    monkeypatch.setattr(G, "call_llm", fake)
    warnings = []
    G.generate_mock_paper(
        {
            "predictions": [_topic(1)],
            "syllabus": _syllabus([1]),
            "exam_pattern": {"sections": ["A: 4 x 5"], "num_questions": 4},
        },
        warnings,
    )
    assert warnings == ["Mock paper: only 0 of 4 questions were generated."]


def test_large_paper_partial_failure_keeps_the_rest(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _paper_fake(calls, failing=("B:",)))
    state = {
        "predictions": [_topic(1)],
        "syllabus": _syllabus([1, 2]),
        "exam_pattern": {"sections": ["A: 5 x 10", "B: 5 x 10"], "num_questions": 20},
    }
    warnings = []
    paper = G.generate_mock_paper(state, warnings)
    assert len(paper.sections) == 1
    assert "B: 5 x 10" in warnings[0]
    assert "missing" in warnings[0]


def test_large_paper_total_failure_raises(monkeypatch):
    def always_fail(system_prompt=None, user_prompt="", **kwargs):
        raise RuntimeError("model returned prose")

    monkeypatch.setattr(G, "call_llm", always_fail)
    state = {
        "predictions": [_topic(1)],
        "syllabus": _syllabus([1, 2]),
        "exam_pattern": {"sections": ["A: 5 x 10", "B: 5 x 10"], "num_questions": 20},
    }
    with pytest.raises(RuntimeError, match="every section"):
        G.generate_mock_paper(state)


# ── Question bank ─────────────────────────────────────────────────────────────

def test_question_bank_batches_units_and_dedupes(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _bank_fake(calls))
    state = {
        "predictions": [_topic(unit) for unit in range(1, 9)],
        "syllabus": _syllabus(range(1, 9)),
    }
    bank = G.generate_question_bank(state)
    assert len(calls) == 3
    assert [unit.unit for unit in bank.units] == list(range(1, 9))
    assert all(len(unit.must) == 2 for unit in bank.units), "duplicate texts dropped"
    assert bank.units[0].must[0].marks == 5, '"5 marks" coerced'


def test_question_bank_partial_failure_warns(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _bank_fake(calls, failing=(4,)))
    state = {
        "predictions": [_topic(unit) for unit in range(1, 7)],
        "syllabus": _syllabus(range(1, 7)),
    }
    warnings = []
    bank = G.generate_question_bank(state, warnings)
    assert [unit.unit for unit in bank.units] == [1, 2, 3]
    assert "units [4, 5, 6]" in warnings[0]


def test_question_bank_total_failure_raises(monkeypatch):
    def always_fail(system_prompt=None, user_prompt="", **kwargs):
        raise RuntimeError("model returned prose")

    monkeypatch.setattr(G, "call_llm", always_fail)
    state = {
        "predictions": [_topic(unit) for unit in range(1, 7)],
        "syllabus": _syllabus(range(1, 7)),
    }
    with pytest.raises(RuntimeError, match="every unit chunk"):
        G.generate_question_bank(state)


def test_small_question_bank_stays_single_call(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _bank_fake(calls))
    state = {"predictions": [_topic(1), _topic(2)], "syllabus": _syllabus([1, 2])}
    G.generate_question_bank(state)
    assert len(calls) == 1
    assert "Cover ONLY units" not in calls[0]


# ── Model answers ─────────────────────────────────────────────────────────────

def _mock_paper(sections=("A", "B")):
    return MockPaperOutput(header={}, marks_check={}, sections=[
        MockSection(name=name, instructions="", questions=[
            MockQuestion(
                id=f"{name}1", section=name, text=f"{name} question", marks=10,
                type="long", unit=1, topic="t", difficulty="hard", source_tag="new",
            )
        ])
        for name in sections
    ])


def test_model_answers_are_written_section_by_section(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _answers_fake(calls))
    result = G.generate_model_answers({"mock_paper": _mock_paper()})
    assert len(calls) == 2
    assert [answer.question_id for answer in result.answers] == ["A1", "B1"]
    assert result.answers[0].marks == 10, '"10 marks" coerced'
    assert "Answer ONLY the questions in section 'A'" in calls[0]


def test_model_answers_partial_failure_warns(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _answers_fake(calls, failing=("B",)))
    warnings = []
    result = G.generate_model_answers({"mock_paper": _mock_paper()}, warnings)
    assert [answer.question_id for answer in result.answers] == ["A1"]
    assert "B:" in warnings[0]
    assert "missing" in warnings[0]


def test_model_answers_total_failure_raises(monkeypatch):
    def always_fail(system_prompt=None, user_prompt="", **kwargs):
        raise RuntimeError("output cut off")

    monkeypatch.setattr(G, "call_llm", always_fail)
    with pytest.raises(RuntimeError, match="every section"):
        G.generate_model_answers({"mock_paper": _mock_paper()})


def test_large_section_is_split_into_question_batches(monkeypatch):
    calls = []

    def fake(system_prompt=None, user_prompt="", **kwargs):
        calls.append(user_prompt)
        ids = re.findall(r'"id": "([^"]+)"', user_prompt)
        return json.dumps({"answers": [{
            "question_id": qid, "question_text": f"Q{qid}", "marks": 5,
            "answer": "A", "marking_scheme": [],
        } for qid in ids]})

    monkeypatch.setattr(G, "call_llm", fake)
    questions = [
        MockQuestion(
            id=f"A{index}", section="A", text=f"Question {index}", marks=5,
            type="short", unit=1, topic="t", difficulty="easy", source_tag="new",
        )
        for index in range(1, 8)
    ]
    paper = MockPaperOutput(header={}, marks_check={}, sections=[
        MockSection(name="A", instructions="", questions=questions)
    ])
    result = G.generate_model_answers({"mock_paper": paper})
    assert len(calls) == 2, "seven questions split into batches of five"
    assert "batch 1 of 2" in calls[0]
    assert "batch 2 of 2" in calls[1]
    assert [answer.question_id for answer in result.answers] == [
        f"A{index}" for index in range(1, 8)
    ]


def test_long_mark_answers_split_before_five_questions(monkeypatch):
    calls = []

    def fake(system_prompt=None, user_prompt="", **kwargs):
        calls.append(user_prompt)
        ids = re.findall(r'"id": "([^"]+)"', user_prompt)
        return json.dumps({"answers": [{
            "question_id": qid, "question_text": f"Q{qid}", "marks": 10,
            "answer": "A", "marking_scheme": [],
        } for qid in ids]})

    monkeypatch.setattr(G, "call_llm", fake)
    questions = [
        MockQuestion(
            id=f"A{index}", section="A", text=f"Question {index}", marks=10,
            type="long", unit=1, topic="t", difficulty="hard", source_tag="new",
        )
        for index in range(1, 7)
    ]
    paper = MockPaperOutput(header={}, marks_check={}, sections=[
        MockSection(name="A", instructions="", questions=questions)
    ])
    result = G.generate_model_answers({"mock_paper": paper})
    assert len(calls) == 3, "two 10-mark answers per call stay inside 25 marks"
    assert [answer.question_id for answer in result.answers] == [
        f"A{index}" for index in range(1, 7)
    ]


def test_chunk_progress_is_reported(monkeypatch):
    messages = []
    token = G.set_progress_reporter(messages.append)
    try:
        monkeypatch.setattr(G, "call_llm", _paper_fake([]))
        monkeypatch.setattr(G, "last_provider", lambda: "Groq")
        state = {
            "predictions": [_topic(1)],
            "syllabus": _syllabus([1, 2]),
            "exam_pattern": {"sections": ["A: 5 x 10", "B: 5 x 10"], "num_questions": 20},
        }
        G.generate_mock_paper(state)
    finally:
        G.reset_progress_reporter(token)
    assert messages == [
        "Writing mock paper section 1/2: A: 5 x 10",
        "Finished mock paper section 1/2: A: 5 x 10 via Groq",
        "Writing mock paper section 2/2: B: 5 x 10",
        "Finished mock paper section 2/2: B: 5 x 10 via Groq",
    ]


def test_single_section_paper_stays_single_call(monkeypatch):
    calls = []
    monkeypatch.setattr(G, "call_llm", _answers_fake(calls))
    G.generate_model_answers({"mock_paper": _mock_paper(sections=("A",))})
    assert len(calls) == 1
    assert "Answer ONLY" not in calls[0]


def test_expected_section_questions_parses_patterns():
    assert G._expected_section_questions("A: 7 x 2 marks") == 7
    assert G._expected_section_questions("Section C: 7 questions") == 7
    assert G._expected_section_questions("B: 6 x 10 (answer any 4)") == 6
    assert G._expected_section_questions("A: 20 marks") == 0
