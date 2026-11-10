"""
tests/test_wizard_flow.py – the four-step wizard keeps its inputs to the end.

Streamlit drops the state of widgets that are not rendered on a rerun, and the
wizard renders one step at a time. This test walks the real app end to end and
asserts the uploads, subject, exam type and preset still reach step 4.
"""
from __future__ import annotations

from pathlib import Path

import datetime

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _walk_to_run_step() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()

    at.file_uploader[0].upload("DS_Syllabus_2026.txt", b"Unit 1: Arrays\nUnit 2: Trees\n", "text/plain")
    at.run()

    at.button(key="next_1").click().run()
    at.text_input(key="subject").input("Data Structures").run()
    at.selectbox(key="exam_type").select("Practical-Viva").run()
    at.number_input(key="total_marks").set_value(70).run()
    at.button(key="next_2").click().run()

    at.radio(key="preset").set_value("Practice only — paper + answers").run()
    at.date_input(key="exam_date").set_value(datetime.date(2030, 4, 1)).run()
    at.button(key="next_3").click().run()
    return at


def test_inputs_survive_to_the_run_step():
    at = _walk_to_run_step()

    assert not at.exception
    assert at.session_state["step"] == 4

    answers = at.session_state["answers"]
    assert answers["subject"] == "Data Structures"
    assert answers["exam_type"] == "Practical-Viva"
    assert answers["total_marks"] == 70
    assert answers["preset"] == "Practice only — paper + answers"
    assert answers["exam_date"] == datetime.date(2030, 4, 1)
    assert [file.name for file in answers["files"]] == ["DS_Syllabus_2026.txt"]
    assert answers["file_roles"] == {"DS_Syllabus_2026.txt": "syllabus"}


def test_run_step_repeats_the_earlier_choices():
    at = _walk_to_run_step()

    # No "add a subject" / "add a document" complaints: both were supplied.
    assert not [error for error in at.error if "subject" in error.value.lower()]
    assert not [error for error in at.error if "document" in error.value.lower()]

    summary = "\n".join(markdown.value for markdown in at.markdown if "**Subject**" in markdown.value)
    assert "Data Structures" in summary
    assert "Practical-Viva" in summary
    assert "1 uploaded" in summary
    assert "Practice only" in summary
