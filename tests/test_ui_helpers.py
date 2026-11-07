"""
tests/test_ui_helpers.py – pure helpers behind the redesigned UI.

Covers the input-absorbing rules (Postel's law) and the computed defaults
(Tesler's law) that the Streamlit pages rely on.
"""
from __future__ import annotations

import datetime

from ui import (
    LABEL_STYLES,
    compute_hard_split,
    days_left,
    detect_role,
    difficulty_warning,
    infer_question_count,
    parse_unit_focus,
    priority_badge,
    split_topics,
)


def test_parse_unit_focus_accepts_messy_input():
    assert parse_unit_focus("1, 2 & 5") == [1, 2, 5]
    assert parse_unit_focus("unit 1-3") == [1, 2, 3]
    assert parse_unit_focus("Units 2,4") == [2, 4]
    assert parse_unit_focus("1 2 5") == [1, 2, 5]
    assert parse_unit_focus("3-1") == [1, 2, 3]
    assert parse_unit_focus("") == []
    assert parse_unit_focus("all of them") == []


def test_parse_unit_focus_drops_impossible_units():
    assert parse_unit_focus("1, 0, 99") == [1]


def test_split_topics_trims_and_dedupes():
    assert split_topics(" Normalisation , indexing,,NORMALISATION ") == ["Normalisation", "indexing"]


def test_hard_split_is_derived_and_never_negative():
    assert compute_hard_split(20, 50) == 30
    assert compute_hard_split(90, 40) == 0
    assert compute_hard_split(0, 0) == 100


def test_difficulty_warning_fires_over_100():
    assert difficulty_warning(20, 50) is None
    assert "140%" in difficulty_warning(90, 50)


def test_infer_question_count_reads_section_patterns():
    pattern = "A: 4 x 5 marks, B: any 5 of 7 x 10 marks"
    assert infer_question_count(pattern, 100) == 9
    assert infer_question_count("12 questions", 100) == 12
    assert infer_question_count("any 5 of 7", 70) == 5
    assert infer_question_count("", 100) == 10
    assert infer_question_count("", 0, fallback=7) == 7


def test_detect_role_guesses_from_filename():
    assert detect_role("DS_Syllabus_2026.pdf") == "syllabus"
    assert detect_role("2024_end_sem_paper.pdf") == "past_paper"
    assert detect_role("teacher_hints.txt") == "teacher_priority"
    assert detect_role("unit3_notes.pptx") == "notes"
    assert detect_role("scan001.jpg") == "auto"


def test_days_left_counts_forward():
    assert days_left(datetime.date.today() + datetime.timedelta(days=7)) == 7
    assert days_left(None) is None


def test_priority_badges_always_carry_text_not_just_colour():
    assert "MUST STUDY" in priority_badge("MUST STUDY")
    assert "OPTIONAL" in priority_badge("OPTIONAL")
    # Unknown labels fall back instead of crashing.
    assert "OPTIONAL" in priority_badge("SOMETHING ELSE")
    # Every label has its own colour + text pair.
    assert set(LABEL_STYLES) == {"MUST STUDY", "SHOULD STUDY", "OPTIONAL"}
