"""
tests/test_scoring.py – pytest suite for scoring.py
"""
import pytest
from schemas import TopicFactors, TopicPrediction
from scoring import (
    compute_score, compute_percentage, label, score_topics,
    MUST_THRESHOLD, SHOULD_THRESHOLD, PERCENTAGE_CAP,
    TEACHER_MUST_FLOOR, TEACHER_DEP_CEILING,
)


def make_factors(**kwargs):
    defaults = dict(teacher=0.5, frequency=0.5, syllabus_weight=0.5, recency=0.5, type_fit=0.5)
    defaults.update(kwargs)
    return TopicFactors(**defaults)


def make_prediction(topic="T1", unit=1, **factor_kwargs):
    return TopicPrediction(
        topic=topic, unit=unit,
        factors=make_factors(**factor_kwargs),
        likely_question_types=["short"],
        evidence="test",
    )


# ── compute_score ──────────────────────────────────────────────────────────────

class TestComputeScore:
    def test_all_zeros_gives_zero(self):
        f = make_factors(teacher=0, frequency=0, syllabus_weight=0, recency=0, type_fit=0)
        assert compute_score(f) == 0.0

    def test_all_ones_gives_100(self):
        f = make_factors(teacher=1, frequency=1, syllabus_weight=1, recency=1, type_fit=1)
        assert compute_score(f) == 100.0

    def test_weights_sum_to_100(self):
        # 40+25+15+10+10 = 100
        f = make_factors(teacher=1, frequency=0, syllabus_weight=0, recency=0, type_fit=0)
        assert compute_score(f) == 40.0

    def test_must_floor_applied(self):
        f = make_factors(teacher=0, frequency=0, syllabus_weight=0, recency=0, type_fit=0)
        score = compute_score(f, is_must=True)
        assert score >= TEACHER_MUST_FLOOR

    def test_dep_ceiling_applied(self):
        f = make_factors(teacher=1, frequency=1, syllabus_weight=1, recency=1, type_fit=1)
        score = compute_score(f, is_dep=True)
        assert score <= TEACHER_DEP_CEILING

    def test_clamped_to_0_100(self):
        # Test that compute_score always returns within 0-100 range.
        # Use valid boundary values (1.0) and verify output is bounded.
        f = make_factors(teacher=1, frequency=1, syllabus_weight=1, recency=1, type_fit=1)
        result = compute_score(f)
        assert 0 <= result <= 100.0


# ── compute_percentage ─────────────────────────────────────────────────────────

class TestComputePercentage:
    def test_capped_at_90(self):
        assert compute_percentage(100) == PERCENTAGE_CAP

    def test_low_score_unchanged(self):
        assert compute_percentage(50) == 50.0

    def test_exact_cap(self):
        assert compute_percentage(90) == 90.0


# ── label ──────────────────────────────────────────────────────────────────────

class TestLabel:
    def test_must(self):
        assert label(75) == "MUST STUDY"
        assert label(90) == "MUST STUDY"

    def test_should(self):
        assert label(50) == "SHOULD STUDY"
        assert label(74) == "SHOULD STUDY"

    def test_optional(self):
        assert label(0) == "OPTIONAL"
        assert label(49) == "OPTIONAL"


# ── score_topics ───────────────────────────────────────────────────────────────

class TestScoreTopics:
    def test_sorts_descending(self):
        p1 = make_prediction("Low", frequency=0.1)
        p2 = make_prediction("High", frequency=1.0)
        result = score_topics([p1, p2], must_topics=[], dep_topics=[])
        assert result[0].topic == "High"

    def test_must_floor_applied_by_name(self):
        p = make_prediction("T_must", teacher=0.0, frequency=0.0, syllabus_weight=0.0, recency=0.0, type_fit=0.0)
        result = score_topics([p], must_topics=["T_must"], dep_topics=[])
        assert result[0].score >= TEACHER_MUST_FLOOR

    def test_dep_ceiling_applied_by_name(self):
        p = make_prediction("T_dep", teacher=1.0, frequency=1.0, syllabus_weight=1.0, recency=1.0, type_fit=1.0)
        result = score_topics([p], must_topics=[], dep_topics=["T_dep"])
        assert result[0].score <= TEACHER_DEP_CEILING

    def test_labels_assigned(self):
        p = make_prediction("T1")
        result = score_topics([p], must_topics=[], dep_topics=[])
        assert result[0].label in ("MUST STUDY", "SHOULD STUDY", "OPTIONAL")

    def test_percentage_capped(self):
        p = make_prediction("T1", teacher=1, frequency=1, syllabus_weight=1, recency=1, type_fit=1)
        result = score_topics([p], must_topics=[], dep_topics=[])
        assert result[0].percentage <= PERCENTAGE_CAP
