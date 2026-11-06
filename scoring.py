"""
scoring.py – Topic scoring engine.

Formula (all factors 0-1):
  raw = 40*teacher + 25*frequency + 15*syllabus_weight + 10*recency + 10*type_fit
  Clamp to 0-100, then cap percentage display at 90.
  Teacher-must  floor  : 75
  Teacher-dep   ceiling: 35
  Labels: MUST >= 75 | SHOULD 50-74 | OPTIONAL < 50
"""

from __future__ import annotations

from typing import List

from schemas import TopicFactors, TopicPrediction

WEIGHTS = {
    "teacher": 40,
    "frequency": 25,
    "syllabus_weight": 15,
    "recency": 10,
    "type_fit": 10,
}

MUST_THRESHOLD = 75
SHOULD_THRESHOLD = 50
PERCENTAGE_CAP = 90
TEACHER_MUST_FLOOR = 75
TEACHER_DEP_CEILING = 35


def compute_score(factors: TopicFactors, is_must: bool = False, is_dep: bool = False) -> float:
    """Return 0-100 raw score from factor object."""
    raw = (
        WEIGHTS["teacher"] * factors.teacher
        + WEIGHTS["frequency"] * factors.frequency
        + WEIGHTS["syllabus_weight"] * factors.syllabus_weight
        + WEIGHTS["recency"] * factors.recency
        + WEIGHTS["type_fit"] * factors.type_fit
    )
    raw = max(0.0, min(100.0, raw))

    if is_must:
        raw = max(raw, float(TEACHER_MUST_FLOOR))
    if is_dep:
        raw = min(raw, float(TEACHER_DEP_CEILING))

    return raw


def compute_percentage(score: float) -> float:
    """Scale score to a display percentage capped at PERCENTAGE_CAP."""
    return min(score, float(PERCENTAGE_CAP))


def label(score: float) -> str:
    if score >= MUST_THRESHOLD:
        return "MUST STUDY"
    if score >= SHOULD_THRESHOLD:
        return "SHOULD STUDY"
    return "OPTIONAL"


def score_topics(
    predictions: List[TopicPrediction],
    must_topics: List[str],
    dep_topics: List[str],
) -> List[TopicPrediction]:
    """
    Annotate each TopicPrediction with .score, .label, .percentage.
    Returns list sorted by score descending.
    """
    for tp in predictions:
        is_must = tp.topic in must_topics
        is_dep = tp.topic in dep_topics
        s = compute_score(tp.factors, is_must=is_must, is_dep=is_dep)
        tp.score = round(s, 1)
        tp.percentage = round(compute_percentage(s), 1)
        tp.label = label(s)

    predictions.sort(key=lambda x: x.score or 0, reverse=True)
    return predictions


def factor_breakdown(factors: TopicFactors) -> dict:
    """Human-readable per-factor contribution."""
    return {
        "teacher": round(WEIGHTS["teacher"] * factors.teacher, 1),
        "frequency": round(WEIGHTS["frequency"] * factors.frequency, 1),
        "syllabus_weight": round(WEIGHTS["syllabus_weight"] * factors.syllabus_weight, 1),
        "recency": round(WEIGHTS["recency"] * factors.recency, 1),
        "type_fit": round(WEIGHTS["type_fit"] * factors.type_fit, 1),
    }
