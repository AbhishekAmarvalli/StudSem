"""
exporters/pdf_exporter.py – PDF export using ReportLab only (no WeasyPrint).
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
STYLES = getSampleStyleSheet()


def _h1(text: str) -> Paragraph:
    s = ParagraphStyle("h1", parent=STYLES["Heading1"], fontSize=16, spaceAfter=6)
    return Paragraph(text, s)


def _h2(text: str) -> Paragraph:
    s = ParagraphStyle("h2", parent=STYLES["Heading2"], fontSize=13, spaceAfter=4)
    return Paragraph(text, s)


def _body(text: str) -> Paragraph:
    return Paragraph(str(text).replace("\n", "<br/>"), STYLES["Normal"])


def _spacer() -> Spacer:
    return Spacer(1, 0.3 * cm)


def export_priority_list(topics: List[Any], subject: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    story = [_h1(f"Priority List – {subject}"), _spacer()]

    for label in ("MUST STUDY", "SHOULD STUDY", "OPTIONAL"):
        section_topics = [t for t in topics if t.label == label]
        if not section_topics:
            continue
        story.append(_h2(label))
        for t in section_topics:
            row_text = (
                f"<b>{t.topic}</b> (Unit {t.unit}) – {t.percentage}%<br/>"
                f"Evidence: {t.evidence}<br/>"
                f"Types: {', '.join(t.likely_question_types)}"
            )
            story.append(_body(row_text))
            story.append(_spacer())

    doc.build(story)
    return buf.getvalue()


def export_mock_paper(mock_paper: Any, subject: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    story = []

    h = mock_paper.header
    story.append(_h1(h.get("title", f"Mock Paper – {subject}")))
    story.append(_body(f"Duration: {h.get('duration', '')} | Total Marks: {h.get('total_marks', '')}"))
    story.append(HRFlowable(width="100%"))
    story.append(_spacer())

    for section in mock_paper.sections:
        story.append(_h2(section.name))
        story.append(_body(section.instructions))
        story.append(_spacer())
        for q in section.questions:
            story.append(_body(
                f"<b>Q{q.id}.</b> [{q.marks} marks | {q.type} | {q.difficulty}]<br/>{q.text}"
            ))
            story.append(_spacer())
        story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()


def export_study_plan(plan: Any) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    story = [_h1("Study Plan"), _spacer()]

    for day in plan.days:
        story.append(_h2(f"{day.date} ({day.hours}h)"))
        for block in day.blocks:
            story.append(_body(f"• {block.topic} – {block.activity} ({block.minutes} min)"))
        story.append(_spacer())

    if plan.warnings:
        story.append(_h2("⚠ Warnings"))
        for w in plan.warnings:
            story.append(_body(f"• {w}"))

    doc.build(story)
    return buf.getvalue()


def export_viva(viva: Any, subject: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    story = [_h1(f"Practical / Viva Prep – {subject}"), _spacer()]

    if viva.practical_tasks:
        story.append(_h2("Practical tasks to be ready for"))
        for task in viva.practical_tasks:
            story.append(_body(f"• {task}"))
        story.append(_spacer())

    groups = (
        ("theory", "Theory checks"),
        ("application", "Application / why-how"),
        ("practical", "Practical & lab"),
        ("troubleshooting", "Troubleshooting"),
    )
    grouped = {key: [q for q in viva.questions if (q.kind or "theory") == key] for key, _ in groups}
    leftovers = [q for q in viva.questions if (q.kind or "theory") not in grouped]

    for key, heading in groups:
        questions = grouped[key]
        if not questions:
            continue
        story.append(_h2(heading))
        for index, q in enumerate(questions, start=1):
            story.append(_body(f"<b>Q{index}.</b> {q.question}"))
            if q.answer_points:
                story.append(_body("Answer points: " + "; ".join(q.answer_points)))
            if q.follow_ups:
                story.append(_body("Likely follow-ups: " + "; ".join(q.follow_ups)))
            story.append(_spacer())

    for index, q in enumerate(leftovers, start=1):
        story.append(_body(f"<b>Q{index}.</b> {q.question}"))
        story.append(_spacer())

    if viva.warnings:
        story.append(_h2("⚠ Warnings"))
        for warning in viva.warnings:
            story.append(_body(f"• {warning}"))

    doc.build(story)
    return buf.getvalue()


def export_flashcards(fc_output: Any) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    story = [_h1("Flashcards & Formula Sheet"), _spacer()]

    if fc_output.flashcards:
        story.append(_h2("Flashcards"))
        data = [["Front", "Back", "Topic"]]
        for fc in fc_output.flashcards:
            data.append([fc.front, fc.back, fc.topic])
        tbl = Table(data, colWidths=[6*cm, 8*cm, 4*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A90D9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
        story.append(_spacer())

    for fu in fc_output.formula_units:
        story.append(_h2(f"Unit {fu.unit} – Quick Revision"))
        story.append(_body(fu.content))
        story.append(_spacer())

    doc.build(story)
    return buf.getvalue()
