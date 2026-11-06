"""
nodes/knowledge_base.py – Merge analyst outputs into one JSON structure per subject.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from schemas import SyllabusOutput, PaperAnalystOutput, TeacherOutput

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def build_knowledge_base(
    subject: str,
    syllabus: Optional[SyllabusOutput],
    paper_analysis: Optional[PaperAnalystOutput],
    teacher_analysis: Optional[TeacherOutput],
) -> Dict[str, Any]:
    kb: Dict[str, Any] = {
        "subject": subject,
        "syllabus": syllabus.dict() if syllabus else None,
        "paper_analysis": paper_analysis.dict() if paper_analysis else None,
        "teacher_analysis": teacher_analysis.dict() if teacher_analysis else None,
    }
    # Persist to disk
    safe_name = "".join(c if c.isalnum() else "_" for c in subject)
    path = DATA_DIR / f"{safe_name}.json"
    path.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Knowledge base saved to {path}")
    return kb


def knowledge_base_node(state: dict) -> dict:
    kb = build_knowledge_base(
        subject=state.get("subject", "unknown"),
        syllabus=state.get("syllabus"),
        paper_analysis=state.get("paper_analysis"),
        teacher_analysis=state.get("teacher_analysis"),
    )
    return {
        "knowledge_base": kb,
        "progress": {**state.get("progress", {}), "knowledge_base": "done"},
    }
