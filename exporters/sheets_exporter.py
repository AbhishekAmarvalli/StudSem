"""
exporters/sheets_exporter.py – Optional Google Sheets / Docs export.
Requires GOOGLE_EXPORT_ENABLED=true and a service account JSON.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

ENABLED = os.getenv("GOOGLE_EXPORT_ENABLED", "false").lower() == "true"
SA_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "secrets/service_account.json")


def _get_client():
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(SA_PATH, scopes=scopes)
    return gspread.authorize(creds)


def export_priority_to_sheet(topics: List[Any], subject: str) -> str:
    """Create a Google Sheet with the priority list. Returns the URL."""
    if not ENABLED:
        raise RuntimeError("Google export is disabled. Set GOOGLE_EXPORT_ENABLED=true.")

    gc = _get_client()
    sheet = gc.create(f"Exam Intelligence – {subject} – Priority List")
    ws = sheet.sheet1
    ws.update_title("Priority List")

    headers = ["Topic", "Unit", "Label", "Score", "Percentage", "Evidence", "Types"]
    rows = [headers]
    for t in topics:
        rows.append([
            t.topic, t.unit, t.label, t.score or 0, t.percentage or 0,
            t.evidence, ", ".join(t.likely_question_types),
        ])
    ws.update(rows)
    logger.info(f"Exported to Google Sheets: {sheet.url}")
    return sheet.url
