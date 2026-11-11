"""
nodes/file_processor.py – Extract text from uploaded files.

Supported:
- .pdf  → pdfplumber (typed) | Gemini vision (scanned / image-heavy)
- .docx → python-docx
- .pptx → python-pptx
- .png / .jpg / .jpeg / .webp / .gif → Gemini vision
- .txt  → plain read

Memory: processes one page at a time; never loads full corpus.
Caching: MD5 hash → data/cache/<hash>.txt so unchanged files are free.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
VISION_THRESHOLD = 0.4  # if >40 % of pages are image-heavy → Gemini vision


# ── Helpers ────────────────────────────────────────────────────────────────────

def _file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _cached(hash_: str) -> Optional[str]:
    p = CACHE_DIR / f"{hash_}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


def _save_cache(hash_: str, text: str) -> None:
    (CACHE_DIR / f"{hash_}.txt").write_text(text, encoding="utf-8")


# ── Extractors ─────────────────────────────────────────────────────────────────

def _extract_pdf(data: bytes) -> str:
    import pdfplumber

    texts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            texts.append(t)
    combined = "\n".join(texts)

    # Heuristic: if less than 100 chars per page on average → likely scanned
    avg = len(combined) / max(len(texts), 1)
    if avg < 100:
        logger.info("PDF appears scanned; routing to Gemini vision.")
        return _extract_via_gemini_vision(data, mime="application/pdf")
    return combined


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_pptx(data: bytes) -> str:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    lines = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                lines.append(shape.text)
    return "\n".join(lines)


def _extract_image(data: bytes, mime: str = "image/jpeg") -> str:
    return _extract_via_gemini_vision(data, mime=mime)


def _extract_via_gemini_vision(data: bytes, mime: str = "application/pdf") -> str:
    """Send file to Gemini vision and get back clean text."""
    from google import genai as genai_new
    from google.genai import types as genai_types

    from llm_client import REQUEST_TIMEOUT_MS, gemini_model

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set; cannot read a scanned file.")
    client = genai_new.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )

    prompt = (
        "Transcribe this document faithfully. Preserve question numbers, "
        "marks in brackets, unit headings, and all text. "
        "Flag illegible words as [unclear]. Return only the transcribed text, no commentary."
    )
    part = genai_types.Part.from_bytes(data=data, mime_type=mime)
    response = client.models.generate_content(
        model=gemini_model(),
        contents=[prompt, part]
    )
    try:
        text = response.text
    except Exception:  # blocked or empty candidate
        text = None
    if not text:
        raise RuntimeError("Gemini returned no readable text for this file.")
    return text


# ── Public API ─────────────────────────────────────────────────────────────────

MIME_MAP: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def extract_text(file_data: bytes, filename: str) -> str:
    """
    Extract text from file bytes. Uses disk cache keyed by MD5.
    Returns extracted text string.
    """
    h = _file_hash(file_data)
    cached = _cached(h)
    if cached is not None:
        logger.info(f"Cache hit for {filename} ({h[:8]})")
        return cached

    ext = Path(filename).suffix.lower()
    logger.info(f"Extracting {filename} (ext={ext})")

    try:
        if ext == ".pdf":
            text = _extract_pdf(file_data)
        elif ext == ".docx":
            text = _extract_docx(file_data)
        elif ext == ".pptx":
            text = _extract_pptx(file_data)
        elif ext == ".txt":
            text = file_data.decode("utf-8", errors="replace")
        elif ext in IMAGE_EXTS:
            mime = MIME_MAP.get(ext, "image/jpeg")
            text = _extract_image(file_data, mime=mime)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Extraction failed for {filename}: {e}")
        raise

    _save_cache(h, text)
    return text


def process_uploads(
    uploads: List[Dict],  # [{"filename": str, "data": bytes, "role": str}]
) -> List[Dict]:
    """
    Process a list of uploaded files.
    Returns list of dicts with keys: filename, role, text.
    """
    results = []
    for upload in uploads:
        try:
            text = extract_text(upload["data"], upload["filename"])
            results.append({
                "filename": upload["filename"],
                "role": upload.get("role", "auto"),
                "text": text,
            })
        except Exception as e:
            results.append({
                "filename": upload["filename"],
                "role": upload.get("role", "auto"),
                "text": "",
                "error": str(e),
            })
    return results
