"""
nodes/classifier.py – Classify each document into a role.
"""
from __future__ import annotations

import json
import logging
from typing import List, Dict

from llm_client import call_llm, parse_json_response
from prompt_loader import system_prompt, build_prompt
from schemas import ClassifierOutput

logger = logging.getLogger(__name__)


def classify_documents(raw_docs: List[Dict]) -> ClassifierOutput:
    """
    raw_docs: list of {"filename": str, "role": str, "text": str}
    If role != "auto", skip classification for that doc.
    Returns ClassifierOutput with all docs classified.
    """
    need_classify = [d for d in raw_docs if d.get("role", "auto") in ("auto", "")]
    manual = [d for d in raw_docs if d.get("role", "auto") not in ("auto", "")]

    classified_docs = []

    # Manual-role docs – trust user label
    for i, d in enumerate(manual):
        classified_docs.append({
            "id": f"doc_{i}",
            "role": d["role"],
            "year": None,
            "confidence": 1.0,
            "clean_text": d["text"],
        })

    offset = len(manual)

    if need_classify:
        docs_text = "\n\n---\n\n".join(
            f"DOCUMENT {i}: {d['filename']}\n{d['text'][:3000]}"
            for i, d in enumerate(need_classify)
        )
        user_prompt = (
            build_prompt("P1")
            + f"\n\nDocuments to classify:\n{docs_text}"
        )
        raw = call_llm(system_prompt=system_prompt(), user_prompt=user_prompt)
        try:
            result: ClassifierOutput = parse_json_response(raw, ClassifierOutput, retry_prompt=user_prompt)
            for j, doc in enumerate(result.documents):
                doc.id = f"doc_{offset + j}"
                # Use full text, not truncated
                if j < len(need_classify):
                    doc.clean_text = need_classify[j]["text"]
            classified_docs.extend([d.dict() for d in result.documents])
        except Exception as e:
            logger.error(f"Classifier LLM failed: {e}. Using raw text with 'unknown' role.")
            for j, d in enumerate(need_classify):
                classified_docs.append({
                    "id": f"doc_{offset + j}",
                    "role": "unknown",
                    "year": None,
                    "confidence": 0.0,
                    "clean_text": d["text"],
                })

    from schemas import ClassifiedDocument
    docs = []
    for cd in classified_docs:
        if isinstance(cd, ClassifiedDocument):
            docs.append(cd)
        elif isinstance(cd, dict):
            docs.append(ClassifiedDocument(**cd))
        else:
            docs.append(cd)
    return ClassifierOutput(documents=docs)


def classify_node(state: dict) -> dict:
    from schemas import WorkflowState, ClassifierOutput
    raw_docs = state.get("raw_documents", [])
    if not raw_docs:
        return {"warnings": state.get("warnings", []) + ["No documents to classify."]}

    result = classify_documents(raw_docs)
    return {"classified_docs": result, "progress": {**state.get("progress", {}), "classifier": "done"}}
