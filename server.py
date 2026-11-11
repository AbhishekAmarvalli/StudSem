"""
server.py – local web UI backend for StudSem.

Run:  python server.py      →  http://127.0.0.1:8000

Endpoints
  GET  /                     neo-brutalist single-page UI
  GET  /api/health           which API keys are configured
  POST /api/run              start a workflow run (multipart: config + files)
  GET  /api/run/{id}/events  server-sent events: one event per completed stage
  GET  /api/run/{id}/result  the finished result as JSON
  GET  /api/run/{id}/download/{kind}   priority | mock | plan | flashcards | viva | dashboard
  POST /api/grade            grade one written answer

The workflow is the same LangGraph pipeline the Streamlit app uses; this file
only adds transport + progress reporting.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("studsem.server")

WEB_DIR = BASE_DIR / "webapp"

NODE_LABELS = {
    "extract": "Reading your documents",
    "classifier": "Sorting the documents",
    "syllabus_analyst": "Reading the syllabus",
    "paper_analyst": "Mining past papers",
    "teacher_analyst": "Reading teacher hints",
    "knowledge_base": "Building the topic map",
    "predictor": "Scoring topics",
    "mock_paper": "Drafting the mock paper",
    "examiner": "Examining the paper",
    "model_answers": "Writing model answers",
    "question_bank": "Building the question bank",
    "flashcards": "Making flashcards",
    "study_plan": "Planning your study days",
    "viva": "Writing practical / viva questions",
}

RUN_TTL_SECONDS = 6 * 60 * 60
RUNS: Dict[str, Dict[str, Any]] = {}
RUNS_LOCK = threading.Lock()

app = FastAPI(title="StudSem local UI", docs_url="/api/docs", redoc_url=None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cleanup_runs() -> None:
    cutoff = time.time() - RUN_TTL_SECONDS
    with RUNS_LOCK:
        for run_id in [rid for rid, run in RUNS.items() if run["created"] < cutoff]:
            RUNS.pop(run_id, None)


def _dump(model: Any) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return model


def _result_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    from scoring import factor_breakdown

    predictions = state.get("predictions") or []
    topics = []
    for t in predictions:
        factors = getattr(t, "factors", None)
        topics.append({
            "topic": t.topic,
            "unit": t.unit,
            "percentage": t.percentage,
            "score": t.score,
            "label": t.label,
            "evidence": t.evidence,
            "confidence_note": getattr(t, "confidence_note", ""),
            "question_types": list(getattr(t, "likely_question_types", []) or []),
            "factors": factor_breakdown(factors) if factors is not None else {},
        })

    return {
        "subject": state.get("subject", ""),
        "exam_type": state.get("exam_type", ""),
        "course_level": state.get("course_level", ""),
        "warnings": list(state.get("warnings") or []),
        "topics": topics,
        "study_plan": _dump(state.get("study_plan")),
        "mock_paper": _dump(state.get("mock_paper")),
        "viva": _dump(state.get("viva_output")),
        "flashcards": _dump(state.get("flashcards_output")),
        "model_answers": _dump(state.get("model_answers")),
        "question_bank": _dump(state.get("question_bank")),
    }


# ── Workflow worker ───────────────────────────────────────────────────────────

def _run_workflow_thread(run_id: str, state: Dict[str, Any]) -> None:
    events: "queue.Queue[Dict[str, Any]]" = RUNS[run_id]["events"]
    final_state: Dict[str, Any] = dict(state)

    try:
        raw_documents = state.get("raw_documents") or []
        if raw_documents:
            from nodes.file_processor import process_uploads

            events.put({"type": "stage", "node": "extract", "status": "start"})
            processed = process_uploads(raw_documents)
            for doc in processed:
                if doc.get("error"):
                    events.put({
                        "type": "warning",
                        "message": f"{doc['filename']}: {doc['error']}",
                    })
            final_state["raw_documents"] = processed
            events.put({"type": "stage", "node": "extract", "status": "done"})

        from graph import build_graph

        graph = build_graph(
            final_state.get("outputs_wanted", []),
            exam_type=final_state.get("exam_type", ""),
        )
        for chunk in graph.stream(final_state, {"recursion_limit": 50}):
            for node_name, node_output in chunk.items():
                if isinstance(node_output, dict):
                    final_state.update(node_output)
                    for warning in node_output.get("warnings") or []:
                        events.put({"type": "warning", "message": str(warning)})
                events.put({"type": "stage", "node": node_name, "status": "done"})

        RUNS[run_id]["state"] = final_state
        events.put({"type": "done", "node_count": len(final_state.get("progress") or {})})

    except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
        logger.exception("Workflow failed")
        message = str(exc)
        if "GEMINI_API_KEY" in message:
            message = (
                "No Gemini API key found. Add GEMINI_API_KEY to your .env file "
                "and restart the server."
            )
        RUNS[run_id]["error"] = message
        events.put({"type": "error", "message": message})
    finally:
        RUNS[run_id]["finished"] = True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    }


@app.post("/api/run")
async def start_run(
    config: str = Form(...),
    files: List[UploadFile] = File(default=[]),
) -> Dict[str, Any]:
    _cleanup_runs()

    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}") from exc

    subject = (cfg.get("subject") or "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="Subject name is required.")

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="No Gemini API key found. Add GEMINI_API_KEY to .env and restart the server.",
        )

    roles = cfg.get("file_roles") or {}
    raw_documents = []
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        raw_documents.append({
            "filename": upload.filename or "document",
            "data": data,
            "role": roles.get(upload.filename or "document", "auto"),
        })

    if not raw_documents and not (cfg.get("pasted_text") or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Add at least one document (syllabus, past paper, notes) or paste the teacher's hints.",
        )

    state: Dict[str, Any] = {
        "subject": subject,
        "exam_type": cfg.get("exam_type") or "End-sem",
        "course_level": cfg.get("course_level") or "",
        "discipline": cfg.get("discipline") or "",
        "exam_pattern": cfg.get("exam_pattern") or {},
        "difficulty_split": cfg.get("difficulty_split") or {"easy": 20, "medium": 50, "hard": 30},
        "unit_focus": cfg.get("unit_focus") or [],
        "include_topics": cfg.get("include_topics") or [],
        "exclude_topics": cfg.get("exclude_topics") or [],
        "exam_date": cfg.get("exam_date"),
        "hours_per_day": cfg.get("hours_per_day", 4.0),
        "mode": cfg.get("mode", "Creative"),
        "language": cfg.get("language", "hinglish"),
        "answer_length": cfg.get("answer_length") or {"2": "2-3 lines", "5": "~150 words", "10": "~350 words"},
        "outputs_wanted": cfg.get("outputs_wanted") or [],
        "pasted_text": cfg.get("pasted_text") or "",
        "viva_count": cfg.get("viva_count") or 15,
        "practical_focus": cfg.get("practical_focus") or "",
        "raw_documents": raw_documents,
        "warnings": [],
        "errors": [],
        "progress": {},
    }

    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = {
        "created": time.time(),
        "events": queue.Queue(),
        "state": None,
        "error": None,
        "finished": False,
        "subject": subject,
    }
    threading.Thread(target=_run_workflow_thread, args=(run_id, state), daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/run/{run_id}/events")
def run_events(run_id: str) -> StreamingResponse:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run id.")

    def stream():
        events = run["events"]
        while True:
            try:
                event = events.get(timeout=10)
            except queue.Empty:
                if run["finished"] and events.empty():
                    yield ": stream-closed\n\n"
                    break
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/run/{run_id}/result")
def run_result(run_id: str) -> JSONResponse:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run id.")
    if run["error"]:
        raise HTTPException(status_code=500, detail=run["error"])
    if run["state"] is None:
        raise HTTPException(status_code=409, detail="Run has not finished yet.")
    return JSONResponse(_result_payload(run["state"]))


@app.get("/api/run/{run_id}/download/{kind}")
def run_download(run_id: str, kind: str) -> Response:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run id.")
    state = run["state"]
    if state is None:
        raise HTTPException(status_code=409, detail="Run has not finished yet.")

    subject = state.get("subject") or "subject"
    safe_subject = "".join(c for c in subject if c.isalnum() or c in " -_").strip() or "subject"
    predictions = state.get("predictions") or []

    try:
        if kind == "priority":
            from exporters.pdf_exporter import export_priority_list

            return Response(
                export_priority_list(predictions, subject),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{safe_subject}_priority.pdf"'},
            )
        if kind == "mock":
            from exporters.pdf_exporter import export_mock_paper

            paper = state.get("mock_paper")
            if paper is None:
                raise HTTPException(status_code=404, detail="No mock paper in this run.")
            return Response(
                export_mock_paper(paper, subject),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{safe_subject}_mock_paper.pdf"'},
            )
        if kind == "plan":
            from exporters.pdf_exporter import export_study_plan

            plan = state.get("study_plan")
            if plan is None:
                raise HTTPException(status_code=404, detail="No study plan in this run.")
            return Response(
                export_study_plan(plan),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{safe_subject}_study_plan.pdf"'},
            )
        if kind == "flashcards":
            from exporters.pdf_exporter import export_flashcards

            cards = state.get("flashcards_output")
            if cards is None:
                raise HTTPException(status_code=404, detail="No flashcards in this run.")
            return Response(
                export_flashcards(cards),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{safe_subject}_flashcards.pdf"'},
            )
        if kind == "viva":
            from exporters.pdf_exporter import export_viva

            viva = state.get("viva_output")
            if viva is None:
                raise HTTPException(status_code=404, detail="No viva set in this run.")
            return Response(
                export_viva(viva, subject),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{safe_subject}_viva.pdf"'},
            )
        if kind == "dashboard":
            from exporters.html_exporter import render_dashboard

            html = render_dashboard({
                "subject": subject,
                "exam_type": state.get("exam_type", ""),
                "mode": state.get("mode", ""),
                "predictions": predictions,
                "study_plan": state.get("study_plan"),
                "mock_paper": state.get("mock_paper"),
                "warnings": state.get("warnings", []),
            })
            return Response(
                html.encode("utf-8"),
                media_type="text/html",
                headers={"Content-Disposition": f'attachment; filename="{safe_subject}_dashboard.html"'},
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Export failed")
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    raise HTTPException(status_code=404, detail=f"Unknown download kind: {kind}")


@app.post("/api/grade")
async def grade(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = (payload.get("question") or "").strip()
    answer = (payload.get("answer") or "").strip()
    if not question or not answer:
        raise HTTPException(status_code=400, detail="Both the question and your answer are required.")
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=400, detail="No API key configured. Add keys to .env.")

    try:
        from nodes.answer_grader import grade_answer

        result = grade_answer(
            question_text=question,
            student_answer=answer,
            marks=int(payload.get("marks") or 5),
            model_answer=payload.get("model_answer") or "",
            syllabus_context=payload.get("syllabus_context") or "",
        )
        return _dump(result) or {}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Grading failed")
        raise HTTPException(status_code=500, detail=f"Grading failed: {exc}") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"\n  StudSem running at http://127.0.0.1:{port}\n  Press Ctrl+C to stop.\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
