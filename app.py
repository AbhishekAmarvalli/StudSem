"""
app.py – StudSem (Exam Intelligence Workflow), Streamlit presentation layer.

Four-step flow (Hick's law):
  1 Upload evidence → 2 Exam pattern → 3 Preferences → 4 Run & results

Backend untouched: graph.py, nodes/, scoring.py, schemas.py, llm_client.py,
exporters/. Everything visual lives in ui.py.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import types

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ── Secrets (.env locally, Streamlit secrets on Cloud) ────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import ui  # noqa: E402  (after dotenv so nothing reads half-loaded config)

st.set_page_config(
    page_title="StudSem — Exam Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:  # Cloud: secrets override .env keys
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

ui.inject_css()  # Principle 1: one theme, defined in ui.py + .streamlit/config.toml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Static copy ───────────────────────────────────────────────────────────────
STEP_LABELS = ("Upload", "Exam pattern", "Preferences", "Run & results")

# Principle 2: three presets replace seven checkboxes. Anything a preset leaves
# out is still reachable from the Advanced options expander.
PRESETS = {
    "Quick — priority list + study plan": ["priority_list", "study_plan"],
    "Full pack — everything": [
        "priority_list", "mock_paper", "model_answers", "study_plan",
        "flashcards", "formula_sheet", "question_bank",
    ],
    "Practice only — paper + answers": ["mock_paper", "model_answers"],
}

EXAM_TYPES = ["End-sem", "Mid-sem", "Unit test-MCQ", "Practical-Viva", "Project viva"]
COURSE_LEVELS = ["School (Class 9-12)", "Diploma / Polytechnic", "Undergraduate",
                 "Postgraduate", "Professional exam"]
DISCIPLINES = ["General", "Engineering", "Science", "Commerce", "Arts and Humanities",
               "Medicine and Health", "Law", "Management", "Education", "Agriculture"]

# Plain-language progress messages (Principle 6: instant, human feedback).
NODE_LABELS = {
    "classifier": "Sorting your documents",
    "syllabus_analyst": "Reading your syllabus",
    "paper_analyst": "Mining past papers",
    "teacher_analyst": "Applying teacher hints",
    "knowledge_base": "Building the topic map",
    "predictor": "Predicting topics",
    "mock_paper": "Writing the mock paper",
    "examiner": "Checking the mock paper",
    "model_answers": "Writing model answers",
    "question_bank": "Building the question bank",
    "flashcards": "Making flashcards",
    "study_plan": "Planning your study days",
    "viva": "Writing practical / viva questions",
}


# ── Session state (Principle 6: widgets and results survive reruns) ───────────
def _init_state() -> None:
    defaults = {
        "step": 1,
        "result": None,
        "run_meta": {},
        "feedback": None,
        "file_signature": "",
        "saved_subjects": [],
        "preset": list(PRESETS)[0],
        "answers": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


def _remember(**values) -> None:
    """Store the wizard's answers outside its widget keys.

    Streamlit drops a widget's state on the rerun after that widget stops being
    rendered, and the wizard only renders one step at a time. Without this
    snapshot, everything filled in on steps 1-3 is empty by step 4 — the run
    then complains about a missing subject and documents.
    """
    st.session_state.setdefault("answers", {}).update(values)


def _answer(key: str, default=None):
    return st.session_state.setdefault("answers", {}).get(key, default)


# ── Sidebar (Principles 4, 16: familiar rail, status first, help last) ────────
def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### Connection")
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))
        has_groq = bool(os.getenv("GROQ_API_KEY"))
        has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))

        st.markdown(
            ui.badge("Gemini " + ("ready" if has_gemini else "missing"),
                     "#c6ff4d" if has_gemini else "#ff5da2")
            + " "
            + ui.badge("Groq " + ("ready" if has_groq else "optional"),
                       "#c6ff4d" if has_groq else "#e6e1d6")
            + " "
            + ui.badge("Fallback " + ("ready" if has_openrouter else "off"),
                       "#c6ff4d" if has_openrouter else "#e6e1d6"),
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}")

        with st.expander("API keys", expanded=not has_gemini):
            gemini_key = st.text_input(
                "Gemini API key", value=os.getenv("GEMINI_API_KEY", ""), type="password",
                help="Free key at aistudio.google.com/apikey — required.",
            )
            groq_key = st.text_input(
                "Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password",
                help="Free key at console.groq.com — used by the examiner.",
            )
            openrouter_key = st.text_input(
                "OpenRouter key", value=os.getenv("OPENROUTER_API_KEY", ""), type="password",
                help="Optional fallback when Gemini hits its quota.",
            )
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
            if openrouter_key:
                os.environ["OPENROUTER_API_KEY"] = openrouter_key

        st.divider()
        try:
            st.page_link("pages/1_Answer_Grader.py", label="Grade an answer", icon="✏️")
        except Exception:  # noqa: BLE001 - page link is a convenience, never fatal
            pass
        if st.button("Start over", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != "saved_subjects":
                    del st.session_state[key]
            st.rerun()

        st.divider()
        # Principle 16: help sits last.
        with st.expander("How this works"):
            st.markdown(
                "1. Upload the syllabus, past papers and any teacher hints.\n"
                "2. Set the exam pattern.\n"
                "3. Pick what you want built.\n"
                "4. Run — every score comes from your own documents."
            )
            st.caption("Percentages are evidence-based estimates, capped at 90%. Nothing is guaranteed.")


_sidebar()


# ── Shared widgets ────────────────────────────────────────────────────────────
def _nav(step: int, can_continue: bool = True, hint: str = "") -> None:
    """
    Principle 3 (Fitts): large grouped Back / Next buttons; Next is the only
    filled accent and sits where the eye lands.
    """
    st.write("")
    back_col, next_col = st.columns([1, 2])
    if back_col.button("← Back", use_container_width=True, disabled=step == 1, key=f"back_{step}"):
        st.session_state.step = step - 1
        st.rerun()
    if next_col.button("Next →", type="primary", use_container_width=True,
                       disabled=not can_continue, key=f"next_{step}"):
        st.session_state.step = step + 1
        st.rerun()
    if hint and not can_continue:
        st.caption(hint)


# ── Step 1: Upload ────────────────────────────────────────────────────────────
def _step_upload() -> None:
    st.markdown("## 1 · Add your evidence")
    st.markdown(
        '<p class="ss-muted">Syllabus, past papers and teacher hints are what every '
        "prediction is built from. More evidence, better estimates.</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### Your files")
        files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="PDF, Word, PowerPoint, text, or photos of pages.",
        )

        # Principle 20: feedback the moment files are added.
        signature = ",".join(sorted(file.name for file in files or []))
        if signature and signature != st.session_state.get("file_signature"):
            st.session_state["file_signature"] = signature
            st.toast(f"{len(files)} file(s) added", icon="📄")

        roles = {}
        for file in files or []:
            # Principle 7: filename and its role selector sit on one row.
            name_col, role_col = st.columns([3, 2])
            guessed = st.session_state.get(f"role_{file.name}") or ui.detect_role(file.name)
            with name_col:
                st.markdown(
                    f'<div style="padding-top:.5rem">{ui.escape(file.name)} '
                    f"{ui.file_role_badge(guessed)}</div>",
                    unsafe_allow_html=True,
                )
            with role_col:
                roles[file.name] = st.selectbox(
                    f"Role for {file.name}",
                    ["auto", "syllabus", "past_paper", "teacher_priority", "notes", "pattern"],
                    index=["auto", "syllabus", "past_paper", "teacher_priority", "notes", "pattern"].index(guessed),
                    key=f"role_{file.name}",
                    label_visibility="collapsed",
                )

        # Principle 15: show what is still missing so the goal feels close.
        chosen = set(roles.values())
        ui.checklist("Evidence so far", {
            "Syllabus": "syllabus" in chosen,
            "Past papers": "past_paper" in chosen,
            "Teacher notes": "teacher_priority" in chosen,
        })

    with st.container(border=True):
        st.markdown("#### Teacher hints")
        st.text_area(
            "Paste anything the teacher said",
            key="hints",
            height=90,
            placeholder="e.g. Unit 3 and 5 se mostly questions aayenge, numericals from unit 4.",
            help="Teacher priority outweighs past-paper frequency in the scoring.",
        )

    # Snapshot the uploads and roles so step 4 can still build the state.
    # Always write the snapshot: skipping it when the list is empty kept files
    # the user had just removed, and step 4 then ran on stale uploads.
    _remember(files=list(files or []), file_roles=dict(roles))
    _remember(pasted_text=st.session_state.get("hints", ""))

    has_evidence = bool(files) or bool(st.session_state.get("hints", "").strip())
    _nav(
        1,
        can_continue=has_evidence,
        hint="Add at least one document or a hint before continuing.",
    )


# ── Step 2: Exam pattern ──────────────────────────────────────────────────────
def _step_pattern() -> None:
    st.markdown("## 2 · Describe the exam")

    with st.container(border=True):
        st.markdown("#### Subject")
        subject_col, type_col = st.columns([3, 2])
        with subject_col:
            st.text_input(
                "Subject or course name",
                key="subject",
                placeholder="e.g. Data Structures, Financial Accounting, Contract Law",
            )
        with type_col:
            st.selectbox(
                "Exam type", EXAM_TYPES, key="exam_type",
                help="Practical and project vivas also produce spoken viva questions.",
            )

        level_col, discipline_col = st.columns(2)
        with level_col:
            st.selectbox("Course level", COURSE_LEVELS, index=2, key="course_level")
        with discipline_col:
            st.selectbox("Discipline", DISCIPLINES, key="discipline")

    with st.container(border=True):
        st.markdown("#### Paper structure")
        marks_col, duration_col = st.columns(2)
        with marks_col:
            st.number_input("Total marks", min_value=10, max_value=500, value=100, step=5, key="total_marks")
        with duration_col:
            st.text_input("Duration", key="duration", value="3 hours")

        st.text_input(
            "Sections",
            key="sections",
            value="A: 4 x 5 marks, B: any 5 of 7 x 10 marks",
            placeholder="Sec B: any 5 of 7 x 5 marks",
            help="Write it however you like — the question count is inferred from this.",
        )
        st.multiselect(
            "Question types",
            ["MCQ", "short", "long", "numerical", "coding", "case study", "diagram", "viva"],
            default=["short", "long"],
            key="qtypes",
        )

        # Principle 17: hard % is computed, never asked.
        easy_col, medium_col, hard_col = st.columns(3)
        with easy_col:
            easy = st.slider("Easy %", 0, 100, 20, step=5, key="easy")
        with medium_col:
            medium = st.slider("Medium %", 0, 100, 50, step=5, key="medium")
        hard = ui.compute_hard_split(easy, medium)
        hard_col.metric("Hard % (auto)", f"{hard}%")

        warning = ui.difficulty_warning(easy, medium)  # Principle 20: inline validation
        if warning:
            st.warning(warning)

    with st.container(border=True):
        st.markdown("#### What to focus on")
        unit_raw = st.text_input(
            "Units to focus",
            key="unit_focus_raw",
            placeholder="blank = all units · accepts 1, 2 & 5 or unit 1-3",
        )
        parsed_units = ui.parse_unit_focus(unit_raw)
        st.caption(f"Understood: units {parsed_units}" if parsed_units else "All units will be used.")

    subject_value = st.session_state.get("subject", "").strip()
    _remember(
        subject=subject_value,
        exam_type=st.session_state.get("exam_type", "End-sem"),
        course_level=st.session_state.get("course_level", ""),
        discipline=st.session_state.get("discipline", ""),
        total_marks=int(st.session_state.get("total_marks", 100)),
        duration=st.session_state.get("duration", "3 hours"),
        sections=st.session_state.get("sections", ""),
        qtypes=list(st.session_state.get("qtypes", ["short", "long"])),
        easy=easy,
        medium=medium,
        unit_focus_raw=st.session_state.get("unit_focus_raw", ""),
    )
    _nav(
        2,
        can_continue=bool(subject_value) and not warning,
        hint="Add a subject name to continue." if not subject_value else "",
    )


# ── Step 3: Preferences ───────────────────────────────────────────────────────
def _step_prefs() -> None:
    st.markdown("## 3 · Choose what to build")

    with st.container(border=True):
        st.markdown("#### Output preset")
        st.radio(
            "Preset", list(PRESETS), key="preset", label_visibility="collapsed",
            captions=[
                "Ranked topics and a day-by-day plan. Fastest, cheapest run.",
                "Priority list, mock paper, model answers, plan, flashcards, question bank, viva.",
                "A practice paper with model answers and a question bank.",
            ],
        )
        outputs = list(PRESETS[st.session_state.preset])
        st.markdown(
            " ".join(ui.badge(item.replace("_", " ").upper(), "#ffffff") for item in sorted(set(outputs))),
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown("#### Study settings")
        date_col, hours_col, mode_col = st.columns(3)
        with date_col:
            exam_date = st.date_input(
                "Exam date", key="exam_date",
                value=datetime.date.today() + datetime.timedelta(days=14),
            )
        with hours_col:
            st.number_input("Study hours per day", min_value=1.0, max_value=16.0, value=4.0,
                            step=0.5, key="hours")
        with mode_col:
            st.selectbox(
                "Question mode", ["Creative", "Strict"], key="mode",
                help="Creative writes fresh questions. Strict reuses past and teacher questions.",
            )

        st.caption(f"{ui.days_left(exam_date)} day(s) until the exam · the plan covers every day in between")

    # Principle 2 + 11: rarely used inputs live in one collapsed expander.
    with st.expander("Advanced options"):
        st.markdown("##### Answer length guide")
        len1, len2, len3 = st.columns(3)
        with len1:
            st.text_input("2 marks", value="2-3 lines", key="al2")
        with len2:
            st.text_input("5 marks", value="~150 words", key="al5")
        with len3:
            st.text_input("10 marks", value="~350 words with diagram or example", key="al10")

        st.markdown("##### Topics")
        st.text_input("Topics to include", key="include_topics", placeholder="comma separated")
        st.text_input("Topics to exclude", key="exclude_topics", placeholder="comma separated")

        st.markdown("##### Practical / viva")
        viva1, viva2 = st.columns([1, 2])
        with viva1:
            st.number_input("Viva questions", min_value=5, max_value=40, value=15, step=5,
                            key="viva_count")
        with viva2:
            st.text_input(
                "Lab / practical focus", key="practical_focus",
                placeholder="e.g. titration experiments, DBMS lab, moot court",
            )

        st.markdown("##### Other")
        st.checkbox("Also export the priority list to Google Sheets", key="google_export")

    _remember(
        preset=st.session_state.get("preset", list(PRESETS)[0]),
        exam_date=exam_date,
        hours=float(st.session_state.get("hours", 4.0)),
        mode=st.session_state.get("mode", "Creative"),
        al2=st.session_state.get("al2", "2-3 lines"),
        al5=st.session_state.get("al5", "~150 words"),
        al10=st.session_state.get("al10", "~350 words with diagram or example"),
        include_topics=st.session_state.get("include_topics", ""),
        exclude_topics=st.session_state.get("exclude_topics", ""),
        viva_count=int(st.session_state.get("viva_count", 15)),
        practical_focus=st.session_state.get("practical_focus", ""),
        google_export=bool(st.session_state.get("google_export", False)),
    )

    _nav(3, can_continue=True)


# ── Step 4: Run & results ─────────────────────────────────────────────────────
def _collect_state() -> dict:
    """Build the workflow state from the wizard, absorbing messy input."""
    files = _answer("files") or []
    file_roles = _answer("file_roles") or {}
    raw_documents = [
        {
            "filename": file.name,
            "data": file.getvalue(),
            "role": file_roles.get(file.name, ui.detect_role(file.name)),
        }
        for file in files
    ]

    sections_text = _answer("sections", "")
    total_marks = int(_answer("total_marks", 100))
    inferred_questions = ui.infer_question_count(sections_text, total_marks, fallback=10)

    easy = int(_answer("easy", 20))
    medium = int(_answer("medium", 50))
    preset_outputs = list(PRESETS[_answer("preset", list(PRESETS)[0])])
    exam_type = _answer("exam_type", "End-sem")

    # Principle 17: a viva exam always gets viva questions, no extra tick needed.
    if "viva" in exam_type.lower() and "viva" not in preset_outputs:
        preset_outputs.append("viva")

    exam_date = _answer("exam_date") or (datetime.date.today() + datetime.timedelta(days=14))

    return {
        "subject": (_answer("subject") or "").strip(),
        "exam_type": exam_type,
        "course_level": _answer("course_level", ""),
        "discipline": _answer("discipline", ""),
        "exam_pattern": {
            "total_marks": total_marks,
            "duration": _answer("duration", "3 hours"),
            "num_questions": inferred_questions,
            "sections": ui.split_topics(sections_text or "A, B"),
            "internal_choice": sections_text,
            "question_types": _answer("qtypes") or ["short", "long"],
        },
        "difficulty_split": {"easy": easy, "medium": medium, "hard": ui.compute_hard_split(easy, medium)},
        "unit_focus": ui.parse_unit_focus(_answer("unit_focus_raw", "")),
        "include_topics": ui.split_topics(_answer("include_topics", "")),
        "exclude_topics": ui.split_topics(_answer("exclude_topics", "")),
        "exam_date": str(exam_date),
        "hours_per_day": float(_answer("hours", 4.0)),
        "mode": _answer("mode", "Creative"),
        "language": "hinglish",
        "answer_length": {
            "2": _answer("al2", "2-3 lines"),
            "5": _answer("al5", "~150 words"),
            "10": _answer("al10", "~350 words"),
        },
        "outputs_wanted": preset_outputs,
        "pasted_text": _answer("pasted_text", ""),
        "viva_count": int(_answer("viva_count", 15)),
        "practical_focus": _answer("practical_focus", ""),
        "raw_documents": raw_documents,
        "warnings": [],
        "errors": [],
        "progress": {},
    }


def _planned_nodes(outputs: list, exam_type: str, has_files: bool) -> list:
    nodes = ["classifier", "syllabus_analyst", "paper_analyst", "teacher_analyst",
             "knowledge_base", "predictor"]
    if "mock_paper" in outputs or "model_answers" in outputs:
        nodes += ["mock_paper", "examiner"]
    if "model_answers" in outputs:
        nodes.append("model_answers")
    if "question_bank" in outputs:
        nodes.append("question_bank")
    if "flashcards" in outputs or "formula_sheet" in outputs:
        nodes.append("flashcards")
    if "study_plan" in outputs:
        nodes.append("study_plan")
    if "viva" in outputs or "viva" in (exam_type or "").lower():
        nodes.append("viva")
    return (["extract"] if has_files else []) + nodes


def _execute(state: dict) -> None:
    """Run the pipeline with per-node progress (Principles 6, 15, 20)."""
    from graph import build_graph
    from nodes.file_processor import process_uploads

    planned = _planned_nodes(state["outputs_wanted"], state["exam_type"], bool(state["raw_documents"]))
    total = max(1, len(planned))
    done = 0
    final = dict(state)

    with st.status("Preparing your exam pack…", expanded=True) as status:
        progress = st.progress(0.0, text="Starting")
        stage_log = st.empty()
        lines = []

        def tick(label: str) -> None:
            nonlocal done
            done += 1
            lines.append(f"✓ {label}")
            progress.progress(min(1.0, done / total), text=f"Step {done} of {total}: {label}")
            stage_log.markdown("  \n".join(lines))

        if state["raw_documents"]:
            st.write("Reading your documents…")
            processed = process_uploads(state["raw_documents"])
            for document in processed:
                if document.get("error"):
                    st.warning(f"{document['filename']}: {document['error']}")
            final["raw_documents"] = processed
            tick("Documents read")

        graph = build_graph(state["outputs_wanted"], exam_type=state["exam_type"])
        for chunk in graph.stream(final, {"recursion_limit": 50}):
            for node_name, node_output in chunk.items():
                if isinstance(node_output, dict):
                    final.update(node_output)
                    for warning in node_output.get("warnings") or []:
                        st.warning(str(warning))
                tick(NODE_LABELS.get(node_name, node_name.replace("_", " ").title()))

        status.update(label="Exam pack ready", state="complete")

    st.session_state["result"] = final
    st.session_state["run_meta"] = {
        "subject": state["subject"],
        "exam_type": state["exam_type"],
        "exam_date": state["exam_date"],
        "days_left": ui.days_left(_answer("exam_date")),
        "google_export": bool(_answer("google_export", False)),
        "outputs": state["outputs_wanted"],
    }
    st.toast("Exam pack ready", icon="✅")
    st.rerun()


def _step_run() -> None:
    result = st.session_state.get("result")

    if result is None:
        st.markdown("## 4 · Run and see results")
        with st.container(border=True):
            st.markdown("#### Check before you run")
            # Principle 12: repeat earlier choices so nothing has to be remembered.
            st.markdown(
                f"**Subject** · {ui.escape(_answer('subject') or '—')}  \n"
                f"**Exam** · {_answer('exam_type', '—')} · "
                f"{_answer('exam_date', '—')}  \n"
                f"**Files** · {len(_answer('files') or [])} uploaded  \n"
                f"**Preset** · {_answer('preset', '—')}"
            )
            if not os.getenv("GEMINI_API_KEY"):
                st.error("Add a Gemini API key in the sidebar before running.")

        outputs = list(PRESETS[_answer("preset", list(PRESETS)[0])])
        st.caption(f"{len(outputs)} output(s) · one AI pass per stage · usually a few minutes")

        if st.button("▶ Run the workflow", type="primary", use_container_width=True,
                     disabled=not os.getenv("GEMINI_API_KEY")):
            state = _collect_state()
            if not state["subject"]:
                st.error("Add a subject name in step 2.")
            elif not state["raw_documents"] and not state["pasted_text"].strip():
                st.error("Add at least one document in step 1, or paste a teacher hint.")
            else:
                if state["subject"] not in st.session_state["saved_subjects"]:
                    st.session_state["saved_subjects"].append(state["subject"])
                _execute(state)
        return

    _render_results(result)


# ── Results ───────────────────────────────────────────────────────────────────
def _to_namespace(value):
    """Turn nested dicts into objects the HTML exporter can read."""
    if isinstance(value, dict):
        return types.SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


@st.cache_data(show_spinner=False, max_entries=3)
def _dashboard_html(payload_json: str) -> str:
    """Cache the expensive dashboard render (Principle 6)."""
    from exporters.html_exporter import render_dashboard

    return render_dashboard(_to_namespace(json.loads(payload_json)))


def _render_results(result: dict) -> None:
    predictions = result.get("predictions") or []
    must = [topic for topic in predictions if topic.label == "MUST STUDY"]
    days = st.session_state.run_meta.get("days_left")
    warnings = result.get("warnings") or []

    # Principle 14: the reveal is the peak — friendly headline, then the numbers.
    headline = (
        f"Your exam pack is ready: {len(must)} must-study topic{'s' if len(must) != 1 else ''} found"
        if must else "Your exam pack is ready"
    )
    st.markdown(f"## {headline}")
    st.caption("Every percentage is an evidence-based estimate (max 90%). Nothing is guaranteed.")

    ui.summary_strip([
        ("Must study", str(len(must))),
        ("Topics scored", str(len(predictions))),
        ("Days to exam", str(days) if days is not None else "—"),
        ("Warnings", str(len(warnings))),
    ])

    if warnings:
        with st.expander(f"⚠ {len(warnings)} warning(s) — read these first"):
            for warning in warnings:
                st.warning(str(warning))

    # Principle 16: priority list first, downloads last.
    tabs = st.tabs([
        "🎯 Priority list", "📝 Mock paper", "📅 Study plan", "🃏 Flashcards",
        "🎤 Viva", "✍️ Model answers", "🗂 Question bank", "⬇️ Downloads",
    ])

    with tabs[0]:
        _tab_priority(predictions)
    with tabs[1]:
        _tab_mock_paper(result, st.session_state.run_meta)
    with tabs[2]:
        _tab_plan(result)
    with tabs[3]:
        _tab_flashcards(result)
    with tabs[4]:
        _tab_viva(result)
    with tabs[5]:
        _tab_answers(result)
    with tabs[6]:
        _tab_bank(result)
    with tabs[7]:
        _tab_downloads(result, predictions)

    st.write("")
    if st.button("← Change something and run again", use_container_width=True):
        st.session_state.step = 2
        st.rerun()


def _tab_priority(predictions: list) -> None:
    if not predictions:
        ui.empty_state("No priority list in this run.")
        return

    for label in ("MUST STUDY", "SHOULD STUDY", "OPTIONAL"):
        group = [topic for topic in predictions if topic.label == label]
        if not group:
            continue
        st.markdown(
            f"### {ui.priority_badge(label)} <span class='ss-muted'>({len(group)})</span>",
            unsafe_allow_html=True,
        )

        if len(group) <= 6:
            # Principle 5: small groups render straight away.
            for topic in group:
                ui.topic_card(topic)
        else:
            # Larger groups chunk by unit to stay under Miller's limit.
            units: dict = {}
            for topic in group:
                units.setdefault(topic.unit, []).append(topic)
            for unit in sorted(units):
                with st.expander(f"Unit {unit} — {len(units[unit])} topic(s)"):
                    for topic in units[unit]:
                        ui.topic_card(topic)


def _tab_mock_paper(result: dict, meta: dict) -> None:
    paper = result.get("mock_paper")
    if paper is None:
        ui.empty_state("No mock paper in this run — pick the Practice or Full preset.")
        return

    header = paper.header or {}
    st.caption(f"{header.get('duration', '')} · {header.get('total_marks', '')} marks · {meta.get('exam_type', '')}")

    for section in paper.sections:
        st.markdown(f"#### {section.name}")
        if section.instructions:
            st.caption(section.instructions)
        for question in section.questions:
            with st.expander(f"Q{question.id} · {question.marks} marks · {question.difficulty}"):
                st.markdown(question.text)
                st.caption(
                    f"Unit {question.unit} · {question.topic} · {question.type} · source: {question.source_tag}"
                )

    exam = result.get("examiner_output")
    if exam:
        # Principle 11: examiner detail collapsed by default.
        with st.expander(f"Examiner verdict: {exam.get('verdict', 'checked')}"):
            issues = exam.get("issues") or []
            if not issues:
                st.success("All checks passed: marks total, syllabus mapping, duplicates, difficulty split.")
            for issue in issues:
                st.warning(f"{issue.get('check', '')}: {issue.get('problem', '')} → {issue.get('fix', '')}")


def _tab_plan(result: dict) -> None:
    plan = result.get("study_plan")
    if plan is None:
        ui.empty_state("No study plan in this run.")
        return

    rows = [
        {"Date": day.date, "Hours": day.hours, "Topic": block.topic,
         "Activity": block.activity, "Minutes": block.minutes}
        for day in plan.days for block in day.blocks
    ]
    if rows:
        import pandas as pd

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        ui.empty_state("The plan came back empty.")
    for warning in plan.warnings or []:
        st.warning(warning)


def _tab_flashcards(result: dict) -> None:
    cards = result.get("flashcards_output")
    if cards is None:
        ui.empty_state("No flashcards in this run.")
        return

    st.markdown("#### Flashcards")
    for card in ui.paginate(list(cards.flashcards), "cards_page", per_page=10):
        with st.expander(card.front):
            st.write(card.back)
            st.caption(card.topic)

    for unit in cards.formula_units:
        with st.expander(f"Unit {unit.unit} — quick revision sheet"):
            st.markdown(unit.content)


def _tab_viva(result: dict) -> None:
    viva = result.get("viva_output")
    if viva is None:
        ui.empty_state(
            "No viva set in this run. Choose a practical/viva exam type in step 2, "
            "or the Full pack preset."
        )
        return

    if viva.practical_tasks:
        st.markdown("#### Practical tasks to rehearse")
        for task in viva.practical_tasks:
            st.markdown(f"- {task}")

    for kind, heading in (("theory", "Theory"), ("application", "Application"),
                          ("practical", "Practical"), ("troubleshooting", "Troubleshooting")):
        group = [question for question in viva.questions if (question.kind or "theory") == kind]
        if not group:
            continue
        st.markdown(f"#### {heading} ({len(group)})")
        for question in ui.paginate(group, f"viva_{kind}_page", per_page=10):
            with st.expander(question.question):
                st.markdown("**What the examiner wants to hear**")
                for point in question.answer_points:
                    st.markdown(f"- {point}")
                if question.follow_ups:
                    st.markdown("**Likely follow-ups** · " + " · ".join(question.follow_ups))
                if question.topic:
                    st.caption(f"Unit {question.unit or '—'} · {question.topic}")


def _tab_answers(result: dict) -> None:
    answers = result.get("model_answers")
    if answers is None or not answers.answers:
        ui.empty_state("No model answers in this run.")
        return
    for answer in answers.answers:
        with st.expander(f"{answer.marks} marks · {answer.question_text[:70]}"):
            st.markdown(answer.question_text)
            st.markdown(answer.answer)
            if answer.marking_scheme:
                st.markdown("**Marking scheme**")
                for line in answer.marking_scheme:
                    st.markdown(f"- {line}")


def _tab_bank(result: dict) -> None:
    bank = result.get("question_bank")
    if bank is None or not bank.units:
        ui.empty_state("No question bank in this run — pick the Practice or Full preset.")
        return
    for unit in bank.units:
        with st.expander(f"Unit {unit.unit}"):
            for heading, items in (("Must", unit.must), ("Should", unit.should), ("Optional", unit.optional)):
                if not items:
                    continue
                st.markdown(f"**{heading}**")
                for question in ui.paginate(list(items), f"bank_{unit.unit}_{heading}", per_page=10):
                    st.markdown(f"- ({question.marks} marks) {question.text}")
                    st.caption(f"{question.type} · {question.difficulty} · {question.topic}")


def _tab_downloads(result: dict, predictions: list) -> None:
    st.markdown("#### Download your pack")
    meta = st.session_state.run_meta
    subject = meta.get("subject", "subject")
    safe = "".join(character for character in subject if character.isalnum() or character in " -_").strip() or "subject"

    # Streamlit renders every tab, so files are built on request instead of on
    # every rerun (Doherty threshold: instant reruns).
    if st.button("📦 Prepare downloads", use_container_width=True, key="prepare_downloads"):
        st.session_state["downloads"] = _build_downloads(result, predictions, subject, safe, meta)

    downloads = st.session_state.get("downloads") or {}
    if downloads:
        columns = st.columns(3)
        for index, (label, payload) in enumerate(downloads.items()):
            data, filename, mime = payload
            with columns[index % 3]:
                st.download_button(label, data=data, file_name=filename, mime=mime,
                                   use_container_width=True)
        if st.button("Rebuild files", key="rebuild_downloads", use_container_width=True):
            st.session_state.pop("downloads", None)
            st.rerun()
    else:
        st.caption("Builds PDFs and the HTML dashboard from this run. Takes a few seconds.")

    if meta.get("google_export") and predictions:
        if st.button("Export priority list to Google Sheets", use_container_width=True):
            try:
                from exporters.sheets_exporter import export_priority_to_sheet

                st.success(f"[Open the sheet]({export_priority_to_sheet(predictions, subject)})")
            except Exception as error:  # noqa: BLE001
                st.error(f"Google Sheets export failed: {error}")

    # Principle 14: calm ending plus a clear next action.
    st.success("That is everything. Start with the MUST STUDY topics, then take one mock paper.")
    st.caption("Next: open the study plan, mark today's block, and stick the priority list above your desk.")

    st.divider()
    _feedback()


def _build_downloads(result: dict, predictions: list, subject: str, safe: str, meta: dict) -> dict:
    """Build every export once and keep the bytes in session state."""
    from exporters import pdf_exporter

    downloads: dict = {}
    artifacts = [
        ("📄 Priority list (PDF)", "priority.pdf", "application/pdf",
         lambda: pdf_exporter.export_priority_list(predictions, subject) if predictions else None),
        ("📝 Mock paper (PDF)", "mock_paper.pdf", "application/pdf",
         lambda: pdf_exporter.export_mock_paper(result["mock_paper"], subject) if result.get("mock_paper") else None),
        ("📅 Study plan (PDF)", "study_plan.pdf", "application/pdf",
         lambda: pdf_exporter.export_study_plan(result["study_plan"]) if result.get("study_plan") else None),
        ("🃏 Flashcards (PDF)", "flashcards.pdf", "application/pdf",
         lambda: pdf_exporter.export_flashcards(result["flashcards_output"]) if result.get("flashcards_output") else None),
        ("🎤 Viva set (PDF)", "viva.pdf", "application/pdf",
         lambda: pdf_exporter.export_viva(result["viva_output"], subject) if result.get("viva_output") else None),
    ]
    for label, filename, mime, builder in artifacts:
        try:
            data = builder()
        except Exception as error:  # noqa: BLE001 - surfaced as a friendly message
            st.caption(f"{label} unavailable: {error}")
            continue
        if data:
            downloads[label] = (data, f"{safe}_{filename}", mime)

    payload = {
        "subject": subject,
        "exam_type": meta.get("exam_type", ""),
        "mode": result.get("mode", ""),
        "predictions": [
            {
                "topic": topic.topic, "unit": topic.unit, "percentage": topic.percentage,
                "label": topic.label, "evidence": topic.evidence,
                "likely_question_types": topic.likely_question_types,
                "confidence_note": getattr(topic, "confidence_note", ""),
                "factors": topic.factors.model_dump() if getattr(topic, "factors", None) else {},
            }
            for topic in predictions
        ],
        "study_plan": result["study_plan"].model_dump() if result.get("study_plan") else None,
        "mock_paper": result["mock_paper"].model_dump() if result.get("mock_paper") else None,
        "warnings": result.get("warnings", []),
    }
    try:
        html = _dashboard_html(json.dumps(payload, default=str))
        downloads["🌐 HTML dashboard"] = (html.encode("utf-8"), f"{safe}_dashboard.html", "text/html")
    except Exception as error:  # noqa: BLE001
        st.caption(f"Dashboard unavailable: {error}")

    return downloads


def _feedback() -> None:
    """Principle 20: let the user be heard, stored in session state."""
    st.markdown('<p class="ss-kicker">Was this useful?</p>', unsafe_allow_html=True)
    up, down, _ = st.columns([1, 1, 4])
    if up.button("👍 Useful", key="feedback_up", use_container_width=True):
        st.session_state.feedback = "up"
        st.toast("Thanks — noted for this session.")
    if down.button("👎 Not useful", key="feedback_down", use_container_width=True):
        st.session_state.feedback = "down"
        st.toast("Thanks — noted for this session.")
    if st.session_state.get("feedback"):
        st.caption(f"Feedback recorded: {st.session_state.feedback}. It stays in this session only.")
    st.text_input("Anything we should fix? (optional)", key="feedback_comment",
                  placeholder="e.g. the study plan packed too much into day 1")


# ── Router ────────────────────────────────────────────────────────────────────
ui.page_header(
    "StudSem",
    "Predict your paper from evidence, not luck — any subject, theory or viva. "
    "Percentages are evidence-based estimates, capped at 90%.",
)

step = st.session_state.step
ui.stepper(step, STEP_LABELS)  # Principles 4, 15

if step == 1:
    _step_upload()
elif step == 2:
    _step_pattern()
elif step == 3:
    _step_prefs()
else:
    _step_run()
