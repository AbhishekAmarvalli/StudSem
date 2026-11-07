"""
app.py – Streamlit entry point for Exam Intelligence Workflow.

Run locally:  streamlit run app.py
Deploy to:    Streamlit Community Cloud (primary) | Hugging Face Spaces (fallback)
"""

from __future__ import annotations

import io
import os
import sys
import logging

from pathlib import Path

# ── Path setup (Streamlit Cloud may need this) ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Load secrets / .env ───────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# Streamlit secrets override .env (must come after import)
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass
import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudSem — Exam Intelligence Workflow",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Neo-Brutalism CSS (injected after set_page_config) ───────────────────────
try:
    css_path = os.path.join(os.path.dirname(__file__), "static", "neo_brutalism.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Main Workflow", "✏️ Answer Grader"],
    index=0,
)

# ── API Key Setup (sidebar) ───────────────────────────────────────────────────
with st.sidebar.expander("🔑 API Keys", expanded=not bool(os.getenv("GEMINI_API_KEY"))):
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get free key at https://aistudio.google.com/apikey",
    )
    groq_key = st.text_input(
        "Groq API Key (for examiner)",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get free key at https://console.groq.com",
    )
    openrouter_key = st.text_input(
        "OpenRouter Key (fallback, optional)",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
    # Reflect key status
    if os.getenv("GEMINI_API_KEY"):
        st.success("✅ Gemini key set")
    else:
        st.warning("⚠ Gemini key missing – workflow will fail")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN WORKFLOW PAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Main Workflow":
    st.title("🎓 StudSem")
    st.caption(
        "Evidence-based exam preparation for B.Tech CSE. "
        "All percentages are likelihood *estimates*, capped at 90%. "
        "Nothing is guaranteed."
    )

    # ── Form ──────────────────────────────────────────────────────────────────
    with st.form("main_form"):
        col1, col2 = st.columns(2)

        with col1:
            subject = st.text_input("Subject name", placeholder="e.g. Data Structures")
            exam_type = st.selectbox(
                "Exam type",
                ["End-sem", "Mid-sem", "Unit test-MCQ", "Practical-Viva"],
            )
            exam_date = st.date_input(
                "Exam date",
                value=datetime.date.today() + datetime.timedelta(days=14),
            )
            hours_per_day = st.number_input(
                "Study hours per day", min_value=1.0, max_value=16.0, value=4.0, step=0.5
            )

        with col2:
            mode = st.selectbox("Mode", ["Creative", "Strict"], index=0)
            language = st.selectbox("Language", ["English + Hinglish", "English only"], index=0)
            st.markdown("**Answer length guide**")
            al2 = st.text_input("2 marks", value="2-3 lines")
            al5 = st.text_input("5 marks", value="~150 words")
            al10 = st.text_input("10 marks", value="~350 words with diagram/code/example")

        st.divider()
        st.subheader("📋 Exam Pattern")
        col3, col4, col5 = st.columns(3)
        with col3:
            total_marks = st.number_input("Total marks", min_value=1, value=100)
            duration = st.text_input("Duration", value="3 hours")
        with col4:
            num_questions = st.number_input("Number of questions", min_value=1, value=10)
            sections = st.text_input("Sections (e.g. A, B, C)", value="A, B")
        with col5:
            internal_choice = st.text_input("Internal choice (e.g. any 5 of 7)", value="any 5 of 7")
            q_types = st.multiselect(
                "Question types",
                ["MCQ", "short", "long", "numerical", "coding", "viva"],
                default=["short", "long"],
            )

        st.divider()
        st.subheader("🎚 Difficulty Split (%)")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            easy_pct = st.slider("Easy", 0, 100, 20)
        with dcol2:
            medium_pct = st.slider("Medium", 0, 100, 50)
        with dcol3:
            hard_pct = st.slider("Hard", 0, 100, 30)

        st.divider()
        st.subheader("🎯 Unit Focus")
        unit_focus_input = st.text_input(
            "Units to focus (comma-separated unit numbers, blank = all)",
            value="",
        )
        include_topics = st.text_input("Topics to include (comma-separated)", value="")
        exclude_topics = st.text_input("Topics to exclude (comma-separated)", value="")

        st.divider()
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload syllabus, past papers, teacher notes, PPTs, etc.",
            type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )

        file_roles = {}
        if uploaded_files:
            st.markdown("**Assign roles to each file:**")
            for f in uploaded_files:
                file_roles[f.name] = st.selectbox(
                    f"{f.name}",
                    ["Auto-detect", "syllabus", "past_paper", "teacher_priority", "notes", "pattern"],
                    key=f"role_{f.name}",
                )

        pasted_text = st.text_area(
            "Paste teacher hints / WhatsApp messages here",
            height=100,
            placeholder="e.g. 'Sir bola hai unit 3 and 5 se mostly questions aayenge…'",
        )

        st.divider()
        st.subheader("📤 Outputs Wanted")
        out_col1, out_col2 = st.columns(2)
        with out_col1:
            want_priority = st.checkbox("Priority list", value=True)
            want_study_plan = st.checkbox("Study plan", value=True)
            want_mock = st.checkbox("Mock paper", value=True)
            want_model_answers = st.checkbox("Model answers", value=True)
        with out_col2:
            want_qbank = st.checkbox("Question bank", value=False)
            want_flashcards = st.checkbox("Flashcards + formula sheet", value=True)

        # Google Sheets toggle
        google_export = st.checkbox("Also export to Google Sheets (requires service account)", value=False)

        submitted = st.form_submit_button("🚀 Run Workflow", use_container_width=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    if submitted:
        if not subject.strip():
            st.error("Please enter a subject name.")
            st.stop()

        outputs_wanted = []
        if want_priority:
            outputs_wanted.append("priority_list")
        if want_study_plan:
            outputs_wanted.append("study_plan")
        if want_mock:
            outputs_wanted.append("mock_paper")
        if want_model_answers:
            outputs_wanted.append("model_answers")
        if want_qbank:
            outputs_wanted.append("question_bank")
        if want_flashcards:
            outputs_wanted += ["flashcards", "formula_sheet"]

        # Build raw_documents list
        raw_documents = []
        if uploaded_files:
            for f in uploaded_files:
                role_choice = file_roles.get(f.name, "Auto-detect")
                role = "auto" if role_choice == "Auto-detect" else role_choice
                raw_documents.append({
                    "filename": f.name,
                    "data": f.read(),
                    "role": role,
                })

        unit_focus = [int(u.strip()) for u in unit_focus_input.split(",") if u.strip().isdigit()]

        exam_pattern = {
            "total_marks": total_marks,
            "duration": duration,
            "num_questions": num_questions,
            "sections": [s.strip() for s in sections.split(",")],
            "internal_choice": internal_choice,
            "question_types": q_types,
        }

        state = {
            "subject": subject.strip(),
            "exam_type": exam_type,
            "exam_pattern": exam_pattern,
            "difficulty_split": {"easy": easy_pct, "medium": medium_pct, "hard": hard_pct},
            "unit_focus": unit_focus,
            "include_topics": [t.strip() for t in include_topics.split(",") if t.strip()],
            "exclude_topics": [t.strip() for t in exclude_topics.split(",") if t.strip()],
            "exam_date": str(exam_date),
            "hours_per_day": hours_per_day,
            "mode": mode.split()[0],
            "language": "hinglish" if "Hinglish" in language else "english",
            "answer_length": {"2": al2, "5": al5, "10": al10},
            "outputs_wanted": outputs_wanted,
            "pasted_text": pasted_text,
            "raw_documents": raw_documents,
            "warnings": [],
            "errors": [],
            "progress": {},
        }

        # --- Pre-process files outside graph ---
        if raw_documents:
            with st.spinner("📄 Extracting text from files…"):
                from nodes.file_processor import process_uploads
                processed = process_uploads(raw_documents)
                state["raw_documents"] = processed

        # --- Run graph with progress bar ---
        progress_bar = st.progress(0, text="Starting workflow…")
        status_placeholder = st.empty()

        try:
            from graph import run_workflow

            NODE_STEPS = [
                "classifier", "syllabus_analyst", "paper_analyst", "teacher_analyst",
                "knowledge_base", "predictor", "mock_paper", "examiner",
                "model_answers", "question_bank", "flashcards", "study_plan",
            ]
            total_steps = len([n for n in NODE_STEPS if n in outputs_wanted or n in [
                "classifier", "syllabus_analyst", "paper_analyst", "teacher_analyst",
                "knowledge_base", "predictor",
            ]])

            # Monkey-patch progress updates (simple polling via state)
            with st.spinner("🧠 Running AI pipeline…"):
                final_state = run_workflow(state)

            progress_bar.progress(100, text="✅ Done!")
            st.success("Workflow complete!")

        except Exception as e:
            st.error(f"❌ Workflow failed: {e}")
            logger.exception("Workflow error")
            st.stop()

        # ── Show warnings ─────────────────────────────────────────────────────
        if final_state.get("warnings"):
            with st.expander("⚠ Warnings", expanded=True):
                for w in final_state["warnings"]:
                    st.warning(w)

        # ── Priority List ─────────────────────────────────────────────────────
        predictions = final_state.get("predictions") or final_state.get("priority_list") or []
        if predictions and want_priority:
            st.subheader("🎯 Priority Topics")
            from scoring import factor_breakdown
            for label_filter in ("MUST STUDY", "SHOULD STUDY", "OPTIONAL"):
                group = [t for t in predictions if t.label == label_filter]
                if not group:
                    continue
                color = {"MUST STUDY": "🔴", "SHOULD STUDY": "🟡", "OPTIONAL": "⚪"}[label_filter]
                st.markdown(f"### {color} {label_filter}")
                for t in group:
                    with st.expander(f"{t.topic} (Unit {t.unit}) — {t.percentage}%"):
                        st.progress(int(t.percentage or 0), text=f"{t.percentage}% likelihood estimate")
                        st.write(f"**Evidence:** {t.evidence}")
                        st.write(f"**Types:** {', '.join(t.likely_question_types)}")
                        fb = factor_breakdown(t.factors)
                        st.json(fb)

        # ── Study Plan ────────────────────────────────────────────────────────
        study_plan = final_state.get("study_plan")
        if study_plan and want_study_plan:
            st.subheader("📅 Study Plan")
            import pandas as pd
            rows = []
            for day in study_plan.days:
                for block in day.blocks:
                    rows.append({
                        "Date": day.date,
                        "Topic": block.topic,
                        "Activity": block.activity,
                        "Minutes": block.minutes,
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # ── Mock Paper ────────────────────────────────────────────────────────
        mock_paper = final_state.get("mock_paper")
        if mock_paper and want_mock:
            st.subheader("📝 Mock Paper")
            for section in mock_paper.sections:
                st.markdown(f"**{section.name}** – *{section.instructions}*")
                for q in section.questions:
                    st.markdown(
                        f"> **Q{q.id}.** [{q.marks} marks | {q.type} | {q.difficulty}] \n"
                        f"> {q.text} \n"
                        f"> *Source: {q.source_tag} | Topic: {q.topic}*"
                    )

        # ── Flashcards ────────────────────────────────────────────────────────
        fc_output = final_state.get("flashcards_output")
        if fc_output and want_flashcards:
            st.subheader("🃏 Flashcards")
            if fc_output.flashcards:
                import pandas as pd
                df = pd.DataFrame([f.dict() for f in fc_output.flashcards])
                st.dataframe(df, use_container_width=True)
            for fu in fc_output.formula_units:
                with st.expander(f"Unit {fu.unit} – Quick Revision Sheet"):
                    st.markdown(fu.content)

        # ── Downloads ─────────────────────────────────────────────────────────
        st.subheader("⬇️ Downloads")
        dcols = st.columns(4)

        if predictions and want_priority:
            with dcols[0]:
                from exporters.pdf_exporter import export_priority_list
                try:
                    pdf_bytes = export_priority_list(predictions, subject)
                    st.download_button(
                        "📄 Priority List PDF",
                        data=pdf_bytes,
                        file_name=f"{subject}_priority.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.warning(f"PDF export failed: {e}")

        if mock_paper and want_mock:
            with dcols[1]:
                from exporters.pdf_exporter import export_mock_paper
                try:
                    pdf_bytes = export_mock_paper(mock_paper, subject)
                    st.download_button(
                        "📝 Mock Paper PDF",
                        data=pdf_bytes,
                        file_name=f"{subject}_mock_paper.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.warning(f"PDF export failed: {e}")

        if study_plan and want_study_plan:
            with dcols[2]:
                from exporters.pdf_exporter import export_study_plan
                try:
                    pdf_bytes = export_study_plan(study_plan)
                    st.download_button(
                        "📅 Study Plan PDF",
                        data=pdf_bytes,
                        file_name=f"{subject}_study_plan.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.warning(f"PDF export failed: {e}")

        # HTML dashboard
        with dcols[3]:
            from exporters.html_exporter import render_dashboard
            try:
                html_content = render_dashboard({
                    "subject": subject,
                    "exam_type": exam_type,
                    "mode": mode,
                    "predictions": predictions,
                    "study_plan": study_plan,
                    "mock_paper": mock_paper,
                    "warnings": final_state.get("warnings", []),
                })
                st.download_button(
                    "🌐 HTML Dashboard",
                    data=html_content.encode("utf-8"),
                    file_name=f"{subject}_dashboard.html",
                    mime="text/html",
                )
            except Exception as e:
                st.warning(f"HTML export failed: {e}")

        # Google Sheets
        if google_export and predictions:
            with st.spinner("Exporting to Google Sheets…"):
                try:
                    from exporters.sheets_exporter import export_priority_to_sheet
                    url = export_priority_to_sheet(predictions, subject)
                    st.success(f"[Open Google Sheet]({url})")
                except Exception as e:
                    st.error(f"Google Sheets export failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ANSWER GRADER PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "✏️ Answer Grader":
    st.title("✏️ Answer Grader")
    st.caption("Submit your answer and get marks, rubric, and feedback.")

    with st.form("grader_form"):
        question_text = st.text_area("Question", height=100)
        marks = st.number_input("Marks for this question", min_value=1, value=5)
        student_answer = st.text_area("Your answer", height=200)
        model_answer_input = st.text_area("Model answer (optional)", height=100)
        syllabus_context = st.text_area("Syllabus context (optional)", height=80)
        grade_btn = st.form_submit_button("📊 Grade My Answer")

    if grade_btn:
        if not question_text.strip() or not student_answer.strip():
            st.error("Please provide both the question and your answer.")
        else:
            with st.spinner("Grading…"):
                try:
                    from nodes.answer_grader import grade_answer
                    result = grade_answer(
                        question_text=question_text,
                        student_answer=student_answer,
                        marks=int(marks),
                        model_answer=model_answer_input,
                        syllabus_context=syllabus_context,
                    )
                    st.success(f"**{result.marks_awarded} / {result.out_of}** marks awarded")

                    st.subheader("📋 Rubric")
                    import pandas as pd
                    rubric_df = pd.DataFrame([r.dict() for r in result.rubric])
                    st.dataframe(rubric_df, use_container_width=True)

                    if result.missing_points:
                        st.subheader("❌ Missing Points")
                        for mp in result.missing_points:
                            st.markdown(f"- {mp}")

                    if result.errors:
                        st.subheader("⚠ Errors in Your Answer")
                        for err in result.errors:
                            st.markdown(f"- {err}")

                    if result.improved_answer:
                        with st.expander("💡 Improved Answer"):
                            st.markdown(result.improved_answer)

                    if result.tips:
                        st.subheader("🔑 Tips")
                        for tip in result.tips:
                            st.markdown(f"- {tip}")

                except Exception as e:
                    st.error(f"Grading failed: {e}")
                    logger.exception("Grader error")