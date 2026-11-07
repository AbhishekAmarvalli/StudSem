"""
pages/1_Answer_Grader.py – grade one written answer.

Split out of app.py so the main flow keeps its four steps (Hick's law).
Shares ui.py and st.session_state with the main page, so subjects you already
ran appear in the dropdown (Principle 12: recognition over recall).
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ui  # noqa: E402

st.set_page_config(
    page_title="Answer Grader — StudSem",
    page_icon="✏️",
    layout="wide",
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

ui.inject_css()

ui.page_header(
    "Answer Grader",
    "Paste a question and your answer. You get marks, a rubric, and exactly what was missing.",
)

if not os.getenv("GEMINI_API_KEY"):
    st.warning("Add a Gemini API key in the sidebar of the main app before grading.")

# Principle 12: reuse what the user already typed instead of asking again.
saved_subjects = st.session_state.get("saved_subjects", [])
subject_options = ["Not specified"] + saved_subjects

with st.container(border=True):
    st.markdown("#### The question")
    top_left, top_right = st.columns([3, 1])
    with top_left:
        subject = st.selectbox(
            "Subject", subject_options, index=0,
            help="Subjects you have run in this session appear here.",
        )
    with top_right:
        marks = st.number_input("Marks", min_value=1, max_value=100, value=5, step=1)

    question_text = st.text_area(
        "Question", height=100,
        placeholder="The question exactly as it appeared in the paper.",
    )

with st.container(border=True):
    st.markdown("#### Your answer")
    student_answer = st.text_area(
        "Answer", height=220, label_visibility="collapsed",
        placeholder="Type or paste what you wrote in the exam.",
    )

# Principle 11: optional context is available but not in the way.
with st.expander("Optional: model answer and syllabus context"):
    model_answer_input = st.text_area(
        "Model answer", height=100,
        placeholder="If you have one, grading gets sharper.",
    )
    syllabus_context = st.text_area(
        "Syllabus context", height=80,
        placeholder="e.g. Unit 3 — Normalisation",
    )

left, right = st.columns([2, 1])
grade_clicked = left.button(
    "📊 Grade my answer", type="primary", use_container_width=True,
    disabled=not os.getenv("GEMINI_API_KEY"),
)
right.caption("Graded like a university examiner: credit for correct points, penalties for errors, no reward for length.")

if grade_clicked:
    # Principle 20: inline validation, never a silent failure.
    if not question_text.strip() or not student_answer.strip():
        st.error("Add both the question and your answer.")
    else:
        with st.spinner("Reading your answer like an examiner…"):
            try:
                from nodes.answer_grader import grade_answer

                result = grade_answer(
                    question_text=question_text,
                    student_answer=student_answer,
                    marks=int(marks),
                    model_answer=model_answer_input,
                    syllabus_context=syllabus_context,
                )
            except Exception as error:  # noqa: BLE001 - the user sees a next step, not a traceback
                st.error(f"Grading stopped: {error}")
                st.caption("Check your API key in the sidebar, then try again.")
                result = None

        if result is not None:
            st.session_state["grader_result"] = result

# Principle 6: results live in session_state, so tab switches and reruns keep them.
result = st.session_state.get("grader_result")

if result is not None:
    st.markdown("## Result")
    st.markdown(
        f'<div class="marks-big" style="border:3px solid #0a0a0a;background:#c6ff4d;'
        f'padding:.7rem 1rem;font-size:2.2rem;box-shadow:6px 6px 0 #0a0a0a">'
        f"{result.marks_awarded} / {result.out_of} marks</div>",
        unsafe_allow_html=True,
    )

    ui.summary_strip([
        ("Rubric points", str(len(result.rubric or []))),
        ("Missing points", str(len(result.missing_points or []))),
        ("Errors", str(len(result.errors or []))),
    ])

    if result.rubric:
        import pandas as pd

        st.markdown("#### Rubric")
        st.dataframe(
            pd.DataFrame([point.model_dump() for point in result.rubric]),
            use_container_width=True,
            hide_index=True,
        )

    if result.missing_points:
        with st.expander(f"❌ {len(result.missing_points)} missing point(s)", expanded=True):
            for point in result.missing_points:
                st.markdown(f"- {point}")

    if result.errors:
        with st.expander(f"⚠ {len(result.errors)} error(s) in your answer", expanded=True):
            for error in result.errors:
                st.markdown(f"- {error}")

    if result.improved_answer:
        # Principle 11: long content stays collapsed until asked for.
        with st.expander("💡 Improved answer"):
            st.markdown(result.improved_answer)

    if result.tips:
        with st.expander("🔑 Tips for next time"):
            for tip in result.tips:
                st.markdown(f"- {tip}")

    # Principle 20: a small feedback loop.
    st.markdown('<p class="ss-kicker">Was this grading fair?</p>', unsafe_allow_html=True)
    yes, no, _ = st.columns([1, 1, 4])
    if yes.button("👍 Fair", key="grader_fair", use_container_width=True):
        st.toast("Thanks — noted for this session.")
        st.session_state["grader_feedback"] = "fair"
    if no.button("👎 Too harsh or too generous", key="grader_unfair", use_container_width=True):
        st.toast("Thanks — noted for this session.")
        st.session_state["grader_feedback"] = "unfair"
