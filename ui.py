"""
ui.py – presentation layer for StudSem's Streamlit app.

Everything visual lives here so app.py and pages/*.py stay readable:
CSS injection, stepper, badges, cards, summary strip, validation helpers.

Design language: neo-brutalism — cream paper, 3px ink borders, hard offset
shadows, one accent per screen. One Google font (Archivo Black) for display
type; body text uses the system stack to keep the download small (Occam).
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import re
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    "ink": "#0a0a0a",
    "paper": "#fdf8ec",
    "card": "#ffffff",
    "muted": "#4a4a4a",
    "yellow": "#ffe600",
    "pink": "#ff5da2",
    "cyan": "#5ee7ff",
    "lime": "#c6ff4d",
}

# Principle 8 (similarity): one colour per priority label, everywhere.
# Text + icon carry the meaning too, so colour is never the only signal.
LABEL_STYLES: Dict[str, Tuple[str, str, str]] = {
    "MUST STUDY": ("#e63946", "#ffffff", "▲ MUST STUDY"),
    "SHOULD STUDY": ("#ffb703", "#0a0a0a", "● SHOULD STUDY"),
    "OPTIONAL": ("#e6e1d6", "#0a0a0a", "○ OPTIONAL"),
}

CSS = """
<style>
:root {
  --ink: #0a0a0a;
  --paper: #fdf8ec;
  --card: #ffffff;
  --muted: #4a4a4a;
  --line: 3px solid var(--ink);
  --shadow: 6px 6px 0 var(--ink);
  --yellow: #ffe600;
  --pink: #ff5da2;
  --cyan: #5ee7ff;
  --lime: #c6ff4d;
  --radius: 2px;
}
/* Principle 1 + accessibility: a dark-mode variant keeps 4.5:1 contrast. */
html[data-theme="dark"] {
  --ink: #f4efe2;
  --paper: #121212;
  --card: #1b1b1b;
  --muted: #c8c2b4;
  --shadow: 6px 6px 0 #f4efe2;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--paper) !important;
  background-image:
    linear-gradient(to right, rgba(128, 128, 128, .12) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(128, 128, 128, .12) 1px, transparent 1px);
  background-size: 32px 32px;
  color: var(--ink) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

.block-container {
  padding: 1.6rem 1.4rem 4rem !important;
  max-width: 1120px;
}

/* Principle 10: one H1, quieter H2, muted helper text. */
h1, h2, h3, [data-testid="stHeading"] h1, [data-testid="stHeading"] h2 {
  font-family: "Archivo Black", "Arial Black", sans-serif !important;
  font-weight: 400 !important;
  letter-spacing: -.02em !important;
  color: var(--ink) !important;
  line-height: 1.05 !important;
}
h1 { font-size: clamp(1.7rem, 4vw, 2.4rem) !important; }
h2 { font-size: 1.25rem !important; margin-top: 1.2rem !important; }
h3 { font-size: 1rem !important; margin-top: .9rem !important; }

p, li, label, span, div { color: inherit; }

/* Principle 10: muted helper text (.ss-muted) is smaller and quieter. */
.ss-muted { color: var(--muted) !important; font-size: .82rem; line-height: 1.45; }
.ss-kicker {
  font-size: .68rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); font-weight: 700;
}

/* Principle 4: familiar stepper at the top, with progress (Principle 15). */
.ss-stepper { display: flex; gap: .5rem; flex-wrap: wrap; margin: .2rem 0 .6rem; }
.ss-step {
  border: var(--line); background: var(--card); padding: .35rem .6rem;
  font-size: .72rem; font-weight: 700; letter-spacing: .04em;
  box-shadow: 3px 3px 0 var(--ink); color: var(--ink) !important;
}
.ss-step[data-state="done"] { background: var(--lime); }
.ss-step[data-state="current"] { background: var(--yellow); }
.ss-step[data-state="todo"] { background: var(--card); color: var(--muted) !important; }

/* Principle 9: related controls share one bordered card. */
[data-testid="stVerticalBlockBorderWrapper"] {
  border: var(--line) !important;
  border-radius: var(--radius) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow) !important;
  padding: .9rem 1rem !important;
}

/* Principle 7: labels sit tight to their inputs. */
[data-testid="stWidgetLabel"] p {
  font-size: .72rem !important; font-weight: 700 !important;
  letter-spacing: .06em !important; text-transform: uppercase !important;
  color: var(--ink) !important;
}
[data-testid="stWidgetLabel"] { margin-bottom: .15rem !important; }

input[type="text"], input[type="number"], input[type="password"], textarea,
[data-baseweb="select"] > div, [data-testid="stDateInput"] input {
  background: var(--paper) !important;
  color: var(--ink) !important;
  border: 2px solid var(--ink) !important;
  border-radius: var(--radius) !important;
  font-size: .92rem !important;
}
input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within {
  outline: 3px solid var(--pink) !important; outline-offset: 1px;
}
[data-baseweb="popover"] [role="listbox"] {
  background: var(--card) !important; border: var(--line) !important;
}
[role="option"]:hover { background: var(--yellow) !important; }

/* Buttons: one shape for all, primary is the only filled accent. */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border: var(--line) !important;
  border-radius: var(--radius) !important;
  background: var(--card) !important;
  color: var(--ink) !important;
  box-shadow: 4px 4px 0 var(--ink) !important;
  font-weight: 700 !important;
  font-size: .82rem !important;
  letter-spacing: .06em !important;
  text-transform: uppercase !important;
  padding: .55rem .9rem !important;
  transition: transform .1s ease, box-shadow .1s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  background: var(--yellow) !important;
  transform: translate(-1px, -1px);
  box-shadow: 6px 6px 0 var(--ink) !important;
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: translate(2px, 2px); box-shadow: 1px 1px 0 var(--ink) !important;
}
/* Principle 13: exactly one filled accent per screen — the primary action. */
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: var(--yellow) !important;
  font-size: .9rem !important;
  padding: .75rem 1rem !important;
}
.stButton > button:disabled, .stDownloadButton > button:disabled {
  opacity: .5 !important; box-shadow: 2px 2px 0 var(--ink) !important;
}

/* Principle 4: tabs behave like familiar tabs. */
[data-testid="stTabs"] [role="tablist"] { gap: .4rem; border-bottom: none; }
[data-testid="stTabs"] [role="tab"] {
  border: 2px solid var(--ink) !important; background: var(--card) !important;
  padding: .35rem .7rem !important; border-radius: var(--radius) !important;
  font-size: .75rem !important; font-weight: 700 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: var(--ink) !important; color: var(--paper) !important;
}

/* Principle 10: summary strip reads as one row of stats. */
[data-testid="stMetric"] {
  border: var(--line); background: var(--card); box-shadow: 4px 4px 0 var(--ink);
  padding: .6rem .8rem;
}
[data-testid="stMetricValue"] { font-size: 1.6rem !important; color: var(--ink) !important; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .7rem !important; }

/* Principle 5 + 11: expanders keep long lists manageable. */
[data-testid="stExpander"] {
  border: 2px solid var(--ink) !important; border-radius: var(--radius) !important;
  background: var(--card) !important;
}
[data-testid="stExpander"] summary { font-weight: 700 !important; font-size: .88rem !important; }
[data-testid="stExpander"] summary:hover { background: rgba(255, 230, 0, .35); }

/* Principle 13 + 20: callouts are distinct from content. */
[data-testid="stAlert"] {
  border: var(--line) !important; border-radius: var(--radius) !important;
  box-shadow: 4px 4px 0 var(--ink); color: var(--ink) !important;
}
[data-testid="stAlert"] p { color: var(--ink) !important; }
div[data-testid="stAlertContentWarning"] { background: #fff3bf !important; }
div[data-testid="stAlertContentError"] { background: #ffd6dd !important; }
div[data-testid="stAlertContentSuccess"] { background: #e4f8c4 !important; }
div[data-testid="stAlertContentInfo"] { background: #d9f6ff !important; }

/* Principle 15: progress is visible and chunky. */
[data-testid="stProgress"] > div { background: var(--card) !important; border: 2px solid var(--ink); }
[data-testid="stProgress"] > div > div > div > div { background: var(--ink) !important; }

/* Principle 1: file uploader gets the same card treatment. */
[data-testid="stFileUploaderDropzone"] {
  background: var(--paper) !important; border: 3px dashed var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] button {
  border: 2px solid var(--ink) !important; background: var(--cyan) !important;
  color: #0a0a0a !important; border-radius: var(--radius) !important;
}

/* Dataframes stay legible (Principle 4: tables for data). */
[data-testid="stDataFrame"] { border: 2px solid var(--ink) !important; }

/* Principle 8: identical badge chips for the same concept everywhere. */
.ss-badge {
  display: inline-block; border: 2px solid var(--ink); padding: .05rem .4rem;
  font-size: .64rem; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; white-space: nowrap;
}

/* Principle 9: one card per topic, all of its facts inside. */
.ss-topic { border: 2px solid var(--ink); background: var(--card); padding: .7rem .8rem; margin: .4rem 0; }
.ss-topic-head { display: flex; justify-content: space-between; gap: .8rem; flex-wrap: wrap; align-items: baseline; }
.ss-topic-title { font-weight: 700; font-size: .95rem; }
.ss-bar { height: 12px; border: 2px solid var(--ink); background: var(--paper); margin: .45rem 0; }
.ss-bar > span { display: block; height: 100%; background: var(--ink); }
.ss-empty { border: 3px dashed var(--ink); padding: .9rem; text-align: center; color: var(--muted); font-size: .85rem; }

/* Sidebar (Principle 16: API status first, help last). */
[data-testid="stSidebar"] { background: var(--card) !important; border-right: var(--line); }
[data-testid="stSidebar"] h2 { font-size: .95rem !important; border-bottom: none; }

/* Accessibility: never remove focus rings. */
:focus-visible { outline: 3px solid var(--pink) !important; outline-offset: 2px; }

@media (max-width: 640px) {
  .block-container { padding: 1rem .8rem 3rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
"""


# ── Rendering helpers ─────────────────────────────────────────────────────────

def inject_css() -> None:
    """Load the single source of truth for styling (Principle 1)."""
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """One H1 plus one muted line (Principle 10)."""
    st.markdown(f"# {title}")
    st.markdown(f'<p class="ss-muted">{subtitle}</p>', unsafe_allow_html=True)


def stepper(current: int, labels: Sequence[str]) -> None:
    """
    Familiar top stepper with completed ticks (Principles 4, 15).
    current is 1-based.
    """
    chips = []
    for index, label in enumerate(labels, start=1):
        state = "done" if index < current else ("current" if index == current else "todo")
        tick = "✓ " if index < current else ""
        chips.append(f'<span class="ss-step" data-state="{state}">{tick}{index}. {label}</span>')
    st.markdown(f'<div class="ss-stepper">{"".join(chips)}</div>', unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, (current - 1) / len(labels))), text=f"Step {current} of {len(labels)}")


def badge(text: str, background: str = "#ffffff", color: str = "#0a0a0a") -> str:
    """Inline chip. Always carries text, never colour alone (accessibility)."""
    return (
        f'<span class="ss-badge" style="background:{background};color:{color}">'
        f"{text}</span>"
    )


def priority_badge(label: str) -> str:
    """Same colour + text for a priority label on screen, PDF and dashboard (Principle 8)."""
    background, color, text = LABEL_STYLES.get(label or "OPTIONAL", LABEL_STYLES["OPTIONAL"])
    return badge(text, background, color)


def topic_card(topic, show_factors: bool = True) -> None:
    """
    One bordered card holding label, percentage, evidence and factors
    (Principles 9, 11, 13).
    """
    percentage = float(getattr(topic, "percentage", 0) or 0)
    evidence = getattr(topic, "evidence", "") or "No evidence returned."
    types = ", ".join(getattr(topic, "likely_question_types", []) or []) or "not classified"

    st.markdown(
        f"""
        <div class="ss-topic">
          <div class="ss-topic-head">
            <span class="ss-topic-title">{topic.topic}</span>
            {priority_badge(topic.label)}
          </div>
          <div class="ss-muted">Unit {topic.unit} · <strong>{percentage:.0f}%</strong> evidence-based estimate (max 90%)</div>
          <div class="ss-bar"><span style="width:{max(2, min(100, percentage)):.0f}%"></span></div>
          <div class="ss-muted">{evidence}</div>
          <div class="ss-muted">Usually asked as: {types}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_factors and getattr(topic, "factors", None) is not None:
        from scoring import factor_breakdown

        with st.expander("How this score was built"):
            st.json(factor_breakdown(topic.factors))


def summary_strip(items: Sequence[Tuple[str, str]]) -> None:
    """Results lead with the headline numbers (Principles 10, 14)."""
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def empty_state(message: str) -> None:
    st.markdown(f'<div class="ss-empty">{message}</div>', unsafe_allow_html=True)


def escape(text: str) -> str:
    """Escape user-supplied text before dropping it into inline HTML."""
    return _html.escape(str(text or ""))


def checklist(title: str, checks: Dict[str, bool]) -> None:
    """
    Upload step shows what is still missing so the goal feels close
    (Principle 15). Ticks are text, not colour-only.
    """
    chips = " ".join(
        badge(f"{'✓' if done else '✗'} {label}", "#e4f8c4" if done else "#ffffff")
        for label, done in checks.items()
    )
    st.markdown(
        f'<div class="ss-muted">{escape(title)}: {chips}</div>',
        unsafe_allow_html=True,
    )


def paginate(items: List, key: str, per_page: int = 10) -> List:
    """
    Keep long lists under Miller's limit (Principle 5): 10 items per page.
    """
    if len(items) <= per_page:
        return items
    pages = (len(items) + per_page - 1) // per_page
    current = st.session_state.get(key, 1)
    current = min(max(1, current), pages)
    start = (current - 1) * per_page

    left, middle, right = st.columns([1, 2, 1])
    if left.button("‹ Previous", key=f"{key}_prev", disabled=current == 1, use_container_width=True):
        st.session_state[key] = current - 1
        st.rerun()
    middle.markdown(
        f'<div class="ss-muted" style="text-align:center;padding-top:.4rem">'
        f"Page {current} of {pages}</div>",
        unsafe_allow_html=True,
    )
    if right.button("Next ›", key=f"{key}_next", disabled=current == pages, use_container_width=True):
        st.session_state[key] = current + 1
        st.rerun()
    return items[start:start + per_page]


# ── Input parsing helpers (Principle 18: be liberal in what you accept) ───────

def parse_unit_focus(raw: str) -> List[int]:
    """
    Accepts "1, 2 & 5", "unit 1-3", "1 2 5", "Units 2,4".
    Returns a sorted, de-duplicated list of unit numbers.
    """
    if not raw or not raw.strip():
        return []
    text = raw.lower().replace("units", " ").replace("unit", " ")
    text = text.replace("&", ",").replace(";", ",").replace("and", ",")
    units: List[int] = []

    for chunk in re.split(r"[,\s]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        range_match = re.fullmatch(r"(\d+)\s*[-–to]+\s*(\d+)", chunk)
        if range_match:
            low, high = int(range_match.group(1)), int(range_match.group(2))
            units.extend(range(min(low, high), max(low, high) + 1))
        elif chunk.isdigit():
            units.append(int(chunk))
    return sorted({unit for unit in units if 0 < unit <= 20})


def split_topics(raw: str) -> List[str]:
    """Trim, drop blanks and duplicates while keeping order."""
    seen, result = set(), []
    for item in re.split(r"[,;\n]", raw or ""):
        cleaned = item.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            result.append(cleaned)
    return result


def infer_question_count(pattern_text: str, total_marks: int, fallback: int = 10) -> int:
    """
    Absorb arithmetic the user should not have to do (Principle 17).
    Reads "A: 4 x 5 marks, B: any 5 of 7 x 10 marks" as 9 questions.
    """
    def clamp(value: int) -> int:
        return max(1, min(60, value))

    text = (pattern_text or "").lower()
    if not text.strip():
        return fallback

    direct = re.search(r"(\d+)\s*(?:questions|question|ques\b|q\b)", text)
    if direct:
        return clamp(int(direct.group(1)))

    counts: List[int] = []

    # "any 5 of 7 x 10 marks" → the student answers 5 questions in that section.
    def _any_of(match: "re.Match") -> str:
        counts.append(int(match.group(1)))
        return " "

    text = re.sub(r"any\s+(\d+)\s+of\s+\d+(?:\s*x\s*\d+\s*marks)?", _any_of, text)

    # Remaining "4 x 5 marks" sections → 4 questions each.
    for match in re.finditer(r"(\d+)\s*x\s*(\d+)\s*marks", text):
        counts.append(int(match.group(1)))

    if counts:
        return clamp(sum(counts))
    if total_marks:
        return clamp(total_marks // 10)
    return fallback


def compute_hard_split(easy: int, medium: int) -> int:
    """Hard is derived, never asked (Principle 17)."""
    return max(0, 100 - int(easy) - int(medium))


def difficulty_warning(easy: int, medium: int) -> Optional[str]:
    """Inline validation text (Principle 20)."""
    if easy + medium > 100:
        return f"Easy ({easy}%) + Medium ({medium}%) is {easy + medium}% — that is over 100%. Lower one of them."
    return None


def days_left(exam_date: Optional[_dt.date]) -> Optional[int]:
    if not exam_date:
        return None
    return (exam_date - _dt.date.today()).days


def file_role_badge(role: str) -> str:
    labels = {
        "syllabus": ("SYLLABUS", "#5ee7ff"),
        "past_paper": ("PAST PAPER", "#ffe600"),
        "teacher_priority": ("TEACHER", "#ff5da2"),
        "notes": ("NOTES", "#c6ff4d"),
        "pattern": ("PATTERN", "#b39dff"),
        "auto": ("AUTO", "#ffffff"),
    }
    text, colour = labels.get(role, labels["auto"])
    return badge(text, colour)


def detect_role(filename: str) -> str:
    """Auto-detect file roles so the user does not have to (Principle 17)."""
    name = (filename or "").lower()
    if any(word in name for word in ("syllab", "curricul", "course-outline")):
        return "syllabus"
    if any(word in name for word in ("paper", "question", "pyq", "past", "previous", "exam")):
        return "past_paper"
    if any(word in name for word in ("teacher", "hint", "whatsapp", "priority", "sir", "madam")):
        return "teacher_priority"
    if any(word in name for word in ("pattern", "blueprint", "scheme", "format")):
        return "pattern"
    if any(word in name for word in ("note", "ppt", "slide", "lecture", "unit")):
        return "notes"
    return "auto"
