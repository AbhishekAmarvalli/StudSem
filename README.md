# StudSem — Exam Intelligence Workflow

> Evidence-based exam preparation and prediction for **any subject** — school, diploma, undergraduate (engineering, science, commerce, arts, law, medicine…), postgraduate or professional exams.  
> Theory papers, practicals and vivas: every exam type gets outputs that fit it.  
> Stack: **LangGraph + Gemini (primary) + OpenRouter / Groq (fallbacks) + local web UI / Streamlit**  
> 100% free tier. No paid APIs. No local LLMs.

---

## Flow

```
STREAMLIT FORM
     ↓
FILE PROCESSOR (PDF/DOCX/PPTX → text; scans/photos → Gemini vision)
     ↓
CLASSIFIER  →  SYLLABUS ANALYST + PAPER ANALYST + TEACHER ANALYST
     ↓
KNOWLEDGE BASE (JSON)
     ↓
PREDICTOR (score + label + %)
     ↓
PRIORITY LIST / STUDY PLAN / MOCK PAPER / QUESTION BANK / MODEL ANSWERS / FLASHCARDS / VIVA SET
     ↓
EXAMINER (Groq) → retry if fail → EXPORT (PDF | HTML | Google Sheets)

SEPARATE: ANSWER GRADER
```

### Generation chunking

Long artifacts are built in pieces so a large subject can't lose its tail to an output-token limit. Thresholds:

| Artifact | One call covers | Above that |
|---|---|---|
| Mock paper | up to 12 planned questions, or a single section | one call per section; a short section gets one continuation call for its missing questions |
| Question bank | up to 3 units | one call per batch of 3 units, merged and de-duplicated |
| Model answers | up to 5 questions total | one call per section, then per 5-question batch within a section |

When the examiner rejects a paper, only the sections its issues name are regenerated; the rest are reused as-is. A section that comes back with fewer questions than its pattern (`7 x 2 marks`) asks for gets one continuation call for the remainder; if it is still short, the artifact is kept and a `only N of M questions` warning is added. A failed chunk keeps the rest of the artifact and adds a warning naming the missing section or units. If every chunk fails the run errors out instead of returning an empty artifact. While running, the Streamlit progress box shows the chunk in flight, e.g. `⏳ Writing mock paper section 2/3: …`, then which provider answered, e.g. `Finished mock paper section 2/3: … via Groq`.

---

## Landing Page (GitHub Pages)

`index.html` is a standalone, dependency-free neo-brutalist site for the project — open it directly in a browser, or publish it:

Live at **https://abhishekamarvalli.github.io/StudSem/** (Settings → Pages → `main` / `/ (root)`).

The exported HTML dashboard (`templates/dashboard.html`), the local web UI (`webapp/app.css`) and the Streamlit app (`ui.py` + `.streamlit/config.toml`) share one design language: cream paper, hard ink borders, offset shadows, one accent per screen.

---

## 1. Free API Key Setup

| Service | URL | Notes |
|---|---|---|
| **Gemini** | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | Free tier; Flash models are quota-generous |
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | Free tier; examiner LLM and last-resort generation fallback |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | Optional fallback; pick a `:free` model |

---

## 2. Local Run

```bash
# 1. Clone
git clone https://github.com/AbhishekAmarvalli/StudSem.git
cd StudSem

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env and fill in your API keys (GEMINI_API_KEY is required)
```

### 2a. Local web UI (recommended)

A proper neo-brutalist interface served from your machine — no Streamlit chrome, guided 3-step form, live per-stage progress and tabbed results.

```bash
python server.py            # → http://127.0.0.1:8000
```

- **Any subject**: pick a course level and discipline, or leave them generic — the prompts adapt question style to the subject.
- **Practical / viva exams**: choosing `Practical / Viva` or `Project viva` adds spoken-style viva questions with answer points, likely follow-ups and the practical tasks to be ready for.
- **Live progress**: one event per pipeline stage over server-sent events, plus warnings as they appear.
- **Exports**: priority list, mock paper, study plan, flashcards and viva set as PDF, plus the HTML dashboard.

The server keeps finished runs in memory for 6 hours; restarting it clears them.

### 2b. Streamlit UI

```bash
# Windows: double-click run.bat, or run this from the project folder
python -m streamlit run app.py

# Mac/Linux
streamlit run app.py
```

> **`'streamlit' is not recognized` in Windows cmd?** `pip` puts the `streamlit.exe`
> shim in your user Scripts folder (`%APPDATA%\Python\Python313\Scripts`), which is
> often not on `PATH`. `python -m streamlit run app.py` and `run.bat` both work
> without touching `PATH`; `run.bat` also picks the project `venv` when it has
> Streamlit installed.

> **Dark mode**: the app ships a light and a dark theme. Open the app menu (⋮,
> top-right) → **Theme** → *System / Light / Dark*. The CSS in `ui.py` follows the
> active theme, so cards, borders and alerts stay readable in both.

> **RAM note**: Designed for 8 GB RAM laptops. Files processed one at a time, never loaded all-at-once into memory.

---

## 3. GitHub Codespaces / github.dev

1. Open repo → **Code** → **Codespaces** → **New codespace**
2. In terminal:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your keys (Codespaces Secrets preferred)
   streamlit run app.py --server.port 8501
   ```
3. Codespaces will auto-forward port 8501.

For **github.dev** (browser IDE): edit files only; run via Codespaces.

---

## 4. Deploy to Streamlit Community Cloud (Primary)

1. Push repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select repo, branch `main`, file `app.py`.
4. Click **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "your_key"
GEMINI_MODEL = "gemini-3.8-flash"
GROQ_API_KEY = "your_key"          # optional; only the examiner uses it
GROQ_MODEL = "openai/gpt-oss-20b"
OPENROUTER_API_KEY = "your_key"    # optional quota fallback
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
# Optional:
GOOGLE_EXPORT_ENABLED = "false"
```

> Model IDs above are current as of November 2026. Retired IDs
> (`gemini-2.0-flash`, `llama3-8b-8192`, `mistralai/mistral-7b-instruct:free`)
> are detected and replaced with these defaults at runtime, so an older `.env`
> keeps working.

5. Deploy. The app URL will be `https://yourapp.streamlit.app`.

---

## 5. Deploy to Hugging Face Spaces (Fallback)

1. Create a Space: **New Space → Streamlit → Python 3.11**.
2. Upload all files or connect GitHub repo.
3. Add secrets in **Settings → Repository secrets**.
4. `app.py` is the entry point (no changes needed).

---

## 6. Google Sheets Export (Optional)

1. Create a Google Cloud project → Enable Sheets API + Drive API.
2. Create a Service Account → download JSON key.
3. Add the JSON contents to Streamlit secrets as `GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT` **or** place the file at `secrets/service_account.json` locally.
4. Set `GOOGLE_EXPORT_ENABLED=true`.
5. Share any target Sheets with the service account email.

---

## 7. Run Tests

```bash
pytest tests/ -v
```

Tests cover:
- `scoring.py` – all weight calculations, floor/ceiling, label assignment, percentage cap
- `schemas.py` – Pydantic model validation for all major outputs

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `google.api_core.exceptions.ResourceExhausted` | Gemini free quota hit. Calls fall back to OpenRouter, then Groq; add `OPENROUTER_API_KEY` / `GROQ_API_KEY` for a wider fallback net. |
| `All LLM providers failed` | Gemini, OpenRouter and Groq were all rate-limited or keyless. The error lists each provider's reason; wait for a quota reset or set another key. |
| Repeated 429s slow every chunk | A rate-limited provider is skipped for `LLM_PROVIDER_COOLDOWN` seconds (default 600) after it exhausts its retries, so later calls go straight to the next provider. Lower the value to retry sooner. |
| Wondering which model answered | A provider that just succeeded stays first for `LLM_PROVIDER_STICKY_SECONDS` (default 900) instead of re-probing the preferred one each chunk; progress lines end with `via Gemini/OpenRouter/Groq`. |
| Provider health | Shown as `Engines` in the Streamlit sidebar and as colour-coded chips in the local UI aside (`blue` ready, `green` in use, `red` paused, grey no key) with a tooltip explaining each state. `Test` sends one tiny request per configured provider; `Reset` clears paused/in-use markers; the strip refreshes itself when a cooldown ends. Cooldowns and stickiness persist in `data/provider_state.json`; override the path with `LLM_PROVIDER_STATE_FILE`. |
| `groq.RateLimitError` | Groq free tier limit. Wait or switch `GROQ_MODEL` to another free model. |
| `404 NOT_FOUND` / `model_not_found` | A model ID was retired. Remove `GEMINI_MODEL` / `GROQ_MODEL` / `OPENROUTER_MODEL` from `.env` to use the current defaults, or set a live ID. |
| Examiner skipped | `GROQ_API_KEY` is missing or the examiner call failed. The paper is still generated; it just is not cross-checked. |
| PDF text empty | File is likely a scan. The app auto-routes to Gemini vision. Ensure `GEMINI_API_KEY` is set. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again in your venv. |
| `'streamlit' is not recognized` (Windows) | Run `python -m streamlit run app.py` or double-click `run.bat`. Add `%APPDATA%\Python\Python313\Scripts` to `PATH` if you want the bare `streamlit` command everywhere. |
| Streamlit app crashes on deploy | Check **Manage app → Logs**. Usually a missing secret or import error. |
| `RAM` / `MemoryError` | The app processes files one page at a time. If still hitting limits, reduce file size or split large PDFs. |
| Marks don't add up in mock paper | Examiner will catch this and trigger a regeneration (max 2 retries). Check warnings section. |
| `… those sections/units are missing` warning | One chunk of a long generation failed or hit the output limit. The rest of the artifact is intact; re-run to fill the gap, or request fewer outputs at once. |
| Progress stuck on `⏳ Writing …` | That chunk is still generating — a batch of 10-mark answers can take ~30 s. If it never advances, check the terminal for a rate-limit error and retry. |

---

## 9. Project Structure

```
StudSem/
├── server.py               # Local web UI backend (FastAPI + SSE progress)
├── webapp/
│   ├── index.html          # Local UI markup (neo-brutalism)
│   ├── app.css             # Local UI design system
│   └── app.js              # Local UI logic (wizard, progress, results, grader)
├── app.py                  # Streamlit 4-step flow (upload → pattern → prefs → run)
├── ui.py                   # Streamlit design layer: CSS, stepper, badges, parsing helpers
├── pages/
│   └── 1_Answer_Grader.py  # Grader page (shares session state with the main app)
├── .streamlit/
│   └── config.toml         # Theme tokens for Streamlit
├── graph.py                # LangGraph workflow definition
├── schemas.py              # Pydantic models (shared typed state)
├── scoring.py              # Topic scoring engine
├── llm_client.py           # Unified LLM caller (Gemini → OpenRouter → Groq fallback)
├── prompt_loader.py        # Loads prompts/P*.md files
├── nodes/
│   ├── file_processor.py   # Text extraction + disk cache
│   ├── classifier.py       # Document role classifier
│   ├── syllabus_analyst.py # Syllabus → structured JSON
│   ├── paper_analyst.py    # Past papers → question stats
│   ├── teacher_analyst.py  # Teacher priority extractor
│   ├── knowledge_base.py   # Merge + persist KB
│   ├── predictor.py        # LLM factor extraction + scoring
│   ├── generators.py       # Mock paper, Q-bank, answers, flashcards, plan, viva
│   ├── examiner.py         # Groq examiner with retry
│   └── answer_grader.py    # Student answer grader
├── exporters/
│   ├── pdf_exporter.py     # ReportLab PDF export
│   ├── html_exporter.py    # Jinja2 HTML dashboard
│   └── sheets_exporter.py  # Google Sheets (optional)
├── prompts/
│   ├── P0.md … P9.md       # All runtime prompts (P6e = practical/viva set)
├── templates/
│   └── dashboard.html      # Jinja2 HTML dashboard (neo-brutalism)
├── index.html              # Standalone landing page (GitHub Pages)
├── data/                   # Cached knowledge bases (ephemeral on cloud)
├── tests/
│   ├── test_scoring.py
│   ├── test_schemas.py
│   └── test_local_server.py # Server integration test with stubbed nodes
├── .env.example
├── requirements.txt
└── README.md
```

---

## 10. v2 Ideas

- Per-topic self-rating so priorities adapt to your weak areas.
- Chroma vector store for large textbooks (node slot already reserved in `graph.py`).
- Telegram reminders from the study plan.
- Multi-user login if classmates join.

---

> *All predictions are evidence-based estimates, capped at 90%. This tool never claims a question will definitely appear.*
