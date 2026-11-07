# StudSem — Exam Intelligence Workflow

> Evidence-based exam preparation and prediction for **any subject** — school, diploma, undergraduate (engineering, science, commerce, arts, law, medicine…), postgraduate or professional exams.  
> Theory papers, practicals and vivas: every exam type gets outputs that fit it.  
> Stack: **LangGraph + Gemini (primary) + Groq (examiner) + local web UI / Streamlit**  
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

---

## Landing Page (GitHub Pages)

`index.html` is a standalone, dependency-free neo-brutalist site for the project — open it directly in a browser, or publish it:

Live at **https://abhishekamarvalli.github.io/StudSem/** (Settings → Pages → `main` / `/ (root)`).

The exported HTML dashboard (`templates/dashboard.html`) and the Streamlit skin (`static/neo_brutalism.css`) use the same design language: cream paper, 4px black borders, hard offset shadows, sticker accents.

---

## 1. Free API Key Setup

| Service | URL | Notes |
|---|---|---|
| **Gemini** | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | Free tier; Flash models are quota-generous |
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | Free tier; used as examiner LLM |
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
streamlit run app.py
```

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
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_API_KEY = "your_key"
GROQ_MODEL = "llama3-8b-8192"
OPENROUTER_API_KEY = "your_key"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"
# Optional:
GOOGLE_EXPORT_ENABLED = "false"
```

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
| `google.api_core.exceptions.ResourceExhausted` | Gemini free quota hit. Wait ~1 min or add `OPENROUTER_API_KEY` as fallback. |
| `groq.RateLimitError` | Groq free tier limit. Wait or switch `GROQ_MODEL` to another free model. |
| PDF text empty | File is likely a scan. The app auto-routes to Gemini vision. Ensure `GEMINI_API_KEY` is set. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again in your venv. |
| Streamlit app crashes on deploy | Check **Manage app → Logs**. Usually a missing secret or import error. |
| `RAM` / `MemoryError` | The app processes files one page at a time. If still hitting limits, reduce file size or split large PDFs. |
| Marks don't add up in mock paper | Examiner will catch this and trigger a regeneration (max 2 retries). Check warnings section. |

---

## 9. Project Structure

```
StudSem/
├── server.py               # Local web UI backend (FastAPI + SSE progress)
├── webapp/
│   ├── index.html          # Local UI markup (neo-brutalism)
│   ├── app.css             # Local UI design system
│   └── app.js              # Local UI logic (wizard, progress, results, grader)
├── app.py                  # Streamlit UI + page routing
├── graph.py                # LangGraph workflow definition
├── schemas.py              # Pydantic models (shared typed state)
├── scoring.py              # Topic scoring engine
├── llm_client.py           # Unified LLM caller (Gemini → OpenRouter fallback; Groq)
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
├── static/
│   └── neo_brutalism.css   # Streamlit neo-brutalism skin
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
