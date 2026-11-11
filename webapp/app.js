/* StudSem local UI — vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const esc = (value) =>
    String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const STAGE_LABELS = {
    extract: "Reading your documents",
    classifier: "Sorting the documents",
    syllabus_analyst: "Reading the syllabus",
    paper_analyst: "Mining past papers",
    teacher_analyst: "Reading teacher hints",
    knowledge_base: "Building the topic map",
    predictor: "Scoring topics",
    mock_paper: "Drafting the mock paper",
    examiner: "Examining the paper",
    model_answers: "Writing model answers",
    question_bank: "Building the question bank",
    flashcards: "Making flashcards",
    study_plan: "Planning your study days",
    viva: "Writing practical / viva questions",
  };

  const state = {
    files: [],
    running: false,
    runId: null,
    result: null,
  };

  /* ── Theme ──────────────────────────────────────────────────────────── */

  const THEME_KEY = "studsem-theme";
  const themeToggle = $("#theme-toggle");

  function applyTheme(theme, persist) {
    const dark = theme === "dark";
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    themeToggle.setAttribute("aria-pressed", String(dark));
    themeToggle.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    themeToggle.title = dark ? "Switch to light mode" : "Switch to dark mode";
    $(".theme-icon", themeToggle).textContent = dark ? "☀️" : "🌙";
    $(".theme-label", themeToggle).textContent = dark ? "Light" : "Dark";
    if (persist) {
      try {
        localStorage.setItem(THEME_KEY, dark ? "dark" : "light");
      } catch (err) {
        /* private mode: keep the theme for this page only */
      }
    }
  }

  themeToggle.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
  });

  /* ── API key status ─────────────────────────────────────────────────── */

  const PROVIDER_STATES = { ready: "ready", paused: "paused", preferred: "in use", off: "no key" };

  function renderProviders(providers) {
    const strip = $("#provider-health");
    if (!strip || !providers || !providers.length) return;
    strip.innerHTML = "Engines · " + providers
      .map((provider) => {
        const label = PROVIDER_STATES[provider.state] || provider.state;
        const wait = provider.state === "paused" && provider.paused_for
          ? ` ${Math.ceil(provider.paused_for / 60)}m`
          : "";
        return `<span class="engine ${esc(provider.state)}">${esc(provider.name)} ${esc(label)}${wait}</span>`;
      })
      .join(" ");
  }

  async function loadHealth() {
    const pill = $("#api-pill");
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      renderProviders(data.providers);
      if (data.gemini) {
        pill.className = "pill ok";
        pill.innerHTML = `<span class="dot" aria-hidden="true"></span> Gemini key ready · ${esc(data.model)}`;
      } else {
        pill.className = "pill bad";
        pill.innerHTML = `<span class="dot" aria-hidden="true"></span> No Gemini key — add one to .env`;
        $("#run-note").innerHTML =
          "Add <code>GEMINI_API_KEY=...</code> to your <code>.env</code> file, then restart <code>python server.py</code>.";
      }
    } catch (err) {
      pill.className = "pill bad";
      pill.textContent = "backend unreachable";
    }
  }

  const providerReset = $("#provider-reset");
  if (providerReset) {
    providerReset.addEventListener("click", async () => {
      providerReset.disabled = true;
      try {
        const res = await fetch("/api/providers/reset", { method: "POST" });
        const data = await res.json();
        renderProviders(data.providers);
      } catch (err) {
        /* backend gone; the strip keeps its last state */
      } finally {
        providerReset.disabled = false;
      }
    });
  }

  /* ── Views ──────────────────────────────────────────────────────────── */

  $$(".viewtab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".viewtab").forEach((t) => t.classList.toggle("active", t === tab));
      const view = tab.dataset.view;
      $("#build-view").hidden = view !== "build";
      $("#grade-view").hidden = view !== "grade";
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });

  /* ── Files ──────────────────────────────────────────────────────────── */

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");

  const ROLE_OPTIONS = [
    ["auto", "Auto-detect"],
    ["syllabus", "Syllabus"],
    ["past_paper", "Past paper"],
    ["teacher_priority", "Teacher priority"],
    ["notes", "Class notes"],
    ["pattern", "Exam pattern"],
  ];

  function renderFiles() {
    const list = $("#file-list");
    list.innerHTML = state.files
      .map((entry, index) => {
        const options = ROLE_OPTIONS.map(
          ([value, label]) =>
            `<option value="${value}" ${entry.role === value ? "selected" : ""}>${label}</option>`
        ).join("");
        const size = entry.file.size > 1048576
          ? `${(entry.file.size / 1048576).toFixed(1)} MB`
          : `${Math.max(1, Math.round(entry.file.size / 1024))} KB`;
        return `<div class="file-row">
          <span class="file-name" title="${esc(entry.file.name)}">${esc(entry.file.name)} <span class="meta-line">${size}</span></span>
          <select data-role-index="${index}" aria-label="Role for ${esc(entry.file.name)}">${options}</select>
          <button class="icon-btn" type="button" data-remove-index="${index}" aria-label="Remove ${esc(entry.file.name)}">×</button>
        </div>`;
      })
      .join("");
    updateSummary();
  }

  function addFiles(fileList) {
    Array.from(fileList).forEach((file) => {
      if (state.files.some((entry) => entry.file.name === file.name && entry.file.size === file.size)) return;
      state.files.push({ file, role: "auto" });
    });
    renderFiles();
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
    fileInput.value = "";
  });
  ["dragenter", "dragover"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("hot");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("hot");
    })
  );
  dropzone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

  $("#file-list").addEventListener("change", (event) => {
    const index = event.target.dataset.roleIndex;
    if (index === undefined) return;
    state.files[Number(index)].role = event.target.value;
  });

  $("#file-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove-index]");
    if (!button) return;
    state.files.splice(Number(button.dataset.removeIndex), 1);
    renderFiles();
  });

  /* ── Exam type → viva ───────────────────────────────────────────────── */

  const VIVA_EXAMS = ["Practical / Viva", "Project viva"];

  function currentExamType() {
    return $('input[name="exam-type"]:checked').value;
  }

  function syncViva() {
    const isViva = VIVA_EXAMS.includes(currentExamType());
    $("#viva-panel").hidden = !isViva;
    if (isViva) $("#viva-output").checked = true;
    updateSummary();
  }

  $$('input[name="exam-type"]').forEach((radio) => radio.addEventListener("change", syncViva));

  /* ── Summary ────────────────────────────────────────────────────────── */

  function selectedOutputs() {
    const outputs = $$("[data-output]")
      .filter((box) => box.checked)
      .map((box) => box.dataset.output);
    if (outputs.includes("flashcards")) outputs.push("formula_sheet");
    return outputs;
  }

  function updateSummary() {
    const subject = $("#subject").value.trim();
    $("#sum-subject").textContent = subject || "not set";
    $("#sum-exam").textContent = currentExamType();

    const count = state.files.length;
    $("#sum-files").textContent = count === 0 ? "no files yet" : `${count} file${count > 1 ? "s" : ""}`;

    const outputs = selectedOutputs();
    $("#sum-outputs").textContent =
      outputs.length === 0 ? "nothing selected" : `${outputs.filter((o) => o !== "formula_sheet").length} outputs`;
  }

  $("#subject").addEventListener("input", updateSummary);
  $$("[data-output]").forEach((box) => box.addEventListener("change", updateSummary));

  /* ── Config ─────────────────────────────────────────────────────────── */

  function numberValue(selector, fallback) {
    const parsed = Number($(selector).value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function splitList(value) {
    return String(value)
      .split(",")
      .map((part) => part.trim())
      .filter(Boolean);
  }

  function buildConfig() {
    const fileRoles = {};
    state.files.forEach((entry) => {
      fileRoles[entry.file.name] = entry.role;
    });

    const qTypes = $$('input[name="qtype"]:checked').map((box) => box.value);

    return {
      subject: $("#subject").value.trim(),
      exam_type: currentExamType(),
      course_level: $("#course-level").value,
      discipline: $("#discipline").value,
      exam_date: $("#exam-date").value || null,
      hours_per_day: numberValue("#hours", 4),
      mode: $("#mode").value,
      language: $("#language").value,
      pasted_text: $("#hints").value.trim(),
      outputs_wanted: selectedOutputs(),
      file_roles: fileRoles,
      viva_count: numberValue("#viva-count", 15),
      practical_focus: $("#practical-focus").value.trim(),
      difficulty_split: {
        easy: numberValue("#easy", 20),
        medium: numberValue("#medium", 50),
        hard: numberValue("#hard", 30),
      },
      unit_focus: splitList($("#unit-focus").value)
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value)),
      include_topics: splitList($("#include-topics").value),
      exclude_topics: splitList($("#exclude-topics").value),
      answer_length: {
        2: $("#len-2").value.trim() || "2-3 lines",
        5: $("#len-5").value.trim() || "~150 words",
        10: $("#len-10").value.trim() || "~350 words",
      },
      exam_pattern: {
        total_marks: numberValue("#total-marks", 100),
        duration: $("#duration").value.trim() || "3 hours",
        num_questions: numberValue("#num-questions", 10),
        sections: splitList($("#sections").value),
        internal_choice: $("#internal-choice").value.trim(),
        question_types: qTypes.length ? qTypes : ["short", "long"],
      },
    };
  }

  /* ── Progress rendering ─────────────────────────────────────────────── */

  function plannedStages(config) {
    const chain = [];
    if (state.files.length) chain.push("extract");
    chain.push("classifier", "syllabus_analyst", "paper_analyst", "teacher_analyst", "knowledge_base", "predictor");

    const hasPaper = config.outputs_wanted.includes("mock_paper") || config.outputs_wanted.includes("model_answers");
    if (hasPaper) chain.push("mock_paper", "examiner");
    if (config.outputs_wanted.includes("model_answers")) chain.push("model_answers");
    if (config.outputs_wanted.includes("question_bank")) chain.push("question_bank");
    if (config.outputs_wanted.includes("flashcards")) chain.push("flashcards");
    if (config.outputs_wanted.includes("study_plan")) chain.push("study_plan");
    if (config.outputs_wanted.includes("viva") || VIVA_EXAMS.includes(config.exam_type)) chain.push("viva");
    return chain;
  }

  function renderStages(nodes) {
    $("#stage-list").innerHTML = nodes
      .map(
        (node) => `<li class="stage" data-stage="${node}">
          <span class="box" aria-hidden="true"></span>
          <span>${esc(STAGE_LABELS[node] || node)}</span>
        </li>`
      )
      .join("");
  }

  const PROGRESS_IDLE = "Each stage is one AI pass. Keep this tab open.";

  function setProgressText(text, busy = false) {
    $("#progress-text").textContent = text;
    $("#progress-spinner").hidden = !busy;
  }

  function markStage(node, status) {
    const item = $(`[data-stage="${node}"]`);
    if (!item) return;
    item.classList.remove("running");
    if (status === "done") {
      item.classList.add("done");
      item.querySelector(".box").textContent = "✓";
      setProgressText(PROGRESS_IDLE);
      const next = $(".stage:not(.done)");
      if (next) next.classList.add("running");
    } else {
      item.classList.add("running");
    }
    const done = $$(".stage.done").length;
    $("#progress-count").textContent = `${done}/${$$(".stage").length} stages`;
  }

  function addWarning(message) {
    const feed = $("#warn-feed");
    if (feed.querySelectorAll(".warn").length > 8) return;
    const div = document.createElement("div");
    div.className = "warn";
    div.textContent = message;
    feed.appendChild(div);
  }

  function failRun(message) {
    state.running = false;
    $("#run-btn").disabled = false;
    $("#run-btn").textContent = "Try again →";
    $("#progress-title").textContent = "That run stopped";
    setProgressText("Fix the issue below and run again — nothing was saved.");
    const div = document.createElement("div");
    div.className = "warn";
    div.style.background = "var(--pink)";
    div.textContent = message;
    $("#warn-feed").appendChild(div);
  }

  /* ── Run ────────────────────────────────────────────────────────────── */

  $("#run-btn").addEventListener("click", startRun);

  async function startRun() {
    if (state.running) return;

    const config = buildConfig();
    if (!config.subject) {
      $("#subject").focus();
      $("#run-note").textContent = "Add a subject name first — that is the one thing the pipeline cannot guess.";
      return;
    }
    if (state.files.length === 0 && !config.pasted_text) {
      $("#run-note").textContent = "Add at least one document, or paste the teacher's hints in step 2.";
      $("#dropzone").scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    if (config.outputs_wanted.length === 0) {
      $("#run-note").textContent = "Pick at least one output in step 3, otherwise there is nothing to build.";
      return;
    }

    state.running = true;
    state.result = null;
    $("#run-btn").disabled = true;
    $("#run-btn").textContent = "Running…";
    $("#warn-feed").innerHTML = "";
    $("#results").hidden = true;

    const nodes = plannedStages(config);
    renderStages(nodes);
    const stages = $("#progress");
    stages.hidden = false;
    $("#progress-title").textContent = "Working through the pipeline";
    setProgressText(PROGRESS_IDLE);
    $("#progress-count").textContent = `0/${nodes.length} stages`;
    stages.scrollIntoView({ behavior: "smooth", block: "start" });
    markStage(nodes[0], "start");

    const form = new FormData();
    form.append("config", JSON.stringify(config));
    state.files.forEach((entry) => form.append("files", entry.file, entry.file.name));

    let response;
    try {
      response = await fetch("/api/run", { method: "POST", body: form });
    } catch (err) {
      failRun("Could not reach the local server. Is `python server.py` still running?");
      return;
    }

    if (!response.ok) {
      let detail = `Request failed (${response.status}).`;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (err) {
        /* keep default */
      }
      failRun(detail);
      return;
    }

    const { run_id: runId } = await response.json();
    state.runId = runId;
    listenForEvents(runId);
  }

  function listenForEvents(runId) {
    const source = new EventSource(`/api/run/${runId}/events`);

    source.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch (err) {
        return;
      }

      if (payload.type === "stage") {
        markStage(payload.node, payload.status === "done" ? "done" : "start");
      } else if (payload.type === "chunk") {
        setProgressText(payload.message, true);
        renderProviders(payload.providers);
      } else if (payload.type === "warning") {
        addWarning(payload.message);
      } else if (payload.type === "error") {
        source.close();
        failRun(payload.message);
      } else if (payload.type === "done") {
        source.close();
        finishRun(runId);
      }
    };

    source.onerror = () => {
      if (!state.running) {
        source.close();
        return;
      }
      // EventSource retries on its own; give it one cycle before complaining.
      setTimeout(() => {
        if (state.running && source.readyState === EventSource.CLOSED) {
          failRun("Lost the connection to the local server mid-run. Check the terminal for the error.");
        }
      }, 4000);
    };
  }

  async function finishRun(runId) {
    try {
      const res = await fetch(`/api/run/${runId}/result`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Could not load the result (${res.status}).`);
      }
      state.result = await res.json();
    } catch (err) {
      failRun(err.message);
      return;
    }

    state.running = false;
    $("#run-btn").disabled = false;
    $("#run-btn").textContent = "Run again →";
    $("#progress-title").textContent = "Pipeline finished";
    setProgressText("Scroll down — your prep is below.");

    renderResult(state.result);
    $("#results").hidden = false;
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ── Result rendering ───────────────────────────────────────────────── */

  function groupTopics(topics) {
    const groups = { "MUST STUDY": [], "SHOULD STUDY": [], OPTIONAL: [] };
    topics.forEach((topic) => {
      const key = groups[topic.label] ? topic.label : "OPTIONAL";
      groups[key].push(topic);
    });
    return groups;
  }

  function renderTopics(panel, result) {
    if (!result.topics.length) {
      panel.innerHTML = `<div class="empty">No priority list in this run.</div>`;
      return;
    }
    const groups = groupTopics(result.topics);
    const headings = {
      "MUST STUDY": { klass: "must", label: "Must study" },
      "SHOULD STUDY": { klass: "should", label: "Should study" },
      OPTIONAL: { klass: "optional", label: "Optional" },
    };

    panel.innerHTML = Object.entries(groups)
      .filter(([, topics]) => topics.length)
      .map(([key, topics]) => {
        const heading = headings[key];
        const cards = topics
          .map((topic) => {
            const pct = Number(topic.percentage || 0);
            const factors = Object.entries(topic.factors || {})
              .map(([name, value]) => `${name} ${value}`)
              .join(" · ");
            return `<article class="topic ${heading.klass}">
              <div class="topic-top">
                <h3>${esc(topic.topic)}</h3>
                <span class="unit">Unit ${esc(topic.unit)} · ${pct}%</span>
              </div>
              <div class="bar"><i style="width:${Math.max(2, Math.min(100, pct))}%"></i></div>
              <p>${esc(topic.evidence || "No evidence text returned.")}</p>
              ${topic.question_types && topic.question_types.length
                ? `<p class="types">Likely as: ${esc(topic.question_types.join(", "))}</p>`
                : ""}
              ${factors ? `<p class="meta-line">Score parts: ${esc(factors)}</p>` : ""}
            </article>`;
          })
          .join("");
        return `<h3 class="group-title">${heading.label}<span class="tag">${topics.length}</span></h3>${cards}`;
      })
      .join("");
  }

  function renderPaper(panel, result) {
    const paper = result.mock_paper;
    if (!paper || !paper.sections || !paper.sections.length) {
      panel.innerHTML = `<div class="empty">No mock paper in this run.</div>`;
      return;
    }
    const header = paper.header || {};
    const head = `<div class="meta-line">${esc(header.duration || "")} ${header.total_marks ? `· ${esc(header.total_marks)} marks` : ""}</div>`;
    const sections = paper.sections
      .map(
        (section) => `<h3 class="group-title">${esc(section.name || "Section")}` +
          `${section.instructions ? `<span class="tag">${esc(section.instructions)}</span>` : ""}</h3>` +
          (section.questions || [])
            .map(
              (q) => `<div class="qa">
                <p class="q">Q${esc(q.id)}. ${esc(q.text)}</p>
                <p class="meta-line">${esc(q.marks)} marks · ${esc(q.type)} · ${esc(q.difficulty)} · ${esc(q.topic || "")} ${q.unit ? `(unit ${esc(q.unit)})` : ""} · ${esc(q.source_tag || "")}</p>
              </div>`
            )
            .join("")
      )
      .join("");
    panel.innerHTML = head + sections;
  }

  function renderViva(panel, result) {
    const viva = result.viva;
    if (!viva) {
      panel.innerHTML = `<div class="empty">No viva set in this run. Pick a practical/viva exam type or tick the viva output.</div>`;
      return;
    }
    const kinds = [
      ["theory", "Theory checks"],
      ["application", "Application / why-how"],
      ["practical", "Practical &amp; lab"],
      ["troubleshooting", "Troubleshooting"],
    ];
    const questions = viva.questions || [];
    const tasks = (viva.practical_tasks || []).length
      ? `<h3 class="group-title">Practical tasks to be ready for</h3>
         <div class="qa"><ul>${viva.practical_tasks.map((task) => `<li>${esc(task)}</li>`).join("")}</ul></div>`
      : "";

    const grouped = kinds
      .map(([kind, heading]) => {
        const items = questions.filter((q) => (q.kind || "theory") === kind);
        if (!items.length) return "";
        const cards = items
          .map(
            (q, index) => `<div class="qa">
              <p class="q">${index + 1}. ${esc(q.question)}</p>
              ${q.answer_points && q.answer_points.length
                ? `<ul>${q.answer_points.map((point) => `<li>${esc(point)}</li>`).join("")}</ul>`
                : ""}
              ${q.follow_ups && q.follow_ups.length
                ? `<div class="follow"><b>Likely follow-ups</b>${q.follow_ups.map((f) => esc(f)).join(" · ")}</div>`
                : ""}
              ${q.topic ? `<p class="meta-line">${esc(q.topic)}${q.unit ? ` · unit ${esc(q.unit)}` : ""}</p>` : ""}
            </div>`
          )
          .join("");
        return `<h3 class="group-title">${heading}<span class="tag">${items.length}</span></h3>${cards}`;
      })
      .join("");

    const leftovers = questions.filter(
      (q) => !kinds.some(([kind]) => kind === (q.kind || "theory"))
    );
    const extra = leftovers
      .map((q) => `<div class="qa"><p class="q">${esc(q.question)}</p></div>`)
      .join("");

    panel.innerHTML = tasks + grouped + extra || `<div class="empty">The viva set came back empty.</div>`;
  }

  function renderPlan(panel, result) {
    const plan = result.study_plan;
    if (!plan || !plan.days || !plan.days.length) {
      panel.innerHTML = `<div class="empty">No study plan in this run.</div>`;
      return;
    }
    const rows = plan.days
      .map((day) => {
        const blocks = (day.blocks || [])
          .map((block) => `${esc(block.topic)} — ${esc(block.activity)} (${esc(block.minutes)} min)`)
          .join("<br>");
        return `<tr><td>${esc(day.date)}</td><td>${esc(day.hours)}h</td><td>${blocks || "—"}</td></tr>`;
      })
      .join("");
    const warnings = (plan.warnings || []).length
      ? `<div class="warn-feed">${plan.warnings.map((w) => `<div class="warn">${esc(w)}</div>`).join("")}</div>`
      : "";
    panel.innerHTML = `<table class="plan">
      <thead><tr><th>Date</th><th>Hours</th><th>What to do</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>${warnings}`;
  }

  function renderCards(panel, result) {
    const cards = result.flashcards;
    if (!cards) {
      panel.innerHTML = `<div class="empty">No flashcards in this run.</div>`;
      return;
    }
    const flash = (cards.flashcards || [])
      .map(
        (card) => `<div class="qa">
          <p class="q">${esc(card.front)}</p>
          <p style="font-size:.88rem;">${esc(card.back)}</p>
          ${card.topic ? `<p class="meta-line">${esc(card.topic)}</p>` : ""}
        </div>`
      )
      .join("");
    const sheets = (cards.formula_units || [])
      .map(
        (unit) => `<details class="advanced" style="margin-bottom:.7rem;">
          <summary>Unit ${esc(unit.unit)} — quick revision</summary>
          <div class="advanced-body" style="font-size:.88rem; white-space:pre-wrap;">${esc(unit.content)}</div>
        </details>`
      )
      .join("");
    panel.innerHTML =
      (flash ? `<h3 class="group-title">Flashcards<span class="tag">${(cards.flashcards || []).length}</span></h3>${flash}` : "") +
      (sheets ? `<h3 class="group-title">Revision sheets</h3>${sheets}` : "") ||
      `<div class="empty">The flashcard set came back empty.</div>`;
  }

  function renderAnswers(panel, result) {
    const doc = result.model_answers;
    if (!doc || !(doc.answers || []).length) {
      panel.innerHTML = `<div class="empty">No model answers in this run.</div>`;
      return;
    }
    panel.innerHTML = doc.answers
      .map(
        (answer) => `<div class="qa">
          <p class="q">${esc(answer.question_text)} <span class="meta-line">${esc(answer.marks)} marks</span></p>
          <div style="font-size:.88rem; white-space:pre-wrap; margin-top:.4rem;">${esc(answer.answer)}</div>
          ${(answer.marking_scheme || []).length
            ? `<div class="follow"><b>Marking scheme</b><ul>${answer.marking_scheme.map((line) => `<li>${esc(line)}</li>`).join("")}</ul></div>`
            : ""}
        </div>`
      )
      .join("");
  }

  function renderBank(panel, result) {
    const bank = result.question_bank;
    if (!bank || !(bank.units || []).length) {
      panel.innerHTML = `<div class="empty">No question bank in this run.</div>`;
      return;
    }
    panel.innerHTML = bank.units
      .map((unit) => {
        const groups = [
          ["must", "Must"],
          ["should", "Should"],
          ["optional", "Optional"],
        ]
          .map(([key, label]) => {
            const items = unit[key] || [];
            if (!items.length) return "";
            return `<h3 class="group-title">${label}<span class="tag">${items.length}</span></h3>` +
              items
                .map(
                  (q) => `<div class="qa">
                    <p class="q">${esc(q.text)}</p>
                    <p class="meta-line">${esc(q.marks)} marks · ${esc(q.type)} · ${esc(q.difficulty)} · ${esc(q.topic || "")}</p>
                  </div>`
                )
                .join("");
          })
          .join("");
        return `<h3 class="group-title">Unit ${esc(unit.unit)}</h3>${groups}`;
      })
      .join("");
  }

  function renderDownloads(result) {
    const runId = state.runId;
    const links = [];
    const add = (kind, label, present) => {
      if (present) links.push(`<a class="btn sm" href="/api/run/${runId}/download/${kind}">${label}</a>`);
    };
    add("priority", "Priority PDF", result.topics.length);
    add("mock", "Mock paper PDF", !!result.mock_paper);
    add("viva", "Viva PDF", !!result.viva);
    add("plan", "Study plan PDF", !!result.study_plan);
    add("flashcards", "Flashcards PDF", !!result.flashcards);
    add("dashboard", "HTML dashboard", true);

    $("#downloads").innerHTML = links.length
      ? links.join("")
      : `<span class="hint">Nothing to download from this run.</span>`;
  }

  function renderResult(result) {
    $("#result-title").textContent = `${result.subject} — prep ready`;
    $("#result-sub").textContent =
      "Ranked from your evidence only. Treat every percentage as a likelihood estimate, not a promise.";

    const must = result.topics.filter((topic) => topic.label === "MUST STUDY").length;
    const meta = [
      result.exam_type,
      result.course_level,
      `${result.topics.length} topics scored`,
      `${must} must-study`,
    ].filter(Boolean);
    $("#result-meta").innerHTML = meta.map((item) => `<span>${esc(item)}</span>`).join("");

    renderTopics($('[data-panel="topics"]'), result);
    renderPaper($('[data-panel="paper"]'), result);
    renderViva($('[data-panel="viva"]'), result);
    renderPlan($('[data-panel="plan"]'), result);
    renderCards($('[data-panel="cards"]'), result);
    renderAnswers($('[data-panel="answers"]'), result);
    renderBank($('[data-panel="bank"]'), result);
    renderDownloads(result);

    const resultWarnings = result.warnings || [];
    const warningBox = $("#result-warnings");
    warningBox.innerHTML = resultWarnings
      .map((warning) => `<div class="warn">${esc(warning)}</div>`)
      .join("");
    warningBox.hidden = resultWarnings.length === 0;
    resultWarnings.forEach(addWarning);

    const available = {
      topics: result.topics.length > 0,
      paper: !!result.mock_paper,
      viva: !!result.viva,
      plan: !!result.study_plan,
      cards: !!result.flashcards,
      answers: !!(result.model_answers && result.model_answers.answers || []).length,
      bank: !!(result.question_bank && result.question_bank.units || []).length,
    };
    $$("#result-tabs .tab").forEach((tab) => {
      tab.hidden = !available[tab.dataset.tab];
    });

    const firstAvailable = $$("#result-tabs .tab").find((tab) => !tab.hidden);
    if (firstAvailable) showTab(firstAvailable.dataset.tab);
  }

  function showTab(name) {
    $$("#result-tabs .tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
    $$(".panel[data-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.panel !== name;
    });
  }

  $("#result-tabs").addEventListener("click", (event) => {
    const tab = event.target.closest("[data-tab]");
    if (tab) showTab(tab.dataset.tab);
  });

  /* ── Grader ─────────────────────────────────────────────────────────── */

  $("#grade-btn").addEventListener("click", async () => {
    const payload = {
      question: $("#g-question").value.trim(),
      answer: $("#g-answer").value.trim(),
      marks: numberValue("#g-marks", 5),
      model_answer: $("#g-model").value.trim(),
      syllabus_context: $("#g-context").value.trim(),
    };
    const output = $("#grade-result");

    if (!payload.question || !payload.answer) {
      output.hidden = false;
      output.innerHTML = `<div class="warn">Add both the question and your answer.</div>`;
      return;
    }

    const button = $("#grade-btn");
    button.disabled = true;
    button.textContent = "Grading…";
    output.hidden = false;
    output.innerHTML = `<div class="panel"><p>Reading your answer like an examiner…</p></div>`;

    try {
      const res = await fetch("/api/grade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || `Grading failed (${res.status}).`);
      renderGrading(body);
    } catch (err) {
      output.innerHTML = `<div class="warn" style="background:var(--pink)">${esc(err.message)}</div>`;
    } finally {
      button.disabled = false;
      button.textContent = "Grade it →";
    }
  });

  function renderGrading(result) {
    const rubric = (result.rubric || [])
      .map(
        (point) => `<tr>
          <td>${esc(point.point)}</td>
          <td>${esc(point.awarded)} / ${esc(point.max)}</td>
          <td>${esc(point.comment || "")}</td>
        </tr>`
      )
      .join("");

    const list = (items, heading, color) =>
      (items || []).length
        ? `<h3 class="group-title">${heading}<span class="tag" style="background:${color}">${items.length}</span></h3>
           <div class="qa"><ul>${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div>`
        : "";

    $("#grade-result").innerHTML = `
      <div class="marks-big">${esc(result.marks_awarded)} <small>/ ${esc(result.out_of)} marks</small></div>
      ${rubric ? `<div class="panel"><h3 class="group-title">Rubric</h3>
        <table class="plan"><thead><tr><th>Point</th><th>Awarded</th><th>Comment</th></tr></thead>
        <tbody>${rubric}</tbody></table></div>` : ""}
      <div class="panel" style="margin-top:1rem;">
        ${list(result.missing_points, "Missing points", "var(--pink)")}
        ${list(result.errors, "Errors in your answer", "var(--yellow)")}
        ${result.improved_answer
          ? `<h3 class="group-title">Improved answer</h3>
             <div class="qa" style="white-space:pre-wrap; font-size:.88rem;">${esc(result.improved_answer)}</div>`
          : ""}
        ${list(result.tips, "Tips", "var(--cyan)")}
      </div>`;
  }

  /* ── Boot ───────────────────────────────────────────────────────────── */

  const today = new Date();
  today.setDate(today.getDate() + 14);
  $("#exam-date").value = today.toISOString().slice(0, 10);

  applyTheme(document.documentElement.dataset.theme || "light", false);
  loadHealth();
  syncViva();
  renderFiles();
})();
