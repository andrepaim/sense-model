"use strict";
const DEMO_EXAMPLES = [
  {
    id: "demo-smoke-0",
    text: "She sat on the river bank.",
    target: { start: 21, end: 25, text: "bank" },
    candidates: [
      { id: "river", definition: "sloping land beside a body of water" },
      {
        id: "finance",
        definition: "a financial institution that accepts deposits",
      },
    ],
    language: "en",
    definition_language: "en",
    provenance: {
      source: "authored demonstration",
      document_id: "smoke-0",
      license: "MIT",
      annotation: "synthetic",
      split_group: "smoke-0",
      demo: true,
      review_status: "demo-only",
    },
  },
  {
    id: "demo-smoke-1",
    text: "He deposited money at the bank.",
    target: { start: 26, end: 30, text: "bank" },
    candidates: [
      { id: "river", definition: "sloping land beside a body of water" },
      {
        id: "finance",
        definition: "a financial institution that accepts deposits",
      },
    ],
    language: "en",
    definition_language: "en",
    provenance: {
      source: "authored demonstration",
      document_id: "smoke-1",
      license: "MIT",
      annotation: "synthetic",
      split_group: "smoke-1",
      demo: true,
      review_status: "demo-only",
    },
  },
  {
    id: "demo-smoke-2",
    text: "The model predicts the next token.",
    target: { start: 4, end: 9, text: "model" },
    candidates: [
      {
        id: "representation",
        definition: "a simplified representation of a system",
      },
      { id: "person", definition: "a person who poses for an artist" },
    ],
    language: "en",
    definition_language: "en",
    provenance: {
      source: "authored demonstration",
      document_id: "smoke-2",
      license: "MIT",
      annotation: "synthetic",
      split_group: "smoke-2",
      demo: true,
      review_status: "demo-only",
    },
  },
  {
    id: "demo-smoke-3",
    text: "She decided to give up.",
    target: { start: 15, end: 22, text: "give up" },
    candidates: [
      {
        id: "stop",
        definition: "stop maintaining or insisting on; of ideas or claims",
      },
    ],
    language: "en",
    definition_language: "en",
    provenance: {
      source: "authored demonstration",
      document_id: "smoke-3",
      license: "MIT",
      annotation: "synthetic",
      split_group: "smoke-3",
      demo: true,
      review_status: "demo-only",
    },
  },
  {
    id: "demo-smoke-4",
    text: "Ela sentou no banco da praça.",
    target: { start: 14, end: 19, text: "banco" },
    candidates: [
      { id: "seat", definition: "Assento comprido para várias pessoas." },
      {
        id: "finance",
        definition: "Instituição que presta serviços financeiros.",
      },
    ],
    language: "pt-BR",
    definition_language: "pt-BR",
    provenance: {
      source: "authored demonstration",
      document_id: "smoke-4",
      license: "MIT",
      annotation: "synthetic",
      split_group: "smoke-4",
      demo: true,
      review_status: "demo-only",
    },
  },
  {
    id: "demo-smoke-5",
    text: "Ela depositou dinheiro no banco.",
    target: { start: 26, end: 31, text: "banco" },
    candidates: [
      { id: "seat", definition: "Assento comprido para várias pessoas." },
      {
        id: "finance",
        definition: "Instituição que presta serviços financeiros.",
      },
    ],
    language: "pt-BR",
    definition_language: "pt-BR",
    provenance: {
      source: "authored demonstration",
      document_id: "smoke-5",
      license: "MIT",
      annotation: "synthetic",
      split_group: "smoke-5",
      demo: true,
      review_status: "demo-only",
    },
  },
];
const $ = (id) => document.getElementById(id);
const state = {
  csrf: "",
  reviewer: "",
  credential: { configured: false },
  examples: [],
  current: null,
  detail: null,
  stats: {},
  loading: 0,
  busy: false,
};
const OUTCOMES = {
  selected: "Selected meanings",
  none_fits: "None fits",
  insufficient_context: "Insufficient context",
  extraction_issue: "Extraction issue",
  inventory_issue: "Dictionary issue",
  skip: "Skipped",
};
const BUCKETS = {
  jev_high_confidence: "Jev: high confidence",
  jev_low_codex_agreed: "Jev low confidence; Codex agreed",
  jev_low_codex_disagreed: "Jev low confidence; Codex disagreed",
  jev_error_codex: "Jev error; Codex labeled",
  unresolved: "Unresolved",
};
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function clear(node) {
  node.replaceChildren();
}
function languageName(code) {
  return (
    {
      en: "English",
      "pt-BR": "Portuguese · Brazil",
      "pt-PT": "Portuguese · Portugal",
      pt: "Portuguese",
    }[code] ||
    code ||
    "Unspecified"
  );
}
function notice(message, error = false) {
  const node = $("notice");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.hidden = !message;
}
async function api(path, body, method) {
  const options = {
    method: method || (body === undefined ? "GET" : "POST"),
    headers: {},
    credentials: "same-origin",
  };
  if (options.method !== "GET") options.headers["X-CSRF-Token"] = state.csrf;
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error(
      "The local workspace is unavailable. Check that the annotation server is running.",
    );
  }
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error("The workspace returned an unreadable response.");
  }
  if (!response.ok)
    throw new Error(result.error || `Request failed (${response.status}).`);
  return result;
}
async function action(task) {
  if (state.busy) return;
  state.busy = true;
  document.body.setAttribute("aria-busy", "true");
  updateActions();
  try {
    await task();
  } catch (error) {
    notice(error.message, true);
    if ($("settings-dialog").open)
      $("credential-status").textContent = error.message;
  } finally {
    state.busy = false;
    document.body.removeAttribute("aria-busy");
    updateActions();
    if (state.refreshRequested) {
      state.refreshRequested = false;
      action(() => refreshQueue());
    }
  }
}
async function refreshSession() {
  const session = await api("/api/session");
  state.csrf = session.csrf;
  state.reviewer = session.reviewer || "";
  state.credential = session.credential || { configured: false };
  $("reviewer-name").value = state.reviewer;
  renderCredentials();
}
function renderCredentials() {
  const key = state.credential;
  $("key-dot").textContent = key.configured ? "●" : "○";
  $("credential-status").textContent = key.configured
    ? `Key present ${key.persisted ? "in local config" : "in session memory"} · ${key.provider} · ${key.model}. Connection is checked when you request a suggestion.`
    : "No API key saved for this session.";
  if (key.provider) $("provider").value = key.provider;
  if (key.model) $("jev-model").value = key.model;
  $("save-to-config").checked = Boolean(key.persisted);
  updateActions();
}
function normalizeSummary(row) {
  const ex = row.example || row;
  return {
    ...row,
    id: row.id || ex.id,
    text: row.text || ex.text || "",
    target: row.target || ex.target || {},
    language: row.language || ex.language || "",
    status: row.status || "pending",
  };
}
async function refreshQueue(preferred = state.current) {
  if (!state.reviewer) {
    state.examples = [];
    state.current = null;
    state.detail = null;
    renderQueue();
    $("empty-state").hidden = false;
    $("review-workspace").hidden = true;
    return;
  }
  const query = new URLSearchParams({
    language: $("language-filter").value,
    status: $("status-filter").value,
    ...Object.fromEntries(
      ["bucket", "difficulty", "area", "paper", "confidence"].map((key) => [
        key,
        $(key + "-filter").value,
      ]),
    ),
  });
  const [result, all] = await Promise.all([
    api("/api/examples?" + query),
    api("/api/examples"),
  ]);
  for (const key of ["difficulty", "area", "paper"]) {
    const select = $(key + "-filter"),
      selected = select.value;
    const first = select.options[0].textContent;
    select.replaceChildren(new Option(first, ""));
    [...new Set(all.examples.map((row) => row[key]).filter(Boolean))]
      .sort()
      .forEach((value) => select.add(new Option(value, value)));
    select.value = selected;
  }
  state.examples = (result.examples || []).map(normalizeSummary);
  renderQueue();
  await refreshStats();
  if (state.examples.length) {
    const id = state.examples.some((e) => e.id === preferred)
      ? preferred
      : state.examples[0].id;
    await selectExample(id);
  } else {
    state.current = null;
    state.detail = null;
    $("empty-state").hidden = false;
    $("review-workspace").hidden = true;
    $("empty-state").querySelector("h2").textContent = $("status-filter").value
      ? "No examples in this view."
      : "Good labels start with an independent reading.";
  }
}
function renderQueue() {
  clear($("queue"));
  $("queue-count").textContent = state.examples.length;
  if (!state.examples.length) {
    $("queue").append(
      element(
        "p",
        "Import candidates or change filters to begin.",
        "queue-empty",
      ),
    );
    return;
  }
  state.examples.forEach((row, index) => {
    const button = element("button", undefined, "queue-item");
    button.type = "button";
    button.dataset.exampleId = row.id;
    button.classList.toggle("active", row.id === state.current);
    if (row.id === state.current) button.setAttribute("aria-current", "true");
    const top = element("span", undefined, "queue-top");
    top.append(
      element(
        "strong",
        typeof row.target === "string"
          ? row.target
          : row.target?.text || row.id,
      ),
      element("span", row.status.replaceAll("_", " "), "queue-status"),
    );
    button.append(
      top,
      element(
        "small",
        row.text || `${languageName(row.language)} · example ${index + 1}`,
      ),
    );
    button.addEventListener("click", () => action(() => selectExample(row.id)));
    $("queue").append(button);
  });
}
async function selectExample(id) {
  const seq = ++state.loading;
  if (state.examples.find((row) => row.id === id)?.automatic)
    $("adjudication-mode").checked = false;
  const path = $("adjudication-mode").checked
    ? "/api/adjudication"
    : "/api/example";
  const detail = await api(path + "?" + new URLSearchParams({ id }));
  if (seq !== state.loading) return;
  state.current = id;
  state.detail = detail;
  $("empty-state").hidden = true;
  $("review-workspace").hidden = false;
  renderQueue();
  renderExample();
  notice("");
}
function renderExample() {
  const detail = state.detail,
    ex = detail.example,
    p = ex.provenance || {};
  const automatic = detail.mode === "automatic" || Boolean(p.automatic);
  $("automatic-actions").hidden = !automatic;
  $("automatic-evidence").hidden = !automatic;
  $("legacy-assistance").hidden = automatic;
  $("legacy-outcomes").hidden = automatic;
  $("protocol-link").href =
    `https://github.com/andrepaim/sense-model/blob/main/docs/${automatic ? "automatic-review" : "pilot-plan"}.md`;
  $("protocol-link").textContent = automatic
    ? "Read the review guide ↗"
    : "Read the pilot plan ↗";
  $("adjudication-mode").disabled = automatic;
  $("adjudication-mode").closest("label").hidden = automatic;
  $("selection-help").textContent = automatic
    ? "Check every acceptable definition, or choose none fits / insufficient context. Save a correction, approve the automatic result, or reject the example."
    : "Select every acceptable definition. The order is shuffled for your review.";
  const index = state.examples.findIndex((e) => e.id === ex.id);
  $("example-counter").textContent =
    `EXAMPLE ${String(index + 1).padStart(2, "0")} OF ${state.examples.length}`;
  $("example-language").textContent = languageName(ex.language);
  $("demo-badge").hidden = !(
    p.demo ||
    p.annotation === "synthetic" ||
    ex.id.startsWith("demo-") ||
    detail.demo
  );
  $("domain-badge").textContent = p.domain || "Unspecified domain";
  clear($("sentence"));
  const points = Array.from(ex.text);
  $("sentence").append(
    document.createTextNode(points.slice(0, ex.target.start).join("")),
    element("mark", points.slice(ex.target.start, ex.target.end).join("")),
    document.createTextNode(points.slice(ex.target.end).join("")),
  );
  clear($("source-details"));
  [
    p.source,
    p.paper_title,
    (p.generation?.page ?? p.page) !== undefined
      ? `Page: ${p.generation?.page ?? p.page}`
      : "",
    p.generation?.difficulty ? `Difficulty: ${p.generation.difficulty}` : "",
    p.generation?.area ? `Area: ${p.generation.area}` : "",
    p.generation?.paper_id ? `Paper: ${p.generation.paper_id}` : "",
    p.document_id ? `Document ${p.document_id}` : "",
    p.license ? `License: ${p.license}` : "License not recorded",
    p.revision ? `Revision: ${p.revision}` : "",
  ]
    .filter(Boolean)
    .forEach((value) => $("source-details").append(element("span", value)));
  const sourceURL = p.source_url || p.paper_url || p.url;
  if (sourceURL) {
    try {
      const url = new URL(sourceURL);
      if (["http:", "https:"].includes(url.protocol)) {
        const a = element("a", "Open original ↗");
        a.href = url.href;
        a.target = "_blank";
        a.rel = "noreferrer";
        $("source-details").append(a);
      }
    } catch {}
  }
  clear($("candidate-options"));
  $("candidate-options").append(
    element("legend", "Dictionary meanings", "sr-only"),
  );
  const order = detail.candidate_order || ex.candidates.map((c) => c.id);
  const candidates = order
    .map((id) => ex.candidates.find((c) => c.id === id))
    .filter(Boolean);
  candidates.forEach((candidate, i) => {
    const label = element("label", undefined, "candidate");
    const input = element("input");
    input.type = "checkbox";
    input.name = "candidate";
    input.value = candidate.id;
    input.addEventListener("change", () => {
      if (input.checked)
        document
          .querySelectorAll("input[name=outcome]")
          .forEach((r) => (r.checked = false));
      updateActions();
    });
    const body = element("span", undefined, "candidate-body");
    body.append(
      element("span", `Meaning ${i + 1} · ${candidate.id}`, "candidate-label"),
      element("span", candidate.definition, "definition"),
    );
    label.append(input, body);
    $("candidate-options").append(label);
  });
  restoreVote(automatic ? detail.review : detail.my_vote);
  if (automatic) {
    const proposal = proposalSelection(p);
    if (!detail.review) restoreVote(proposal);
    $("restore-initial").hidden = true;
    $("vote-status").textContent = detail.review
      ? `Human review saved · ${detail.review.status.replaceAll("_", " ")} · ${detail.review.reviewer}`
      : "Unreviewed automatic proposal. Only explicit human approval or correction enters the gold export.";
    renderAutomatic(p);
  }
  renderAssistance(detail.assistance || []);
  renderAdjudication();
  updateActions();
}
function restoreVote(vote) {
  document
    .querySelectorAll("input[name=candidate]")
    .forEach(
      (input) => (input.checked = (vote?.gold_ids || []).includes(input.value)),
    );
  document
    .querySelectorAll("input[name=outcome]")
    .forEach((input) => (input.checked = vote?.outcome === input.value));
  $("review-note").value = vote?.note || "";
  const initial = vote?.initial;
  $("restore-initial").hidden = !initial;
  if (vote) {
    $("vote-status").textContent = vote.ai_exposed
      ? "Independent first vote preserved. This revision is marked as AI-exposed."
      : "Your independent review is saved. You can revise it or ask for assistance.";
  } else {
    $("vote-status").textContent =
      "Your answer is private until adjudication. Save it before viewing AI assistance.";
  }
}
function selectedOutcome() {
  const ids = [
    ...document.querySelectorAll("input[name=candidate]:checked"),
  ].map((input) => input.value);
  const other = document.querySelector("input[name=outcome]:checked");
  return {
    outcome: ids.length ? "selected" : other?.value || "",
    gold_ids: ids,
  };
}
function updateActions() {
  for (const id of [
    "export-button",
    "export-all-button",
    "backup-button",
    "import-button",
    "empty-import",
    "load-demo",
    "empty-demo",
    "settings-button",
    "workspace-button",
    "random-audit",
  ]) {
    $(id).disabled = state.busy;
  }
  const vote = state.detail?.my_vote;
  const named = Boolean(state.reviewer.trim());
  $("save-vote").disabled = !named || state.busy;
  const automatic = Boolean(state.detail?.example?.provenance?.automatic);
  $("save-vote").textContent = automatic
    ? "Save human correction"
    : vote
      ? "Save revised review"
      : "Save independent review";
  $("approve-result").disabled =
    state.busy ||
    !named ||
    state.detail?.example?.provenance?.automatic?.status !== "labeled";
  $("reject-result").disabled = state.busy || !named;
  $("assist-button").disabled =
    !vote ||
    !state.credential.configured ||
    state.busy ||
    $("adjudication-mode").checked;
  $("refresh-assistance").disabled = $("assist-button").disabled;
  $("assistance-lock").hidden = Boolean(vote);
  if (vote && !state.credential.configured) {
    $("assistance-intro").textContent =
      "Your independent answer is saved. Add your API key in Jev settings to request a second opinion.";
  } else {
    $("assistance-intro").textContent = vote
      ? "Compare Jev’s judgment with your own. Your first answer remains on record."
      : "First, save your own reading. Then compare it with Jev’s typed decisions.";
  }
  $("previous-example").disabled =
    state.examples.findIndex((e) => e.id === state.current) <= 0;
  $("next-example").disabled =
    state.examples.findIndex((e) => e.id === state.current) >=
    state.examples.length - 1;
  $("finalize-button").disabled =
    state.busy ||
    !named ||
    !state.detail?.reviews ||
    state.detail.reviews.length < 2;
}
async function saveVote() {
  if (!state.detail) return;
  const selection = selectedOutcome();
  if (!selection.outcome)
    throw new Error("Choose at least one definition or a review outcome.");
  if (state.detail.example.provenance?.automatic)
    return saveAutomatic("correct", selection);
  await api("/api/vote", {
    id: state.current,
    ...selection,
    note: $("review-note").value,
    expected_revision: state.detail.revision,
  });
  await refreshQueue(state.current);
  notice("Review saved. Your independent first vote remains on record.");
}
function proposalSelection(p) {
  const automatic = p.automatic || {},
    result = automatic[automatic.selected_by] || {};
  const ids = result.selected_ids || p.proposed_gold_ids || [];
  const choice = result.outcome || result.choice;
  return {
    gold_ids: ids,
    outcome: ids.length
      ? "selected"
      : { none: "none_fits", insufficient: "insufficient_context" }[choice] ||
        choice,
  };
}
async function saveAutomatic(actionName, selection = selectedOutcome()) {
  await api("/api/review", {
    id: state.current,
    action: actionName,
    ...selection,
    note: $("review-note").value,
    expected_revision: state.detail.revision,
  });
  await refreshQueue(state.current);
  notice(
    `Human review saved · ${actionName === "approve" ? "approved" : actionName === "reject" ? "rejected" : "corrected"}. Automatic evidence is preserved.`,
  );
}
function renderAutomatic(p) {
  const node = $("automatic-evidence"),
    a = p.automatic;
  clear(node);
  node.append(
    element("span", "AUTOMATIC EVIDENCE · SILVER", "eyebrow"),
    element("h2", "Compare the judgments"),
    element("p", `Bucket: ${BUCKETS[a.bucket] || "Unresolved"}`),
    element(
      "p",
      `Selected by: ${{ jev: "Jev", codex: "Codex" }[a.selected_by] || "Unresolved"}`,
    ),
  );
  if (a.escalation_reason)
    node.append(
      element(
        "p",
        `Escalation: ${{ jev_error: "Jev request failed", jev_missing_confidence: "Jev returned no confidence", jev_low_confidence: "Jev confidence below threshold", low_confidence: "Jev confidence below threshold" }[a.escalation_reason] || a.escalation_reason.replaceAll("_", " ")}`,
      ),
    );
  for (const key of ["jev", "codex"]) {
    const result = a[key],
      section = element("section", undefined, "judgment");
    section.append(
      element("h3", key === "jev" ? "Jev judgment" : "Codex judgment"),
    );
    if (!result)
      section.append(
        element(
          "p",
          a[key + "_error"]
            ? `Error: ${typeof a[key + "_error"] === "string" ? a[key + "_error"] : `Request failed after ${a[key + "_error"].attempts || "unknown"} attempts`}`
            : "Not requested / no result",
        ),
      );
    else {
      section.append(
        element(
          "p",
          `Outcome: ${result.selected_ids?.length ? OUTCOMES.selected : OUTCOMES[result.outcome || result.choice] || { none: "None fits", insufficient: "Insufficient context" }[result.choice] || "Unknown"}`,
        ),
        element("p", `IDs: ${(result.selected_ids || []).join(", ") || "—"}`),
      );
      if (typeof result.confidence === "number")
        section.append(
          element(
            "p",
            `Jev confidence: ${(result.confidence * 100).toFixed(1)}% · threshold ${((a.threshold ?? 0.8) * 100).toFixed(0)}%`,
          ),
        );
      if (result.rationale) section.append(element("p", result.rationale));
      if (result.probabilities) {
        for (const [id, probability] of Object.entries(result.probabilities))
          section.append(
            element(
              "p",
              `${id}: ${(probability * 100).toFixed(1)}%`,
              "probability-label",
            ),
          );
        for (const [key, label] of [
          ["none_probability", "None fits"],
          ["insufficient_probability", "Insufficient context"],
        ]) {
          if (typeof result[key] === "number")
            section.append(
              element(
                "p",
                `${label}: ${(result[key] * 100).toFixed(1)}%`,
                "probability-label",
              ),
            );
        }
      }
    }
    node.append(section);
  }
  if (p.proposed_gold_ids)
    node.append(
      element(
        "p",
        `Original proposed IDs: ${p.proposed_gold_ids.join(", ") || "none fits"}`,
        "ai-meta",
      ),
    );
}
function renderAssistance(results) {
  clear($("ai-results"));
  $("ai-results").hidden = !results.length;
  $("refresh-assistance").hidden = !results.length;
  if (!results.length) return;
  const raw = results[results.length - 1];
  const result = raw.result || raw;
  const ex = state.detail.example;
  let heading = "Jev’s suggestion";
  if (result.selected_ids?.length)
    heading = ex.candidates
      .filter((c) => result.selected_ids.includes(c.id))
      .map((c) => c.definition)
      .join(" / ");
  else if (["none", "none_fits"].includes(result.choice))
    heading = "None of the definitions fits";
  else if (["insufficient", "insufficient_context"].includes(result.choice))
    heading = "Insufficient context";
  $("ai-results").append(element("p", heading, "ai-summary"));
  const rows = ex.candidates.map((c) => [
    c.definition,
    Number(result.probabilities?.[c.id] || 0),
  ]);
  rows.push(
    ["None fits", Number(result.none_probability || 0)],
    ["Insufficient context", Number(result.insufficient_probability || 0)],
  );
  rows.sort((a, b) => b[1] - a[1]);
  rows.forEach(([label, value]) => {
    const row = element("div", undefined, "probability-row");
    const line = element("div", undefined, "probability-label");
    line.append(
      element("span", label),
      element("strong", `${(value * 100).toFixed(1)}%`),
    );
    const track = element("div", undefined, "probability-track"),
      fill = element("div", undefined, "probability-fill");
    fill.style.width = `${Math.min(100, Math.max(0, value * 100))}%`;
    track.append(fill);
    row.append(line, track);
    $("ai-results").append(row);
  });
  const usage = result.usage || {};
  $("ai-results").append(
    element(
      "p",
      [
        `${result.provider || "Jev"} · ${result.model || "unknown model"}`,
        result.confidence !== null && result.confidence !== undefined
          ? `Provider confidence: ${(result.confidence * 100).toFixed(1)}%`
          : "",
        usage.input_tokens !== undefined
          ? `Tokens: ${usage.input_tokens} in / ${usage.output_tokens || 0} out`
          : "",
        result.created_at
          ? `Requested ${new Date(result.created_at).toLocaleString()}`
          : "",
        "These scores are suggestions, not verified certainty.",
      ]
        .filter(Boolean)
        .join("\n"),
      "ai-meta",
    ),
  );
}
async function askJev(refresh = false) {
  if (!state.detail?.my_vote)
    throw new Error("Save an independent review first.");
  $("assist-button").textContent = "Asking Jev…";
  try {
    await api("/api/assist", { id: state.current, refresh });
    await selectExample(state.current);
    await refreshStats();
    notice("Jev’s suggestion is recorded. Your original vote has not changed.");
  } finally {
    $("assist-button").textContent = "Get Jev’s suggestion";
  }
}
function renderAdjudication() {
  const active = $("adjudication-mode").checked;
  $("adjudication-panel").hidden = !active;
  if (!active) return;
  clear($("other-reviews"));
  const reviews = state.detail.reviews || [];
  reviews.forEach((review) => {
    const initial = review.initial || review;
    const node = element("div", undefined, "review-evidence");
    node.append(
      element("strong", review.reviewer || review.reviewer_id || "Reviewer"),
      element(
        "div",
        `First vote: ${OUTCOMES[initial.outcome] || initial.outcome || "Unknown"}${initial.gold_ids?.length ? " · " + initial.gold_ids.join(", ") : ""}`,
      ),
    );
    if (
      review.outcome !== initial.outcome ||
      JSON.stringify(review.gold_ids) !== JSON.stringify(initial.gold_ids)
    )
      node.append(
        element(
          "div",
          `Revised: ${OUTCOMES[review.outcome] || review.outcome} · ${(review.gold_ids || []).join(", ")}`,
        ),
      );
    if (review.note || initial.note)
      node.append(element("div", review.note || initial.note));
    node.append(
      element(
        "small",
        review.ai_exposed
          ? "AI assistance was viewed."
          : "No AI exposure recorded.",
      ),
    );
    $("other-reviews").append(node);
  });
  if (reviews.length < 2)
    $("other-reviews").append(
      element(
        "p",
        "Two different reviewers must first save independent votes before final adjudication.",
        "section-help",
      ),
    );
  const final = state.detail.adjudication;
  $("adjudication-status").textContent = final
    ? `Finalized: ${OUTCOMES[final.outcome] || final.outcome} · ${(final.gold_ids || []).join(", ")}${final.note ? " — " + final.note : ""}`
    : "No final adjudication yet.";
}
async function finalize() {
  const selection = selectedOutcome();
  if (!selection.outcome)
    throw new Error("Choose the final outcome in the form above.");
  await api("/api/adjudicate", {
    id: state.current,
    ...selection,
    note: $("review-note").value,
    expected_revision: state.detail.revision,
  });
  await refreshQueue(state.current);
  notice(
    "Adjudication saved. Only eligible adjudicated records enter the gold export.",
  );
}
function languageCounts(stats, language) {
  const source = stats.by_language || stats.languages || {};
  const keys = Object.keys(source).filter((key) =>
    language === "pt" ? key.startsWith("pt") : key.startsWith("en"),
  );
  let total = 0;
  keys.forEach((key) => {
    const row = source[key];
    if (typeof row === "number") return;
    total += Number(row.accepted ?? row.gold ?? row.exportable ?? 0);
  });
  return total;
}
async function refreshStats() {
  state.stats = await api("/api/stats");
  const audit = state.stats.automatic_review;
  const dashboard = $("automatic-dashboard");
  dashboard.hidden = !audit?.total;
  document.querySelector(".progress-strip").hidden = Boolean(audit?.total);
  clear(dashboard);
  if (audit?.total) {
    const top = element("div", undefined, "dashboard-top");
    top.append(
      element("span", "AUTOMATIC LABEL AUDIT", "eyebrow"),
      element("h2", `${audit.reviewed} / ${audit.total} reviewed`),
      element(
        "p",
        `${audit.accepted} human-approved · ${audit.unreviewed} awaiting review`,
      ),
    );
    const reportButton = element("button", "Workspace report ↗", "text-button");
    reportButton.addEventListener("click", report);
    top.append(reportButton);
    dashboard.append(top);
    const table = element("table");
    const header = element("tr");
    [
      "Jev confidence",
      "Reviewed",
      "Comparable",
      "Correct",
      "Observed accuracy",
    ].forEach((label) => header.append(element("th", label)));
    const head = element("thead"),
      body = element("tbody");
    head.append(header);
    table.append(head, body);
    for (const [band, counts] of Object.entries(
      audit.by_confidence || {},
    ).sort()) {
      const row = element("tr");
      [
        band,
        counts.reviewed,
        counts.comparable,
        counts.correct,
        counts.accuracy === null
          ? "—"
          : `${(counts.accuracy * 100).toFixed(1)}%`,
      ].forEach((value) => row.append(element("td", value)));
      body.append(row);
    }
    dashboard.append(table, element("p", audit.note, "ai-meta"));
  }
  for (const language of ["en", "pt"]) {
    const count = languageCounts(state.stats, language);
    $(language + "-count").textContent = `${count} / 100`;
    $(language + "-progress").value = Math.min(count, 100);
  }
}
function download(filename, contents, type = "application/json") {
  const blob = new Blob([contents], { type });
  const a = element("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
async function importExamples(examples, demo = false) {
  if (!state.reviewer) {
    $("reviewer-name").focus();
    throw new Error(
      "Enter a reviewer name first so the review queue can be opened.",
    );
  }
  if (!Array.isArray(examples) || !examples.length)
    throw new Error("The import contains no examples.");
  const result = await api("/api/import", { examples, demo });
  await refreshQueue();
  notice(
    `Imported ${examples.length} ${demo ? "demo " : ""}candidate records. ${demo ? "These are synthetic demonstrations, excluded from real gold exports." : examples.some((ex) => ex.provenance?.automatic) ? "Automatic proposals are visible and preserved separately from human-approved labels." : "Incoming gold labels are hidden; collect independent reviews."}`,
  );
  return result;
}
async function importFile(file) {
  if (!file) return;
  if (file.size > 8 * 1024 * 1024)
    throw new Error("Choose a JSONL file under 8 MB.");
  const text = await file.text();
  let examples;
  try {
    if (text.trim().startsWith("[")) examples = JSON.parse(text);
    else
      examples = text
        .split(/\r?\n/)
        .filter((line) => line.trim())
        .map((line, index) => {
          try {
            return JSON.parse(line);
          } catch {
            throw new Error(`Invalid JSON on line ${index + 1}.`);
          }
        });
  } catch (error) {
    throw new Error(
      error.message || "The file must contain valid JSONL examples.",
    );
  }
  await importExamples(examples);
}
async function exportGold(scope = "gold") {
  const result = await api("/api/export", { include_demo: false, scope });
  const examples = result.examples || [];
  download(
    scope === "all" ? "review-all-with-status.jsonl" : "pilot-gold.jsonl",
    result.jsonl ??
      examples.map((ex) => JSON.stringify(ex)).join("\n") +
        (examples.length ? "\n" : ""),
    "application/x-ndjson",
  );
  download(
    "pilot-manifest.json",
    JSON.stringify(result.manifest || {}, null, 2),
  );
  notice(
    scope === "all"
      ? `Exported ${examples.length} records with review status and preserved automatic evidence.`
      : `Exported ${examples.length} reviewed gold records and the manifest. Demo and unresolved records are excluded.`,
  );
}
function report() {
  clear($("report-content"));
  for (const [key, value] of Object.entries(state.stats)) {
    if (value === null || typeof value !== "object") {
      const row = element("div", undefined, "report-row");
      row.append(
        element("span", key.replaceAll("_", " ")),
        element("strong", value ?? "—"),
      );
      $("report-content").append(row);
    }
  }
  const details = element("details");
  details.append(
    element("summary", "Detailed progress, agreement and coverage"),
    element("pre", JSON.stringify(state.stats, null, 2)),
  );
  $("report-content").append(details);
  $("report-dialog").showModal();
}
function move(delta) {
  const index = state.examples.findIndex((e) => e.id === state.current);
  const next = state.examples[index + delta];
  if (next) action(() => selectExample(next.id));
}
function initEvents() {
  $("random-audit").addEventListener("click", () =>
    action(async () => {
      const query = new URLSearchParams({
        status: "unreviewed",
        bucket: "jev_high_confidence",
        language: $("language-filter").value,
        ...Object.fromEntries(
          ["difficulty", "area", "paper"].map((key) => [
            key,
            $(key + "-filter").value,
          ]),
        ),
      });
      const { examples } = await api("/api/examples?" + query);
      if (!examples.length)
        throw new Error(
          "No unreviewed high-confidence examples match the language, difficulty, area and paper filters.",
        );
      $("bucket-filter").value = "jev_high_confidence";
      $("status-filter").value = "unreviewed";
      $("confidence-filter").value = "";
      await refreshQueue(
        examples[Math.floor(Math.random() * examples.length)].id,
      );
      notice("Random unreviewed high-confidence example opened for audit.");
    }),
  );
  $("reviewer-form").addEventListener("submit", (event) => {
    event.preventDefault();
    action(async () => {
      const reviewer = $("reviewer-name").value.trim();
      if (!reviewer)
        throw new Error("Enter a reviewer name before saving a review.");
      await api("/api/session", { reviewer });
      await refreshSession();
      await refreshQueue();
      notice(
        `Reviewing as ${reviewer}. Other reviewers’ answers remain hidden outside adjudication.`,
      );
    });
  });
  [
    "language-filter",
    "status-filter",
    "bucket-filter",
    "difficulty-filter",
    "area-filter",
    "paper-filter",
    "confidence-filter",
  ].forEach((id) =>
    $(id).addEventListener("change", () => {
      if (state.busy) state.refreshRequested = true;
      else action(() => refreshQueue());
    }),
  );
  $("vote-form").addEventListener("submit", (event) => {
    event.preventDefault();
    action(saveVote);
  });
  document.querySelectorAll("input[name=outcome]").forEach((input) =>
    input.addEventListener("change", () => {
      if (input.checked)
        document
          .querySelectorAll("input[name=candidate]")
          .forEach((c) => (c.checked = false));
    }),
  );
  $("restore-initial").addEventListener("click", () => {
    if (state.detail?.my_vote?.initial) {
      restoreVote(state.detail.my_vote.initial);
      notice(
        "First answer restored in the form. Save a revised review to record this change.",
      );
    }
  });
  $("assist-button").addEventListener("click", () =>
    action(() => askJev(false)),
  );
  $("refresh-assistance").addEventListener("click", () =>
    action(() => askJev(true)),
  );
  $("settings-button").addEventListener("click", () => {
    $("api-key").value = "";
    renderCredentials();
    $("settings-dialog").showModal();
  });
  $("close-settings").addEventListener("click", () =>
    $("settings-dialog").close(),
  );
  $("settings-dialog").addEventListener(
    "close",
    () => ($("api-key").value = ""),
  );
  $("provider").addEventListener(
    "change",
    () =>
      ($("jev-model").value =
        $("provider").value === "openrouter"
          ? "typesafe/jev-1.13"
          : "jev-latest"),
  );
  $("settings-form").addEventListener("submit", (event) => {
    event.preventDefault();
    action(async () => {
      const key = $("api-key").value;
      $("api-key").value = "";
      if (!key.trim())
        throw new Error(
          "Enter an API key. It will only be held in session memory.",
        );
      await api("/api/key", {
        key,
        provider: $("provider").value,
        model: $("jev-model").value.trim(),
        save_to_config: $("save-to-config").checked,
      });
      await refreshSession();
      $("settings-dialog").close();
      notice(
        state.credential.persisted
          ? "API key saved to local owner-only config. No provider request has been made."
          : "API key saved for this session only. No provider request has been made.",
      );
    });
  });
  $("clear-key").addEventListener("click", () =>
    action(async () => {
      await api("/api/key", undefined, "DELETE");
      $("api-key").value = "";
      await refreshSession();
      notice("API key cleared from this session.");
    }),
  );
  [$("import-button"), $("empty-import")].forEach((button) =>
    button.addEventListener("click", () => $("import-file").click()),
  );
  $("import-file").addEventListener("change", (event) =>
    action(async () => {
      try {
        await importFile(event.target.files[0]);
      } finally {
        event.target.value = "";
      }
    }),
  );
  [$("load-demo"), $("empty-demo")].forEach((button) =>
    button.addEventListener("click", () =>
      action(() => importExamples(DEMO_EXAMPLES, true)),
    ),
  );
  $("adjudication-mode").addEventListener("change", () =>
    action(async () => {
      if (!state.current) return;
      try {
        await selectExample(state.current);
      } catch (error) {
        $("adjudication-mode").checked = false;
        throw error;
      }
    }),
  );
  $("finalize-button").addEventListener("click", () => action(finalize));
  $("previous-example").addEventListener("click", () => move(-1));
  $("next-example").addEventListener("click", () => move(1));
  $("export-button").addEventListener("click", () => action(exportGold));
  $("export-all-button").addEventListener("click", () =>
    action(() => exportGold("all")),
  );
  $("approve-result").addEventListener("click", () =>
    action(() => saveAutomatic("approve")),
  );
  $("reject-result").addEventListener("click", () =>
    action(() => saveAutomatic("reject")),
  );
  $("backup-button").addEventListener("click", () =>
    action(async () => {
      download(
        "pilot-workspace-backup.json",
        JSON.stringify(await api("/api/backup", {}), null, 2),
      );
      notice("Workspace backup downloaded. API keys are excluded.");
    }),
  );
  $("workspace-button").addEventListener("click", () =>
    action(async () => {
      await refreshStats();
      report();
    }),
  );
  $("close-report").addEventListener("click", () => $("report-dialog").close());
  $("download-report").addEventListener("click", () =>
    download("pilot-report.json", JSON.stringify(state.stats, null, 2)),
  );
  document.addEventListener("keydown", (event) => {
    if ($("settings-dialog").open || $("report-dialog").open) return;
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      action(saveVote);
      return;
    }
    if (
      ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)
    )
      return;
    if (event.key === "j") move(1);
    if (event.key === "k") move(-1);
  });
}
initEvents();
(async () => {
  try {
    await refreshSession();
    await refreshQueue();
  } catch (error) {
    notice(error.message, true);
  }
})();
