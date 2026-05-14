const state = {
  pureText: "",
  hybridText: "",
  contexts: [],
  history: [],
  historyLimit: 10,
  historyOffset: 0,
  historyTotal: 0,
  activeTab: "pure",
};

const el = (id) => document.getElementById(id);

function setStatus(msg) {
  el("status").textContent = msg;
}

function setBusy(busy) {
  ["pureBtn", "hybridBtn", "demoPresetBtn", "clearBtn"].forEach((id) => {
    const node = el(id);
    if (node) node.disabled = !!busy;
  });
}

function activateTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  el("purePanel").classList.toggle("active", tab === "pure");
  el("hybridPanel").classList.toggle("active", tab === "hybrid");
  el("contextsPanel").classList.toggle("active", tab === "contexts");
}

function payloadFromInputs(includeRetrieval) {
  return {
    prompt: el("prompt").value.trim(),
    max_new_tokens: Number(el("maxNewTokens").value),
    temperature: Number(el("temperature").value),
    top_k: Number(el("topK").value),
    ...(includeRetrieval ? { retrieval_top_k: Number(el("retrievalTopK").value) } : {}),
  };
}

function applyDemoPreset() {
  el("temperature").value = "0.4";
  el("topK").value = "10";
  el("maxNewTokens").value = "25";
  el("retrievalTopK").value = "2";
  setStatus("Applied High Coherence demo preset.");
}

async function callApi(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    let detail = data.detail;
    if (Array.isArray(detail)) {
      detail = detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
    }
    const msg = detail || data.message || JSON.stringify(data);
    throw new Error(`${path} failed (${res.status}): ${msg}`);
  }
  return data;
}

async function getJson(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return res.json();
}

async function del(path) {
  const res = await fetch(path, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return res.json();
}

function renderContexts() {
  const list = el("contextsList");
  list.innerHTML = "";
  if (!state.contexts.length) {
    const li = document.createElement("li");
    li.textContent = "No contexts returned.";
    list.appendChild(li);
    return;
  }
  state.contexts.forEach((c) => {
    const li = document.createElement("li");
    const score = typeof c.score === "number" ? ` (score: ${c.score.toFixed(4)})` : "";
    li.textContent = `${c.text || ""}${score}`;
    list.appendChild(li);
  });
}

async function runPure() {
  const body = payloadFromInputs(false);
  if (!body.prompt) {
    setStatus("Please enter a prompt.");
    return;
  }
  setBusy(true);
  setStatus("Running /generate...");
  try {
    const data = await callApi("/generate", body);
    state.pureText = data.generated_text || "";
    el("pureText").textContent = state.pureText;
    activateTab("pure");
    await loadHistory();
    const ms = typeof data.latency_ms === "number" ? `${data.latency_ms.toFixed(1)} ms` : "n/a";
    setStatus(`Pure generation complete (${ms}).`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

async function runHybrid() {
  const body = payloadFromInputs(true);
  if (!body.prompt) {
    setStatus("Please enter a prompt.");
    return;
  }
  setBusy(true);
  setStatus("Running /hybrid_generate...");
  try {
    const data = await callApi("/hybrid_generate", body);
    state.hybridText = data.generated_text || "";
    state.contexts = data.contexts || [];
    el("hybridText").textContent = state.hybridText;
    renderContexts();
    activateTab("hybrid");
    await loadHistory();
    const ms = typeof data.latency_ms === "number" ? `${data.latency_ms.toFixed(1)} ms` : "n/a";
    setStatus(`Hybrid generation complete (${ms}).`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

function clearAll() {
  el("prompt").value = "";
  state.pureText = "";
  state.hybridText = "";
  state.contexts = [];
  el("pureText").textContent = "";
  el("hybridText").textContent = "";
  renderContexts();
  setStatus("Cleared.");
}

function applyHistoryRecord(record) {
  el("prompt").value = record.prompt || "";
  if (record.mode === "pure") {
    state.pureText = record.generated_text || "";
    el("pureText").textContent = state.pureText;
    activateTab("pure");
  } else {
    state.hybridText = record.generated_text || "";
    state.contexts = record.contexts || [];
    el("hybridText").textContent = state.hybridText;
    renderContexts();
    activateTab("hybrid");
  }
  setStatus(`Loaded history #${record.id}.`);
}

function renderHistory() {
  const list = el("historyList");
  list.innerHTML = "";
  if (!state.history.length) {
    const li = document.createElement("li");
    li.textContent = "No history yet. Run generation to create records.";
    list.appendChild(li);
    return;
  }
  state.history.forEach((r) => {
    const li = document.createElement("li");
    li.className = "historyItem";
    li.innerHTML = `
      <div class="meta">#${r.id} | ${r.mode.toUpperCase()} | ${r.latency_ms.toFixed(1)} ms | ${r.created_at}</div>
      <p class="prompt"><strong>Prompt:</strong> ${r.prompt}</p>
      <p class="snippet"><strong>Output:</strong> ${(r.generated_text || "").slice(0, 220)}${(r.generated_text || "").length > 220 ? "..." : ""}</p>
      <div class="miniActions">
        <button class="btn ghost loadHistoryBtn">Load This Run</button>
        <button class="btn ghost danger deleteHistoryBtn">Delete</button>
      </div>
    `;
    li.querySelector(".loadHistoryBtn").addEventListener("click", () => applyHistoryRecord(r));
    li.querySelector(".deleteHistoryBtn").addEventListener("click", () => deleteRecord(r.id));
    list.appendChild(li);
  });
}

async function loadHistory() {
  try {
    const mode = el("historyMode").value;
    const q = el("historySearch").value.trim();
    const params = new URLSearchParams({ limit: String(state.historyLimit), offset: String(state.historyOffset) });
    if (mode) params.set("mode", mode);
    if (q) params.set("q", q);
    const payload = await getJson(`/history?${params.toString()}`);
    state.history = payload.items || [];
    state.historyTotal = payload.total || 0;
    renderHistory();
    const start = state.historyTotal === 0 ? 0 : state.historyOffset + 1;
    const end = Math.min(state.historyOffset + state.historyLimit, state.historyTotal);
    setStatus(`History ${start}-${end} of ${state.historyTotal}`);
  } catch (err) {
    setStatus(`History unavailable: ${err.message}`);
  }
}

async function deleteRecord(id) {
  try {
    await del(`/history/${id}`);
    await loadHistory();
    setStatus(`Deleted history #${id}.`);
  } catch (err) {
    setStatus(`Delete failed: ${err.message}`);
  }
}

async function clearHistory() {
  if (!window.confirm("Delete all history records?")) {
    return;
  }
  try {
    const out = await del("/history");
    state.historyOffset = 0;
    await loadHistory();
    setStatus(`Deleted ${out.count} records.`);
  } catch (err) {
    setStatus(`Delete all failed: ${err.message}`);
  }
}

function downloadFromUrl(url) {
  const a = document.createElement("a");
  a.href = url;
  a.target = "_blank";
  a.click();
}

function exportHistory(fmt) {
  const mode = el("historyMode").value;
  const q = el("historySearch").value.trim();
  const params = new URLSearchParams({ fmt });
  if (mode) params.set("mode", mode);
  if (q) params.set("q", q);
  downloadFromUrl(`/history/export?${params.toString()}`);
}

async function copyVisible() {
  const text =
    state.activeTab === "pure"
      ? state.pureText
      : state.activeTab === "hybrid"
      ? state.hybridText
      : JSON.stringify(state.contexts, null, 2);
  if (!text) {
    setStatus("Nothing to copy.");
    return;
  }
  await navigator.clipboard.writeText(text);
  setStatus("Copied current output.");
}

function downloadJson() {
  const payload = {
    prompt: el("prompt").value,
    pure_output: state.pureText,
    hybrid_output: state.hybridText,
    contexts: state.contexts,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "novalm_hybrid_output.json";
  a.click();
  URL.revokeObjectURL(url);
  setStatus("Downloaded output JSON.");
}

async function checkHealth() {
  const badge = el("healthBadge");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Health not ok");
    const data = await res.json().catch(() => ({}));
    const device = data.device ? ` (${String(data.device).toUpperCase()})` : "";
    badge.textContent = `API healthy${device}`;
    badge.classList.add("ok");
    badge.classList.remove("error");
  } catch (_) {
    badge.textContent = "API unavailable";
    badge.classList.add("error");
    badge.classList.remove("ok");
  }
}

function bind() {
  el("demoPromptSelect").addEventListener("change", (ev) => {
    const v = ev.target.value;
    if (v) {
      el("prompt").value = v;
      setStatus("Applied suggested prompt.");
      ev.target.value = "";
    }
  });
  el("demoPresetBtn").addEventListener("click", applyDemoPreset);
  el("pureBtn").addEventListener("click", runPure);
  el("hybridBtn").addEventListener("click", runHybrid);
  el("clearBtn").addEventListener("click", clearAll);
  el("copyBtn").addEventListener("click", copyVisible);
  el("downloadBtn").addEventListener("click", downloadJson);
  el("refreshHistoryBtn").addEventListener("click", loadHistory);
  el("applyHistoryFilterBtn").addEventListener("click", loadHistory);
  el("clearHistoryBtn").addEventListener("click", clearHistory);
  el("exportHistoryJsonBtn").addEventListener("click", () => exportHistory("json"));
  el("exportHistoryCsvBtn").addEventListener("click", () => exportHistory("csv"));
  el("historyPrevBtn").addEventListener("click", () => {
    state.historyOffset = Math.max(0, state.historyOffset - state.historyLimit);
    loadHistory();
  });
  el("historyNextBtn").addEventListener("click", () => {
    if (state.historyOffset + state.historyLimit < state.historyTotal) {
      state.historyOffset += state.historyLimit;
      loadHistory();
    }
  });
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => activateTab(b.dataset.tab)));
}

bind();
checkHealth();
renderContexts();
loadHistory();
