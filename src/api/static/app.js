const byId = (id) => document.getElementById(id);
const form = byId("diagnosis-form");
const equipment = byId("equipment");
const sensorRows = byId("sensor-rows");
const presetButtons = [...document.querySelectorAll("[data-preset]")];
const evidence = byId("evidence");
const statusLabels = {
  answered: ["Answered", "green"],
  insufficient_evidence: ["Insufficient evidence", "amber"],
  out_of_scope: ["Out of scope", "blue"],
  found: ["Found", "blue"],
  normal: ["Normal", "green"],
  warning: ["Warning", "amber"],
  critical: ["Critical", "red"],
  ambiguous: ["Ambiguous", "amber"],
  unknown: ["Unknown", "amber"],
};
const STORAGE_API_KEY = "diagagent.api_key";
const STORAGE_ADMIN_KEY = "diagagent.admin_api_key";
let health = null;
let meta = null;
let presets = {};
let authRequired = false;
let authPanelOpen = false;
let memoryKeys = { apiKey: "", adminKey: "" };
let building = false;
let diagnosing = false;
let refreshId = 0;
let evidenceId = 0;

function readSessionKey(storageKey, fallbackField) {
  try {
    return window.sessionStorage.getItem(storageKey) || "";
  } catch {
    return memoryKeys[fallbackField];
  }
}

function writeSessionKey(storageKey, fallbackField, value) {
  memoryKeys[fallbackField] = value;
  try {
    if (value) window.sessionStorage.setItem(storageKey, value);
    else window.sessionStorage.removeItem(storageKey);
  } catch {
    // Fallback to in-memory credentials when sessionStorage is unavailable.
  }
}

function getApiKey() {
  return readSessionKey(STORAGE_API_KEY, "apiKey");
}

function getAdminKey() {
  return readSessionKey(STORAGE_ADMIN_KEY, "adminKey");
}

function hasAnyKey() {
  return Boolean(getApiKey() || getAdminKey());
}

function authHeaders(path) {
  if (!path.startsWith("/v1/")) return {};
  const key = path === "/v1/ingest" ? (getAdminKey() || getApiKey()) : getApiKey();
  return key ? { "X-API-Key": key } : {};
}

function syncAuthUi() {
  const panel = byId("auth-panel");
  const toggle = byId("auth-toggle");
  const showPanel = authRequired || authPanelOpen;
  panel.hidden = !showPanel;
  toggle.hidden = !authRequired && !hasAnyKey() && !authPanelOpen;
  toggle.textContent = hasAnyKey() ? "API keys (set)" : "API keys";
}

function buildPresets(currentMeta) {
  const isZh = (currentMeta?.domain?.language || "").toLowerCase().startsWith("zh");
  const faultEntry = Object.entries(currentMeta?.fault_codes || {}).find(([, codes]) => codes.length);
  const fault = faultEntry
    ? {
        query: isZh
          ? `${faultEntry[1][0]} 故障码代表什么含义？应如何初步排查？`
          : `What does ${faultEntry[1][0]} mean?`,
        equipment_id: faultEntry[0],
        fault_code: faultEntry[1][0],
      }
    : null;
  const cache = fault
    ? { ...fault, query: fault.query.replace(/[?？]+$/, "") }
    : null;
  const firstSensor = (currentMeta?.sensors || [])[0];
  const critical = firstSensor
    ? {
        query: isZh
          ? `查询 ${firstSensor.equipment_id} 的 ${firstSensor.sensor} 严重告警阈值并判断当前读数。`
          : `Find the ${firstSensor.equipment_id} ${firstSensor.sensor} critical threshold.`,
        equipment_id: firstSensor.equipment_id,
        sensor_readings: { [firstSensor.sensor]: firstSensor.critical_threshold },
      }
    : null;
  const firstAsset = (currentMeta?.equipment_ids || [])[0];
  const missing = {
    query: firstAsset
      ? (isZh
          ? `${firstAsset} 的安装螺栓额定紧固力矩是多少？`
          : `What is the mounting bolt torque for ${firstAsset}?`)
      : (isZh
          ? "该设备的安装螺栓额定紧固力矩规范是多少？"
          : "What is the mounting bolt torque specification for this equipment?"),
  };
  const scope = {
    query: isZh ? "明天北京的天气预报是什么？" : "What is the Tokyo weather forecast for tomorrow?",
  };
  return { fault, critical, cache, missing, scope };
}

function element(tag, content = "", className = "") {
  const node = document.createElement(tag);
  node.textContent = content;
  if (className) node.className = className;
  return node;
}

function message(node, text, error = false) {
  node.textContent = text;
  node.hidden = !text;
  node.dataset.error = String(error);
}

async function requestJson(path, options = {}) {
  const headers = { ...authHeaders(path), ...(options.headers || {}) };
  const response = await fetch(path, { cache: "no-store", ...options, headers });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) {
    const detail = typeof data?.detail === "string"
      ? data.detail
      : data?.detail?.map?.((issue) => `${issue.loc.slice(1).join(".")}: ${issue.msg}`).join(" · ");
    const error = new Error(detail || `Request failed (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function setActions() {
  byId("build-index").disabled = building || diagnosing;
  byId("build-index").textContent = building
    ? "Building index…"
    : health?.status === "ok" ? "Rebuild index" : "Build index";
  byId("run-diagnosis").disabled = health?.status !== "ok" || authRequired || building || diagnosing;
}

function renderPresets() {
  presets = buildPresets(meta);
  presetButtons.forEach((button) => {
    const available = Boolean(presets[button.dataset.preset]);
    button.hidden = !available;
    button.disabled = !available;
    if (!available) button.setAttribute("aria-pressed", "false");
  });
}

function renderStatus() {
  const ready = health?.status === "ok";
  const mode = byId("mode");
  mode.textContent = meta
    ? (meta.mode === "mock" ? "Mock" : "Live")
    : authRequired ? "Auth required" : health ? "Connected" : "Offline";
  mode.title = meta ? `${meta.llm_model} · ${meta.embedding_model} · ${meta.reranker_model}` : "";
  if (meta) byId("domain").textContent = meta.domain.name;
  const status = byId("index-status");
  status.dataset.status = health?.status || "error";
  status.textContent = ready ? "Index ready" : health ? "Index needed" : "API unavailable";
  byId("chunk-count").textContent = ready ? `${health.chunk_count} chunks` : "";
  const version = byId("copy-version");
  version.hidden = !health?.index_version;
  version.textContent = health?.index_version ? `${health.index_version.slice(0, 12)}… ⧉` : "";
  version.title = health?.index_version || "";
  version.setAttribute("aria-label", "Copy index version");
  byId("empty-hint").textContent = authRequired
    ? "Enter an API key above to load equipment metadata and run diagnoses."
    : ready
      ? "Choose a preset or enter a question to inspect results and evidence."
      : "Build the index to enable diagnosis.";
  renderPresets();
  syncAuthUi();
  setActions();
  updateSuggestions();
}

async function refreshStatus() {
  const id = ++refreshId;
  try {
    const nextHealth = await requestJson("/health");
    if (id !== refreshId) return;
    health = nextHealth;
  } catch {
    if (id !== refreshId) return;
    health = null;
    meta = null;
    authRequired = false;
    renderStatus();
    return;
  }
  try {
    const nextMeta = await requestJson("/v1/meta");
    if (id !== refreshId) return;
    meta = nextMeta;
    if (authRequired) message(byId("system-notice"), "");
    authRequired = false;
  } catch (error) {
    if (id !== refreshId) return;
    meta = null;
    if (error.status === 401 || error.status === 403) {
      authRequired = true;
      message(byId("system-notice"), "API key required to access diagnostic endpoints.", true);
    }
  }
  renderStatus();
}

function setOptions(listId, entries) {
  byId(listId).replaceChildren(...entries.map(([value, label]) => {
    const option = element("option");
    option.value = value;
    if (label) option.label = label;
    return option;
  }));
}

function matchingSensorRules() {
  const asset = equipment.value.trim().toLowerCase();
  return (meta?.sensors || []).filter((rule) => !asset || rule.equipment_id.toLowerCase() === asset);
}

function updateSensorHint(row, rules = matchingSensorRules()) {
  const name = row.querySelector(".sensor-name").value.trim().toLowerCase();
  const hint = row.querySelector(".sensor-hint");
  const units = [...new Set(rules.filter((rule) => rule.sensor.toLowerCase() === name).map((rule) => rule.unit))];
  hint.textContent = units.length ? `Expected unit: ${units.join(" / ")}` : "";
  hint.hidden = !units.length;
}

function updateSuggestions() {
  const asset = equipment.value.trim().toLowerCase();
  const equipmentIds = meta?.equipment_ids || [];
  const faults = Object.entries(meta?.fault_codes || {});
  const matchingFaults = faults.filter(([id]) => !asset || id.toLowerCase() === asset);
  const faultCodes = [...new Set(matchingFaults.flatMap(([, codes]) => codes))];
  equipment.placeholder = equipmentIds[0] ? `e.g. ${equipmentIds[0]}` : "Optional equipment ID";
  byId("fault-code").placeholder = faultCodes[0] ? `e.g. ${faultCodes[0]}` : "Optional fault code";
  setOptions("equipment-options", equipmentIds.map((id) => [id]));
  setOptions("fault-options", faultCodes.map((code) => [code]));
  const rules = matchingSensorRules();
  setOptions("sensor-options", [...new Map(rules.map((rule) => [rule.sensor, rule.unit]))]);
  sensorRows.querySelectorAll(".sensor-row").forEach((row) => updateSensorHint(row, rules));
}

function addSensor(name = "", value = "") {
  const row = element("div", "", "sensor-row");
  const sensor = element("input", "", "sensor-name");
  sensor.setAttribute("aria-label", "Sensor name");
  sensor.setAttribute("list", "sensor-options");
  sensor.placeholder = "Sensor name";
  sensor.value = name;
  const reading = element("input", "", "sensor-value");
  reading.setAttribute("aria-label", "Sensor value");
  reading.type = "number";
  reading.step = "any";
  reading.placeholder = "Value";
  reading.value = value;
  const remove = element("button", "×", "button secondary remove-sensor");
  remove.type = "button";
  remove.setAttribute("aria-label", "Remove sensor reading");
  remove.addEventListener("click", () => {
    row.remove();
    byId("add-sensor").focus();
  });
  const hint = element("div", "", "hint sensor-hint");
  row.append(sensor, reading, remove, hint);
  sensorRows.append(row);
  sensor.addEventListener("input", () => updateSensorHint(row));
  if (!name) sensor.focus();
  updateSensorHint(row);
}

function loadPreset(name) {
  const preset = presets[name];
  if (!preset) return;
  byId("query").value = preset.query;
  equipment.value = preset.equipment_id || "";
  byId("fault-code").value = preset.fault_code || "";
  byId("bypass-cache").checked = false;
  byId("include-trace").checked = true;
  sensorRows.replaceChildren();
  Object.entries(preset.sensor_readings || {}).forEach(([sensor, value]) => addSensor(sensor, value));
  presetButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.preset === name)));
  message(byId("preset-note"), name === "cache" ? "Run Fault code first to warm this cache entry." : "");
  message(byId("form-error"), "");
  updateSuggestions();
}

function diagnosisRequest() {
  const query = byId("query").value.trim();
  if (!query) throw new Error("Enter a diagnostic question.");
  const request = {
    query,
    bypass_cache: byId("bypass-cache").checked,
    include_trace: byId("include-trace").checked,
    sensor_readings: Object.create(null),
  };
  const faultCode = byId("fault-code").value.trim();
  if (equipment.value.trim()) request.equipment_id = equipment.value.trim();
  if (faultCode) request.fault_code = faultCode;
  const labels = new Set();
  for (const row of sensorRows.querySelectorAll(".sensor-row")) {
    const name = row.querySelector(".sensor-name").value.trim();
    const reading = row.querySelector(".sensor-value");
    if (!name && !reading.value) continue;
    if (!name || !Number.isFinite(reading.valueAsNumber)) {
      throw new Error("Enter a sensor name and a numeric reading for each row.");
    }
    const key = name.toLowerCase();
    if (labels.has(key)) throw new Error("Use each sensor name once.");
    labels.add(key);
    request.sensor_readings[name] = reading.valueAsNumber;
  }
  return request;
}

function definitionList(node, definitions) {
  node.replaceChildren(...definitions.flatMap(([label, value]) => [element("dt", label), element("dd", value)]));
}

function statusBadge(node, status) {
  const [label, tone] = statusLabels[status] || [status, ""];
  node.textContent = label;
  node.dataset.tone = tone;
  node.dataset.status = status;
}

async function showEvidence(chunkId) {
  const id = ++evidenceId;
  byId("evidence-detail").hidden = true;
  byId("evidence-state").textContent = "Loading the original passage…";
  evidence.showModal();
  try {
    const chunk = await requestJson(`/v1/chunks?chunk_id=${encodeURIComponent(chunkId)}`);
    if (id !== evidenceId || !evidence.open) return;
    byId("evidence-section").textContent = chunk.section_title;
    definitionList(byId("evidence-meta"), [
      ["Source", chunk.source_path],
      ["Page", chunk.page_number ?? "Unpaginated"],
      ["Chunk", chunk.chunk_id],
    ]);
    byId("evidence-text").textContent = chunk.text;
    byId("evidence-state").textContent = "";
    byId("evidence-detail").hidden = false;
  } catch (error) {
    if (id === evidenceId && evidence.open) byId("evidence-state").textContent = error.message;
  }
}

function renderAnswer(text, citations) {
  const ids = new Set(citations.map((citation) => citation.chunk_id));
  const fragment = document.createDocumentFragment();
  let offset = 0;
  for (const match of text.matchAll(/\[([^\[\]\r\n]+)\]/g)) {
    if (!ids.has(match[1])) continue;
    fragment.append(text.slice(offset, match.index));
    const citation = element("button", match[0], "citation");
    citation.type = "button";
    citation.addEventListener("click", () => showEvidence(match[1]));
    fragment.append(citation);
    offset = match.index + match[0].length;
  }
  fragment.append(text.slice(offset));
  byId("answer").replaceChildren(fragment);
}

function renderTool(finding) {
  const card = element("article", "", "finding");
  const heading = element("header");
  const badge = element("span", "", "badge");
  statusBadge(badge, finding.status);
  heading.append(element("strong", finding.sensor || finding.fault_code), badge);
  card.append(heading);
  const rule = finding.rule;
  const fields = [];
  if (finding.kind === "sensor_threshold") {
    const unit = rule ? ` ${rule.unit}` : "";
    fields.push(["Reading", `${finding.reading}${unit}`]);
    if (rule) {
      const compare = rule.direction === "high" ? "≥" : "≤";
      fields.push(["Warning", `${compare} ${rule.warning_threshold}${unit}`]);
      fields.push(["Critical", `${compare} ${rule.critical_threshold}${unit}`]);
    }
  } else if (rule) {
    fields.push(["Meaning", rule.meaning], ["Severity", rule.severity], ["First check", rule.first_check]);
  }
  if (rule) fields.push(["Equipment", rule.equipment_id]);
  if (fields.length) {
    const details = element("dl");
    definitionList(details, fields);
    card.append(details);
  }
  return card;
}

function renderTrace(trace, requested, cached) {
  const available = requested && trace?.nodes?.length;
  byId("trace-section").hidden = !available && !(requested && cached);
  byId("trace-note").hidden = Boolean(available);
  byId("trace-note").textContent = available ? "" : "Cache hit. Bypass the cache to run a fresh workflow trace.";
  const nodes = available ? trace.nodes : [];
  const topKs = nodes.filter((node) => node.node === "retriever" && node.top_k).map((node) => node.top_k);
  const badges = available ? [element("span", `Retries ${trace.retry_count}`, "badge")] : [];
  if (topKs.length) badges.push(element("span", `Top K ${topKs.join(" → ")}`, "badge"));
  byId("trace-summary").replaceChildren(...badges);
  const max = Math.max(1, ...nodes.map((node) => node.latency_ms));
  byId("trace-nodes").replaceChildren(...nodes.map((node) => {
    const row = element("li", "", "trace-node");
    const latency = `${node.latency_ms.toFixed(1)} ms`;
    const progress = element("progress");
    progress.max = max;
    progress.value = node.latency_ms;
    progress.setAttribute("aria-label", `${node.node} latency`);
    const label = node.node === "retriever" && node.top_k ? `${node.node} · K=${node.top_k}` : node.node;
    row.append(element("span", label, "trace-label"), progress, element("span", latency));
    return row;
  }));
}

function renderResult(data, request, elapsed) {
  byId("empty-result").hidden = true;
  byId("result").hidden = false;
  statusBadge(byId("diagnosis-status"), data.status);
  byId("intent").textContent = (data.intent || "").replaceAll("_", " ");
  const cache = byId("cache-state");
  cache.textContent = data.cache_hit ? "Cache hit" : request.bypass_cache ? "Cache bypassed" : "Cache miss";
  cache.dataset.tone = data.cache_hit ? "blue" : "";
  byId("elapsed").textContent = `${Math.round(elapsed).toLocaleString()} ms total`;
  renderAnswer(data.answer, data.citations || []);
  const findings = data.tool_findings || [];
  byId("tool-section").hidden = !findings.length;
  byId("tool-findings").replaceChildren(...findings.map(renderTool));
  const missing = data.missing_aspects || [];
  byId("refusal-section").hidden = !missing.length && !data.escalation_recommendation;
  byId("missing-aspects").hidden = !missing.length;
  byId("missing-aspects").replaceChildren(...missing.map((item) => element("li", item)));
  message(byId("escalation"), data.escalation_recommendation || "");
  renderTrace(data.trace, request.include_trace, data.cache_hit);
}

equipment.addEventListener("input", updateSuggestions);
byId("add-sensor").addEventListener("click", () => addSensor());
presetButtons.forEach((button) => button.addEventListener("click", () => loadPreset(button.dataset.preset)));

byId("copy-version").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(health.index_version);
    message(byId("system-notice"), "Index version copied.");
  } catch {
    message(byId("system-notice"), `Index version: ${health?.index_version || "unavailable"}`);
  }
});

byId("auth-toggle").addEventListener("click", () => {
  authPanelOpen = !authPanelOpen;
  syncAuthUi();
});

byId("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  writeSessionKey(STORAGE_API_KEY, "apiKey", byId("api-key").value.trim());
  writeSessionKey(STORAGE_ADMIN_KEY, "adminKey", byId("admin-key").value.trim());
  authPanelOpen = false;
  message(byId("system-notice"), "");
  await refreshStatus();
});

byId("clear-keys").addEventListener("click", async () => {
  byId("api-key").value = "";
  byId("admin-key").value = "";
  writeSessionKey(STORAGE_API_KEY, "apiKey", "");
  writeSessionKey(STORAGE_ADMIN_KEY, "adminKey", "");
  authPanelOpen = false;
  message(byId("system-notice"), "API credentials cleared for this session.");
  await refreshStatus();
});

byId("build-index").addEventListener("click", async () => {
  building = true;
  setActions();
  message(byId("system-notice"), "Building the search index…");
  try {
    const index = await requestJson("/v1/ingest", { method: "POST" });
    message(byId("system-notice"), `Index ready with ${index.chunk_count} chunks.`);
    if (health?.index_version && health.index_version !== index.index_version) {
      byId("result").hidden = true;
      byId("empty-result").hidden = false;
      byId("result-state").textContent = "";
    }
  } catch (error) {
    if (error.status === 401 || error.status === 403) {
      authPanelOpen = true;
      syncAuthUi();
      message(
        byId("system-notice"),
        error.status === 403
          ? "Admin API key required for index rebuild."
          : "API key required for index rebuild.",
        true,
      );
    } else {
      message(byId("system-notice"), error.message, true);
    }
  } finally {
    await refreshStatus();
    building = false;
    setActions();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (diagnosing || building || health?.status !== "ok" || authRequired) return;
  const panel = document.querySelector(".result-panel");
  const resultState = byId("result-state");
  message(byId("form-error"), "");
  try {
    const request = diagnosisRequest();
    diagnosing = true;
    setActions();
    panel.setAttribute("aria-busy", "true");
    resultState.textContent = "Running diagnosis…";
    const start = performance.now();
    const data = await requestJson("/v1/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    renderResult(data, request, performance.now() - start);
    resultState.textContent = "Diagnosis complete";
  } catch (error) {
    if (error.status === 401 || error.status === 403) {
      authRequired = error.status === 401;
      authPanelOpen = true;
      syncAuthUi();
    }
    message(byId("form-error"), error.message, true);
    resultState.textContent = "";
  } finally {
    diagnosing = false;
    panel.setAttribute("aria-busy", "false");
    setActions();
  }
});

byId("api-key").value = getApiKey();
byId("admin-key").value = getAdminKey();
refreshStatus();
setInterval(() => { if (!building) refreshStatus(); }, 15000);
