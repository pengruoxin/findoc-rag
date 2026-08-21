"use strict";

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
const short = (value, length = 12) => value ? String(value).slice(0, length) : "—";
const milliseconds = (value) => value == null ? "" : `${Math.round(Number(value))} ms`;

const state = {
  apiKey: "",
  taskType: "query",
  indexId: "",
  capabilities: null,
  manifest: null,
  citations: [],
  claimCitations: [],
  resolvedEvidence: new Map(),
  evidenceStatus: new Map(),
  currentTrace: null,
  currentTaskId: "",
  selectedFile: null,
  uploadJobId: "",
  uploadTimer: null,
};

async function api(path, options = {}, useKey = false) {
  const headers = new Headers(options.headers || {});
  if (useKey && state.apiKey) headers.set("X-DeepSeek-API-Key", state.apiKey);
  const response = await fetch(path, { ...options, headers });
  let payload = null;
  try { payload = await response.json(); } catch (_) { /* empty or non-JSON response */ }
  if (!response.ok) {
    const error = new Error(payload?.error?.message || payload?.detail || `请求失败（${response.status}）`);
    error.code = payload?.error?.code || "";
    error.status = response.status;
    error.details = payload?.error?.details || null;
    throw error;
  }
  return payload;
}

function postJSON(path, body, useKey = false) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, useKey);
}

let toastTimer = null;
function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("show"), 2200);
}

function showNotice(message, kind = "error") {
  const element = $("notice");
  element.textContent = message;
  element.className = `notice ${kind}`;
  element.hidden = false;
}
function clearNotice() { $("notice").hidden = true; }

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${label}已复制`);
  } catch (_) {
    toast("复制失败，请手动选择");
  }
}

function setHealth(status, label) {
  $("healthDot").className = `status-dot ${status}`;
  $("healthText").textContent = label;
}

function openDrawer(panel = "model") {
  $("drawer").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
  $("drawerScrim").classList.add("open");
  selectPanel(panel);
}

function closeDrawer() {
  $("drawer").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  $("drawerScrim").classList.remove("open");
}

const PANEL_TITLES = {
  model: "模型配置", upload: "上传文档", settings: "任务设置", reviews: "人工审核", runtime: "服务状态",
};
function selectPanel(panel) {
  document.querySelectorAll("[data-panel]").forEach((button) => {
    button.classList.toggle("active", button.dataset.panel === panel);
  });
  document.querySelectorAll("[data-panel-content]").forEach((content) => {
    content.classList.toggle("active", content.dataset.panelContent === panel);
  });
  $("drawerTitle").textContent = PANEL_TITLES[panel] || "工作台设置";
  if (panel === "reviews") loadReviews();
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("findoc-theme", theme); } catch (_) { /* private browsing */ }
}
try {
  const savedTheme = localStorage.getItem("findoc-theme");
  if (savedTheme === "dark" || savedTheme === "light") applyTheme(savedTheme);
} catch (_) { /* private browsing */ }

function updateKeyState() {
  const configured = Boolean(state.apiKey);
  $("keyStatus").textContent = configured ? "DeepSeek 已就绪" : "配置 DeepSeek Key";
  $("keyButton").classList.toggle("configured", configured);
  $("clearKeyButton").hidden = !configured;
  updateTaskContext();
}

function saveKey() {
  const value = $("apiKeyInput").value.trim();
  if (!value) {
    showNotice("请输入 DeepSeek API Key。", "error");
    $("apiKeyInput").focus();
    return;
  }
  state.apiKey = value;
  $("apiKeyInput").value = "";
  $("apiKeyInput").type = "password";
  $("toggleKeyButton").textContent = "显示";
  updateKeyState();
  clearNotice();
  toast("Key 已在当前标签页中启用");
  closeDrawer();
}

function clearKey() {
  state.apiKey = "";
  $("apiKeyInput").value = "";
  updateKeyState();
  toast("当前 Key 已清除");
}

async function refreshStatus() {
  setHealth("busy", "正在连接");
  const results = await Promise.allSettled([
    api("/health/ready"), api("/v1/capabilities"), api("/v1/index"),
  ]);
  if (results[0].status === "fulfilled") {
    state.indexId = results[0].value.index_id || "";
    setHealth("ready", "服务正常");
  } else {
    state.indexId = "";
    setHealth("error", "服务未连接");
  }
  state.capabilities = results[1].status === "fulfilled" ? results[1].value : null;
  state.manifest = results[2].status === "fulfilled" ? results[2].value : null;
  renderCapabilities();
}

function renderCapabilities() {
  const capabilities = state.capabilities;
  const modes = capabilities?.modes || ["lexical"];
  const currentMode = $("retrievalMode").value;
  const names = { lexical: "关键词检索", dense: "向量检索", hybrid: "混合检索" };
  $("retrievalMode").innerHTML = modes.map((mode) => `<option value="${esc(mode)}">${esc(names[mode] || mode)}</option>`).join("");
  if (modes.includes(currentMode)) $("retrievalMode").value = currentMode;
  $("modeHint").textContent = modes.length > 1 ? "当前索引支持多种检索模式。" : "当前索引只支持关键词检索。";
  $("scopeToggle").disabled = !capabilities?.features?.scope;
  if (!capabilities?.features?.scope) $("scopeToggle").checked = false;

  const featureLabels = {
    ingestion_jobs: "PDF 上传", agent_tasks: "Agent 任务", human_reviews: "人工审核",
    request_api_keys: "标签页 Key", structured_table_artifacts: "结构化表格",
  };
  const featureText = Object.entries(featureLabels).map(([key, label]) => {
    const on = capabilities?.features?.[key];
    return `${label} ${on ? "可用" : "未启用"}`;
  });
  const manifest = state.manifest || {};
  $("runtimeGrid").innerHTML = [
    ["服务", state.indexId ? "已就绪" : "未连接"],
    ["当前索引", short(state.indexId, 16)],
    ["证据块", manifest.chunk_count ?? "—"],
    ["文档数", manifest.document_ids?.length ?? "—"],
    ["检索模式", modes.join("、")],
    ["网页能力", featureText.join(" · ")],
  ].map(([label, value]) => `<div class="runtime-item"><small>${esc(label)}</small><b>${esc(value)}</b></div>`).join("");
  $("indexDetails").textContent = state.indexId
    ? `index_id: ${state.indexId}\nsource SHA-256: ${manifest.source_chunk_sha256 || "—"}\ntokenizer: ${manifest.tokenizer || "—"}\nstructured tables: ${manifest.structured_table_count ?? 0}`
    : "服务未连接。";
}

const TASK_COPY = {
  query: { button: "查找答案", mode: "本地检索 + 证据门禁", verifier: "无需第二 Agent" },
  compare: { button: "运行对比 Agent", mode: "主 Agent + 工具调用", verifier: "本地证据门禁" },
  extract: { button: "运行抽取 Agent", mode: "主 Agent + 结构工具", verifier: "高风险时启用第二 Agent" },
  calculate: { button: "运行计算 Agent", mode: "主 Agent + Decimal 计算", verifier: "本地证据门禁" },
};
function selectTask(task) {
  state.taskType = task;
  document.querySelectorAll("[data-task]").forEach((button) => {
    if (!button.classList.contains("task-option")) return;
    const active = button.dataset.task === task;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  updateTaskContext();
}

function updateTaskContext() {
  const copy = TASK_COPY[state.taskType];
  $("runButtonText").textContent = copy.button;
  $("agentModeText").textContent = copy.mode;
  $("verifierModeText").textContent = copy.verifier;
  if (state.taskType !== "query" && !state.apiKey) {
    $("verifierModeText").textContent = "需要 DeepSeek Key";
  }
}

function retrievalPayload(query) {
  const payload = { query };
  const mode = $("retrievalMode").value;
  if (mode) payload.mode = mode;
  const topK = Number($("topKInput").value);
  if (Number.isFinite(topK) && topK > 0) payload.top_k = topK;
  const candidateK = Number($("candidateKInput").value);
  if ($("candidateKInput").value.trim() && Number.isFinite(candidateK)) payload.candidate_k = candidateK;
  if ($("scopeToggle").checked) payload.scope_routing = true;
  if ($("adaptiveToggle").checked) payload.adaptive_candidate_budget = true;
  return payload;
}

function setRunning() {
  clearNotice();
  $("runButton").disabled = true;
  $("outcomeBadge").className = "outcome-badge running";
  $("outcomeBadge").textContent = "分析中";
  $("resultTitle").textContent = state.taskType === "query" ? "正在查找可核验证据" : "Agent 正在执行任务";
  $("answerContent").className = "answer-content";
  $("answerContent").innerHTML = '<div class="skeleton" style="height:22px;width:78%;margin-bottom:12px"></div><div class="skeleton" style="height:16px;width:94%;margin-bottom:9px"></div><div class="skeleton" style="height:16px;width:65%"></div>';
  $("resultSummary").hidden = true;
  $("evidenceCount").textContent = "读取中";
  $("evidenceList").innerHTML = '<div class="evidence-card"><div style="padding:18px"><div class="skeleton" style="height:18px;width:52%;margin-bottom:13px"></div><div class="skeleton" style="height:70px;width:100%"></div></div></div>';
  $("processList").innerHTML = '<div class="process-step"><span class="step-mark">1</span><div><b>理解任务并规划</b><p>正在确定文档范围、指标口径和所需工具。</p></div></div>';
  setHealth("busy", "任务运行中");
}

function setFailure(error) {
  $("outcomeBadge").className = "outcome-badge error";
  $("outcomeBadge").textContent = "失败";
  $("resultTitle").textContent = "任务未完成";
  $("answerContent").className = "answer-content plain-answer";
  $("answerContent").textContent = error.message || "未知错误";
  $("evidenceCount").textContent = "0 条";
  $("evidenceList").innerHTML = evidenceEmpty("没有可展示的证据", "请检查服务、Key、索引或任务设置后重试。");
  showNotice(`${error.code ? `${error.code}：` : ""}${error.message}`, "error");
  if (error.code === "provider_key_required") openDrawer("model");
}

async function runTask() {
  const query = $("queryInput").value.trim();
  if (!query) { showNotice("请先输入问题。", "error"); $("queryInput").focus(); return; }
  if (state.taskType !== "query" && !state.apiKey) {
    showNotice("Agent 任务需要 DeepSeek API Key。", "info");
    openDrawer("model");
    return;
  }
  setRunning();
  try {
    if (state.taskType === "query") {
      const response = await postJSON("/v1/query", retrievalPayload(query), true);
      renderQueryResponse(response);
    } else {
      const settings = retrievalPayload(query);
      const response = await postJSON("/v1/agent/tasks", {
        task_type: state.taskType,
        query,
        mode: settings.mode || "lexical",
        top_k: Math.min(Number(settings.top_k || 3), 10),
        max_rounds: state.taskType === "extract" ? 4 : 3,
        max_tool_calls: 8,
        verifier_policy: $("verifierPolicy").value,
        verifier_support_proof: true,
      }, true);
      renderAgentResponse(response);
    }
  } catch (error) {
    setFailure(error);
  } finally {
    $("runButton").disabled = false;
    refreshStatus();
  }
}

const OUTCOME = {
  answer: ["已回答", "answer"], abstain: ["安全拒答", "abstain"],
  clarify: ["需要补充", "abstain"], evidence_only: ["仅返回证据", "abstain"],
};
function renderCommonAnswer(data, context = {}) {
  state.citations = data.citations || [];
  state.claimCitations = data.claim_citations || [];
  state.resolvedEvidence.clear();
  state.evidenceStatus.clear();
  state.currentTaskId = context.taskId || data.trace_id || "";
  const [label, className] = OUTCOME[data.outcome] || ["已完成", "answer"];
  $("outcomeBadge").className = `outcome-badge ${className}`;
  $("outcomeBadge").textContent = label;
  $("resultTitle").textContent = context.title || (data.outcome === "answer" ? "基于证据的结论" : "系统没有生成未经证明的结论");
  $("copyTraceButton").hidden = !state.currentTaskId;

  const validOrdinals = new Set(state.citations.map((citation) => Number(citation.ordinal)));
  const answer = $("answerContent");
  const claims = state.claimCitations;
  if (claims.length) {
    answer.className = "answer-content";
    answer.innerHTML = claims.map((claim, index) => {
      const links = (claim.citation_ordinals || []).map((ordinal) => citationButton(ordinal, validOrdinals.has(Number(ordinal)))).join("");
      return `<p class="claim-row" data-claim="${index}">${esc(claim.claim)} ${links}</p>`;
    }).join("");
  } else {
    answer.className = "answer-content plain-answer";
    answer.innerHTML = linkifyCitations(esc(data.answer || "服务没有返回回答。"), validOrdinals);
  }

  const summary = [
    ["任务", context.taskLabel || "快速问答"],
    ["证据", `${state.citations.length} 条`],
    ["生成方式", context.provider || providerName(data.provider)],
    ...(context.summary || []),
  ];
  $("resultSummary").innerHTML = summary.map(([key, value]) => `<span class="summary-chip">${esc(key)}：<b>${esc(value)}</b></span>`).join("");
  $("resultSummary").hidden = false;
  renderEvidence();
  if (state.citations.length) verifyEvidence();
}

function citationButton(ordinal, valid = true) {
  return valid
    ? `<button class="citation-link" type="button" data-citation="${esc(ordinal)}" aria-label="查看第 ${esc(ordinal)} 条证据">${esc(ordinal)}</button>`
    : `<span class="citation-link" title="引用不存在">${esc(ordinal)}</span>`;
}
function linkifyCitations(html, validOrdinals) {
  return html.replace(/\[(\d+)\]/g, (_, raw) => citationButton(Number(raw), validOrdinals.has(Number(raw))));
}
function providerName(provider) {
  const names = {
    "deterministic-table": "本地表格抽取", "openai-compatible": "DeepSeek",
    clarification: "澄清门禁", abstention: "证据门禁", "guardrail-abstention": "安全门禁",
  };
  return names[provider] || provider || "本地执行";
}

function renderQueryResponse(response) {
  state.currentTrace = response;
  renderCommonAnswer(response, {
    taskLabel: "快速问答",
    title: response.outcome === "answer" ? "基于文档证据的回答" : "证据不足，系统已停止生成",
    summary: [
      ["查询处理", `${response.rewrite_mode || "none"} → ${response.rewrite_gate || "none"}`],
      ["索引", short(response.index_id)],
    ],
  });
  loadRetrievalTrace(response.trace_id);
  $("technicalTrace").hidden = false;
  $("technicalTraceBody").textContent = `request_id: ${response.request_id || "—"}\ntrace_id: ${response.trace_id || "—"}\nindex_id: ${response.index_id || "—"}\nresolved_query: ${response.resolved_query || response.original_query || "—"}`;
}

function agentCitations(trace) {
  const answerCitations = trace.result?.answer?.citations || [];
  if (answerCitations.length) return answerCitations;
  return (trace.evidence_memory?.items || []).map((item, index) => ({
    ordinal: index + 1, chunk_id: item.chunk_id, page_start: item.page_start,
    page_end: item.page_end, section_path: item.section_path, excerpt: item.excerpt,
  }));
}

function renderAgentResponse(trace) {
  state.currentTrace = trace;
  const answer = trace.result?.answer || {};
  const verification = trace.evidence_verification;
  renderCommonAnswer({
    ...answer,
    outcome: trace.result?.outcome || "abstain",
    citations: agentCitations(trace),
  }, {
    taskId: trace.task_id,
    taskLabel: { compare: "对比分析", extract: "精确抽取", calculate: "计算核验" }[trace.task_type] || trace.task_type,
    provider: trace.model_trace?.model || "DeepSeek Agent",
    title: trace.result?.outcome === "answer" ? "Agent 已完成并通过证据门禁" : "Agent 已停止，未输出未经证明的答案",
    summary: [
      ["工具调用", `${trace.tool_calls?.length || 0} 次`],
      ["停止原因", trace.stop_reason || "—"],
      ["第二 Agent", verification?.routed ? verification.final_decision : "未触发"],
    ],
  });
  renderAgentProcess(trace);
  $("technicalTrace").hidden = false;
  $("technicalTraceBody").textContent = `task_id: ${trace.task_id}\nindex_id: ${trace.index_id}\nruntime: ${trace.runtime}\nrounds: ${trace.rounds_completed}\nstop_reason: ${trace.stop_reason}\nmodel: ${trace.model_trace?.model || "—"}\ninput_tokens: ${trace.model_trace?.input_tokens ?? "—"}\noutput_tokens: ${trace.model_trace?.output_tokens ?? "—"}`;
}

async function loadRetrievalTrace(traceId) {
  if (!traceId) return;
  try {
    const trace = await api(`/v1/traces/${encodeURIComponent(traceId)}`);
    const steps = [
      ["理解问题与口径", `模式 ${trace.mode || "lexical"}，候选池 ${trace.candidate_k || "—"}`, ""],
      ...(trace.stages || []).map((stage) => [stageName(stage.stage), `${stage.candidate_count} 个候选`, milliseconds(stage.duration_ms)]),
      ["生成并核验引用", `${trace.result_count || 0} 条证据进入回答`, milliseconds(trace.total_duration_ms)],
    ];
    renderProcessSteps(steps);
  } catch (error) {
    $("processList").innerHTML = `<div class="process-empty">执行过程暂不可用：${esc(error.message)}</div>`;
  }
}

function stageName(stage) {
  const names = { lexical: "关键词检索", dense: "向量检索", fusion: "融合排序", scope: "口径校验", rerank: "精细重排" };
  return names[stage] || stage;
}

function renderAgentProcess(trace) {
  const steps = [["任务规划", planSummary(trace.plan), ""]];
  (trace.tool_calls || []).forEach((call) => {
    const label = {
      search_evidence: "检索文档证据", search_authoritative_source: "查找权威来源",
      get_page_window: "读取相邻页面", reconstruct_page_layout: "重建页面结构",
      inspect_page_region: "检查 PDF 局部", calculate: "执行精确计算",
    }[call.tool] || call.tool;
    steps.push([label, `${call.status === "success" ? "完成" : "失败"} · ${call.evidence_chunk_ids?.length || 0} 条证据`, milliseconds(call.duration_ms)]);
  });
  if (trace.claim_risk_gate) {
    steps.push(["本地风险门禁", `${trace.claim_risk_gate.status} · ${trace.claim_risk_gate.findings?.length || 0} 个风险信号`, "0 token"]);
  }
  if (trace.evidence_verification?.routed) {
    steps.push(["独立 Evidence Verifier", `${trace.evidence_verification.final_decision} · ${trace.evidence_verification.turns?.length || 0} 次复核`, "第二 Agent"]);
  }
  steps.push(["形成最终结论", trace.stop_reason || trace.status, ""]);
  renderProcessSteps(steps);
}

function planSummary(plan) {
  if (!plan) return "已建立任务计划";
  const parts = [];
  if (plan.targets?.length) parts.push(`${plan.targets.length} 个目标`);
  if (plan.required_metrics?.length) parts.push(`${plan.required_metrics.length} 个指标`);
  if (plan.fact_requirements?.length) parts.push(`${plan.fact_requirements.length} 项原子事实`);
  return parts.join(" · ") || "已确定文档范围与证据要求";
}

function renderProcessSteps(steps) {
  $("processList").innerHTML = steps.map(([title, detail, time], index) => `<div class="process-step"><span class="step-mark">${index + 1}</span><div><b>${esc(title)}</b><p>${esc(detail)}</p></div>${time ? `<time>${esc(time)}</time>` : ""}</div>`).join("");
}

function evidenceEmpty(title, detail) {
  return `<div class="evidence-empty"><svg viewBox="0 0 48 48" aria-hidden="true"><path d="M13 7h17l6 6v28H13Z"/><path d="M30 7v7h7M18 22h13M18 28h13M18 34h8"/></svg><b>${esc(title)}</b><span>${esc(detail)}</span></div>`;
}

function claimForOrdinal(ordinal) {
  const claims = state.claimCitations.filter((claim) => (claim.citation_ordinals || []).map(Number).includes(Number(ordinal))).map((claim) => claim.claim);
  return claims.length ? claims.join("；") : "为回答提供原文依据";
}

function readableExcerpt(text) {
  return esc(text || "服务未返回原文摘录").replace(/(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?/g, (value) => `<mark>${value}</mark>`);
}

function renderEvidence() {
  const items = state.citations;
  $("evidenceCount").textContent = `${items.length} 条`;
  if (!items.length) {
    $("evidenceList").innerHTML = evidenceEmpty("没有引用证据", "拒答或澄清结果不会伪造来源。");
    return;
  }
  $("evidenceList").innerHTML = items.map((citation) => {
    const resolved = state.resolvedEvidence.get(citation.chunk_id)?.chunk;
    const status = state.evidenceStatus.get(citation.chunk_id) || "checking";
    const sectionPath = resolved?.section_path || citation.section_path || [];
    const title = sectionPath.at(-1) || resolved?.company_name || "财务报告原文";
    const pageStart = resolved?.page_start ?? citation.page_start;
    const pageEnd = resolved?.page_end ?? citation.page_end;
    const pages = pageStart === pageEnd ? `第 ${pageStart} 页` : `第 ${pageStart}–${pageEnd} 页`;
    const meta = [resolved?.company_name, resolved?.report_year && `${resolved.report_year} 年`, pages, sectionPath.length > 1 && sectionPath.slice(0, -1).join(" › ")].filter(Boolean);
    const verificationText = { checking: "正在核验", verified: "内容已核验", failed: "核验失败" }[status];
    const sha = state.resolvedEvidence.get(citation.chunk_id)?.sha256 || "尚未返回";
    const excerpt = resolved?.text || citation.excerpt || "";
    return `<article class="evidence-card" data-evidence="${esc(citation.ordinal)}" id="evidence-${esc(citation.ordinal)}">
      <div class="source-head">
        <span class="source-number">${esc(citation.ordinal)}</span>
        <div class="source-title"><h3>${esc(title)}</h3><div class="source-meta">${meta.map((item) => `<span>${esc(item)}</span>`).join("")}</div></div>
        <span class="verify-status ${esc(status)}">${esc(verificationText)}</span>
      </div>
      <div class="supports-claim"><b>支持结论</b>${esc(claimForOrdinal(citation.ordinal))}</div>
      <blockquote class="evidence-quote">${readableExcerpt(excerpt)}</blockquote>
      <details class="evidence-tech"><summary>技术详情与防篡改信息</summary><dl>
        <dt>chunk_id</dt><dd>${esc(citation.chunk_id || "—")}</dd>
        <dt>SHA-256</dt><dd>${esc(sha)}</dd>
        <dt>页码</dt><dd>${esc(pages)}</dd>
        <dt>章节路径</dt><dd>${esc(sectionPath.join(" › ") || "未标注")}</dd>
      </dl></details>
    </article>`;
  }).join("");
}

async function verifyEvidence() {
  const wanted = state.citations.filter((citation) => citation.chunk_id);
  if (!state.indexId || !wanted.length) return;
  wanted.forEach((citation) => state.evidenceStatus.set(citation.chunk_id, "checking"));
  try {
    const response = await postJSON("/v1/evidence:resolve", {
      index_id: state.currentTrace?.index_id || state.currentTrace?.indexId || state.indexId,
      chunk_ids: wanted.map((citation) => citation.chunk_id),
    });
    (response.evidence || []).forEach((item, index) => {
      const chunkId = wanted[index]?.chunk_id;
      if (!chunkId) return;
      state.resolvedEvidence.set(chunkId, item);
      state.evidenceStatus.set(chunkId, "verified");
    });
  } catch (_) {
    wanted.forEach((citation) => state.evidenceStatus.set(citation.chunk_id, "failed"));
  }
  renderEvidence();
}

function focusEvidence(ordinal) {
  document.querySelectorAll(".evidence-card").forEach((card) => card.classList.toggle("active", Number(card.dataset.evidence) === Number(ordinal)));
  document.querySelectorAll(".citation-link").forEach((link) => link.classList.toggle("active", Number(link.dataset.citation) === Number(ordinal)));
  document.querySelector(`[data-evidence="${ordinal}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function focusClaim(element) {
  const ordinals = new Set([...element.querySelectorAll("[data-citation]")].map((button) => Number(button.dataset.citation)));
  document.querySelectorAll(".evidence-card").forEach((card) => card.classList.toggle("dimmed", ordinals.size && !ordinals.has(Number(card.dataset.evidence))));
}
function clearEvidenceFocus() { document.querySelectorAll(".evidence-card").forEach((card) => card.classList.remove("dimmed")); }

function chooseFile(file) {
  if (!file || !file.name.toLowerCase().endsWith(".pdf")) { showNotice("只能上传 PDF 文件。", "error"); return; }
  if (file.size > 100 * 1024 * 1024) { showNotice("PDF 不能超过 100 MB。", "error"); return; }
  state.selectedFile = file;
  state.uploadJobId = "";
  $("fileName").textContent = file.name;
  $("fileHint").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · 已准备好`;
  $("uploadButton").disabled = false;
}

function uploadMetadata() {
  const company = $("uploadCompany").value.trim();
  const year = Number($("uploadYear").value);
  return {
    ...(company ? { company_name: company } : {}),
    ...(Number.isInteger(year) && year > 0 ? { report_year: year } : {}),
    document_type: $("uploadType").value,
  };
}

function automaticDocumentKey(jobId) {
  const metadata = uploadMetadata();
  const company = String(metadata.company_name || "uploaded").replace(/[^\p{L}\p{N}-]+/gu, "-").replace(/^-|-$/g, "");
  const year = metadata.report_year || "unknown";
  return `web:${company || "uploaded"}:${year}:${jobId.slice(0, 12)}`;
}

function setUploadProgress(percent, status, message = "") {
  $("uploadProgress").hidden = false;
  $("uploadProgressBar").style.width = `${Math.max(0, Math.min(100, percent))}%`;
  $("uploadStatus").textContent = status;
  $("uploadMessage").textContent = message;
}

async function uploadAndProcess() {
  if (!state.selectedFile) return;
  $("uploadButton").disabled = true;
  try {
    setUploadProgress(10, "正在上传", "文件将进入当前 FinDocRAG 服务");
    const upload = await api("/v1/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/pdf", "X-Filename": state.selectedFile.name },
      body: state.selectedFile,
    });
    state.uploadJobId = upload.job_id;
    setUploadProgress(25, "上传完成", "正在提交解析与索引任务");
    const documentKey = $("documentKey").value.trim() || automaticDocumentKey(upload.job_id);
    const job = await postJSON(`/v1/uploads/${encodeURIComponent(upload.job_id)}:process`, {
      document_key: documentKey,
      metadata: uploadMetadata(),
    });
    renderUploadJob(job);
    pollUploadJob(upload.job_id);
  } catch (error) {
    setUploadProgress(0, "上传失败", error.message);
    $("uploadButton").disabled = false;
  }
}

const UPLOAD_PROGRESS = { uploaded: 20, validating: 38, ingesting: 60, indexing: 82, ready: 100, failed: 0 };
const UPLOAD_LABEL = { uploaded: "已上传", validating: "正在校验 PDF", ingesting: "正在解析文档", indexing: "正在建立索引", ready: "文档已就绪", failed: "处理失败" };
function renderUploadJob(job) {
  setUploadProgress(UPLOAD_PROGRESS[job.status] ?? 0, UPLOAD_LABEL[job.status] || job.status, job.message || "");
  if (job.status === "ready") {
    clearInterval(state.uploadTimer);
    $("uploadButton").disabled = false;
    $("uploadButton").textContent = "继续上传其他 PDF";
    toast("文档已建立索引，可以开始提问");
    refreshStatus();
  } else if (job.status === "failed") {
    clearInterval(state.uploadTimer);
    $("uploadButton").disabled = false;
  }
}

function pollUploadJob(jobId) {
  clearInterval(state.uploadTimer);
  state.uploadTimer = setInterval(async () => {
    try {
      const job = await api(`/v1/uploads/${encodeURIComponent(jobId)}`);
      renderUploadJob(job);
      if (["ready", "failed"].includes(job.status)) clearInterval(state.uploadTimer);
    } catch (_) {
      clearInterval(state.uploadTimer);
    }
  }, 1200);
}

async function loadReviews() {
  const list = $("reviewList");
  list.innerHTML = '<div class="drawer-empty">正在读取审核队列…</div>';
  try {
    const reviews = await api("/v1/reviews?status=pending");
    if (!reviews.length) {
      list.innerHTML = '<div class="drawer-empty">当前没有待审核任务。</div>';
      return;
    }
    list.innerHTML = reviews.map(reviewCard).join("");
  } catch (error) {
    list.innerHTML = `<div class="drawer-empty">审核队列不可用：${esc(error.message)}</div>`;
  }
}

function reviewCard(item) {
  const packet = item.packet;
  const evidence = (packet.requirements || []).flatMap((requirement) => requirement.evidence || []);
  const uniqueEvidence = [...new Map(evidence.map((entry) => [entry.chunk_id, entry])).values()];
  return `<article class="review-card" data-review="${esc(packet.review_id)}">
    <h4>${esc(packet.query)}</h4>
    <p>${esc((packet.reasons || []).join("；"))}</p>
    ${packet.candidate_result?.answer?.answer ? `<div class="supports-claim"><b>候选答案</b>${esc(packet.candidate_result.answer.answer)}</div>` : ""}
    <details class="advanced-upload"><summary>查看审核证据与修正选项</summary>
      <div class="field"><label>审核人</label><input data-reviewer value="web-reviewer"></div>
      <div class="field"><label>修正后的答案</label><textarea data-corrected placeholder="仅选择“修正”时填写"></textarea></div>
      <div class="field"><label>修正答案使用的证据</label>${uniqueEvidence.map((entry) => `<label style="display:flex;gap:7px;align-items:flex-start;font-weight:400"><input type="checkbox" data-review-evidence value="${esc(entry.chunk_id)}" style="width:auto;min-height:0"><span>第 ${esc(entry.page_start)} 页 · ${esc((entry.section_path || []).at(-1) || entry.chunk_id)}</span></label>`).join("") || "<small>审核包没有可选证据。</small>"}</div>
    </details>
    <div class="review-actions"><button class="approve" type="button" data-review-decision="approve">批准</button><button type="button" data-review-decision="correct">修正</button><button class="reject" type="button" data-review-decision="reject">驳回</button></div>
  </article>`;
}

async function resolveReview(card, decision) {
  const reviewId = card.dataset.review;
  const reviewer = card.querySelector("[data-reviewer]")?.value.trim() || "web-reviewer";
  const correctedAnswer = card.querySelector("[data-corrected]")?.value.trim() || null;
  const evidenceIds = [...card.querySelectorAll("[data-review-evidence]:checked")].map((input) => input.value);
  if (decision === "correct" && (!correctedAnswer || !evidenceIds.length)) {
    toast("修正答案需要填写内容并选择审核包内证据");
    card.querySelector("details")?.setAttribute("open", "");
    return;
  }
  try {
    await postJSON(`/v1/reviews/${encodeURIComponent(reviewId)}:resolve`, {
      decision, reviewer, corrected_answer: correctedAnswer, evidence_chunk_ids: evidenceIds,
    });
    toast("审核结论已写入不可变记录");
    loadReviews();
  } catch (error) {
    toast(`审核失败：${error.message}`);
  }
}

document.querySelectorAll("[data-open-panel]").forEach((button) => button.addEventListener("click", () => openDrawer(button.dataset.openPanel)));
document.querySelectorAll("[data-panel]").forEach((button) => button.addEventListener("click", () => selectPanel(button.dataset.panel)));
$("drawerClose").addEventListener("click", closeDrawer);
$("drawerScrim").addEventListener("click", closeDrawer);
$("themeButton").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
$("saveKeyButton").addEventListener("click", saveKey);
$("clearKeyButton").addEventListener("click", clearKey);
$("toggleKeyButton").addEventListener("click", () => {
  const input = $("apiKeyInput");
  input.type = input.type === "password" ? "text" : "password";
  $("toggleKeyButton").textContent = input.type === "password" ? "显示" : "隐藏";
});
$("apiKeyInput").addEventListener("keydown", (event) => { if (event.key === "Enter") saveKey(); });
$("taskSwitch").addEventListener("click", (event) => { const button = event.target.closest("[data-task]"); if (button) selectTask(button.dataset.task); });
$("suggestions").addEventListener("click", (event) => {
  const button = event.target.closest("[data-query]");
  if (!button) return;
  selectTask(button.dataset.task);
  $("queryInput").value = button.dataset.query;
});
$("runButton").addEventListener("click", runTask);
$("queryInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) runTask();
});
$("answerContent").addEventListener("click", (event) => {
  const citation = event.target.closest("[data-citation]");
  if (citation) focusEvidence(Number(citation.dataset.citation));
});
$("answerContent").addEventListener("mouseover", (event) => { const claim = event.target.closest(".claim-row"); if (claim) focusClaim(claim); });
$("answerContent").addEventListener("mouseleave", clearEvidenceFocus);
$("evidenceList").addEventListener("click", (event) => { const card = event.target.closest("[data-evidence]"); if (card && !event.target.closest("summary")) focusEvidence(Number(card.dataset.evidence)); });
$("copyTraceButton").addEventListener("click", () => state.currentTaskId && copyText(state.currentTaskId, state.taskType === "query" ? "Trace ID" : "Task ID"));
$("refreshReviewsButton").addEventListener("click", loadReviews);
$("reviewList").addEventListener("click", (event) => {
  const button = event.target.closest("[data-review-decision]");
  const card = event.target.closest("[data-review]");
  if (button && card) resolveReview(card, button.dataset.reviewDecision);
});

const dropzone = $("dropzone");
dropzone.addEventListener("click", () => $("fileInput").click());
dropzone.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); $("fileInput").click(); } });
$("fileInput").addEventListener("change", (event) => chooseFile(event.target.files?.[0]));
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", (event) => chooseFile(event.dataTransfer?.files?.[0]));
$("uploadButton").addEventListener("click", uploadAndProcess);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && $("drawer").classList.contains("open")) closeDrawer();
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && document.activeElement !== $("queryInput")) runTask();
});

selectTask("query");
updateKeyState();
refreshStatus();
