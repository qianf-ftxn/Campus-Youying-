const AUTH_KEY = "mindbridge.auth";

const state = {
  profile: null,
  modelName: "mock",
  dashboard: {
    reports: [],
    cases: [],
    excel: [],
    alerts: []
  }
};

const els = {
  serviceState: document.querySelector("#serviceState"),
  modelState: document.querySelector("#modelState"),
  activeAccount: document.querySelector("#activeAccount"),
  switchAccount: document.querySelector("#switchAccount"),
  refreshAdmin: document.querySelector("#refreshAdmin"),
  metricReports: document.querySelector("#metricReports"),
  metricHigh: document.querySelector("#metricHigh"),
  metricCases: document.querySelector("#metricCases"),
  metricExcel: document.querySelector("#metricExcel"),
  metricAlerts: document.querySelector("#metricAlerts"),
  metricHighMeta: document.querySelector("#metricHighMeta"),
  metricCasesMeta: document.querySelector("#metricCasesMeta"),
  metricExcelMeta: document.querySelector("#metricExcelMeta"),
  metricAlertsMeta: document.querySelector("#metricAlertsMeta"),
  lastUpdated: document.querySelector("#lastUpdated"),
  trendTotal: document.querySelector("#trendTotal"),
  activityTrend: document.querySelector("#activityTrend"),
  riskDistribution: document.querySelector("#riskDistribution"),
  emotionDistribution: document.querySelector("#emotionDistribution"),
  intentDistribution: document.querySelector("#intentDistribution"),
  toolDistribution: document.querySelector("#toolDistribution"),
  casesCount: document.querySelector("#casesCount"),
  reportsCount: document.querySelector("#reportsCount"),
  excelCount: document.querySelector("#excelCount"),
  alertCount: document.querySelector("#alertCount"),
  cases: document.querySelector("#cases"),
  reports: document.querySelector("#reports"),
  excelRecords: document.querySelector("#excelRecords"),
  alertRecords: document.querySelector("#alertRecords"),
  conversationState: document.querySelector("#conversationState"),
  conversationDetail: document.querySelector("#conversationDetail"),
  knowledgeState: document.querySelector("#knowledgeState"),
  knowledgeUploadForm: document.querySelector("#knowledgeUploadForm"),
  knowledgeFile: document.querySelector("#knowledgeFile"),
  knowledgeUploadState: document.querySelector("#knowledgeUploadState"),
  rebuildVector: document.querySelector("#rebuildVector"),
  backupVector: document.querySelector("#backupVector"),
  safetyCount: document.querySelector("#safetyCount"),
  safetyStates: document.querySelector("#safetyStates"),
  riskEventCount: document.querySelector("#riskEventCount"),
  riskEvents: document.querySelector("#riskEvents")
};

function readAuth() {
  try {
    return JSON.parse(sessionStorage.getItem(AUTH_KEY) || "null");
  } catch {
    return null;
  }
}

function clearAuth() {
  sessionStorage.removeItem(AUTH_KEY);
}

function authHeader() {
  const auth = readAuth();
  if (!auth?.token) {
    window.location.replace("/");
    return "";
  }
  return `Basic ${auth.token}`;
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}), Authorization: authHeader() };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response;
}

function setPill(el, text, tone = "ok") {
  el.textContent = text;
  el.className = `pill ${tone}`;
}

function isAdmin(profile) {
  return profile.roles?.some((role) => role.authority === "ROLE_ADMIN");
}

function displayModel(model) {
  return (model || "").includes("mindbridge-qwen2.5-7b-ft") ? "微调 Qwen2.5-7B" : model;
}

function displayTime(value) {
  if (!value) return "";
  const utc = typeof value === "string" && !/(Z|[+-]\d{2}:?\d{2})$/.test(value) ? value + "Z" : value;
  return new Date(utc).toLocaleString("zh-CN", {timeZone: "Asia/Shanghai"});
}

function formatShortDate(value) {
  return new Intl.DateTimeFormat("zh-CN", { month: "numeric", day: "numeric" }).format(value);
}

function percentage(value, total) {
  return total ? Math.round((value / total) * 100) : 0;
}

function statusLabel(status) {
  const labels = {
    SUCCESS: "成功",
    FAILED: "失败",
    OPEN: "待跟进",
    ALERT_SENT: "已预警",
    ACKNOWLEDGED: "已确认"
  };
  return labels[(status || "").toUpperCase()] || status || "未知";
}

function roleLabel(role) {
  const value = (role || "").toUpperCase();
  if (value === "USER") return "学生";
  if (value === "ASSISTANT") return "校园有应";
  if (value === "SYSTEM") return "系统";
  return role || "未知角色";
}

async function checkHealth() {
  try {
    const response = await fetch("/actuator/health");
    const body = await response.json();
    setPill(els.serviceState, body.status === "UP" ? "服务正常" : `服务 ${body.status}`, body.status === "UP" ? "ok" : "danger");
  } catch {
    setPill(els.serviceState, "服务 DOWN", "danger");
  }
}

async function loadProfile() {
  try {
    const response = await api("/api/profile");
    const profile = await response.json();
    if (!isAdmin(profile)) {
      window.location.replace("/student.html");
      return null;
    }
    state.profile = profile;
    els.activeAccount.textContent = profile.displayName || profile.username;
    return profile;
  } catch {
    clearAuth();
    window.location.replace("/");
    return null;
  }
}

async function loadAgentStatus() {
  const response = await api("/api/agent/status");
  const status = await response.json();
  state.modelName = status.model || "mock";
  if (status.realModelEnabled) {
    setPill(els.modelState, `${status.provider} / ${displayModel(state.modelName)}`, "ok");
  } else {
    setPill(els.modelState, "mock 演示", "warn");
  }
}

async function loadAdminDashboard() {
  els.refreshAdmin.disabled = true;
  els.refreshAdmin.textContent = "刷新中";
  try {
    const [reportsRes, casesRes, excelRes, alertsRes] = await Promise.all([
      api("/api/admin/reports"),
      api("/api/admin/cases"),
      api("/api/admin/excel-records"),
      api("/api/admin/alerts")
    ]);
    const reports = await reportsRes.json();
    const cases = await casesRes.json();
    const excel = await excelRes.json();
    const alerts = await alertsRes.json();
    state.dashboard = { reports, cases, excel, alerts };

    const highRisk = reports.filter((item) => item.riskLevel === "HIGH").length;
    const openCases = cases.filter((item) => item.status === "OPEN").length;
    const excelSuccess = excel.filter((item) => item.status === "SUCCESS").length;
    const alertSuccess = alerts.filter((item) => item.status === "SUCCESS").length;

    els.metricReports.textContent = reports.length;
    els.metricHigh.textContent = highRisk;
    els.metricCases.textContent = cases.length;
    els.metricExcel.textContent = excel.length;
    els.metricAlerts.textContent = alerts.length;
    els.metricHighMeta.textContent = `占比 ${percentage(highRisk, reports.length)}%`;
    els.metricCasesMeta.textContent = `待跟进 ${openCases}`;
    els.metricExcelMeta.textContent = `成功 ${excelSuccess}`;
    els.metricAlertsMeta.textContent = `发送成功 ${alertSuccess}`;
    els.lastUpdated.textContent = `更新于 ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
    els.casesCount.textContent = `${cases.length} 个个案`;
    els.reportsCount.textContent = `${reports.length} 条报告`;
    els.excelCount.textContent = `${excel.length} 条`;
    els.alertCount.textContent = `${alerts.length} 条`;

    renderTrend(reports);
    renderAnalytics(reports, excel, alerts);
    renderCases(cases);
    renderReports(reports);
    renderToolRecords(els.excelRecords, excel, "excel");
    renderToolRecords(els.alertRecords, alerts, "alert");
  } catch (error) {
    els.lastUpdated.textContent = `刷新失败：${error.message}`;
  } finally {
    els.refreshAdmin.disabled = false;
    els.refreshAdmin.textContent = "刷新数据";
  }
  await loadSafetyMemory();
}

async function loadSafetyMemory() {
  try {
    const [statesRes, eventsRes] = await Promise.all([
      api("/api/admin/safety-states"),
      api("/api/admin/risk-events")
    ]);
    const states = await statesRes.json();
    const events = await eventsRes.json();
    els.safetyCount.textContent = `${states.length} 个用户`;
    els.riskEventCount.textContent = `${events.length} 条`;
    renderSafetyStates(states);
    renderRiskEvents(events);
  } catch (error) {
    els.safetyCount.textContent = `读取失败：${error.message}`;
  }
}

function safetyStatusLabel(status) {
  const labels = { ACTIVE: "未解除", MONITORING: "持续关注", RESOLVED: "已解除", FALSE_POSITIVE: "已确认误判" };
  return labels[status] || status;
}

async function overrideSafety(userId, sessionId, action) {
  let reason;
  if (action === "MONITORING") {
    reason = window.prompt("人工确认进入【持续关注】需要说明安全复核和保护措施：", "已完成专业复核，当前有可信人员陪伴并进入持续关注");
  } else if (action === "RESOLVED") {
    reason = window.prompt("标记为【已解除】需要说明处理依据：", "人工确认处理完成，风险已解除");
  } else {
    reason = window.prompt("标记为【已确认误判】需要说明纠正依据：", "确认为关键词误判或修辞表达");
  }
  if (!reason) return;
  try {
    await api(`/api/admin/safety-states/${userId}/override`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, action, reason })
    });
    await loadSafetyMemory();
  } catch (error) {
    window.alert(`操作失败：${error.message}`);
  }
}

function renderSafetyStates(states) {
  els.safetyStates.innerHTML = "";
  if (!states.length) {
    els.safetyStates.innerHTML = `<div class="empty small"><strong>暂无安全状态记录</strong><p>每个会话独立记录安全状态和复核进度。</p></div>`;
    return;
  }
  for (const item of states) {
    const card = document.createElement("article");
    card.className = "case-card";

    const head = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `${item.displayName || item.username} · ${item.sessionTitle || item.sessionId}`;
    const badge = document.createElement("span");
    badge.className = `status-badge status-${String(item.status || "").toLowerCase()}`;
    badge.textContent = `${safetyStatusLabel(item.status)} · ${item.effectiveRisk}`;
    head.append(title, badge);

    const meta = document.createElement("small");
    const parts = [`复核 ${item.safeReassessmentCount} 次`, `连续 LOW ${item.consecutiveLowAssessments}`, `普通轮 ${item.normalTurnCountSinceReview}`];
    if (item.reassessmentRequired) parts.push("需要复核");
    if (item.humanReviewRequired) parts.push("需人工确认");
    meta.textContent = parts.join(" · ");

    const time = document.createElement("small");
    time.textContent = `最近评估 ${displayTime(item.lastAssessedAt)}${item.lastHighRiskAt ? ` · 最近高风险 ${displayTime(item.lastHighRiskAt)}` : ""}`;

    const actions = document.createElement("div");
    actions.className = "admin-actions";
    if (item.status === "ACTIVE" || item.status === "MONITORING") {
      if (item.status === "ACTIVE" && item.humanReviewRequired) {
        const monitorBtn = document.createElement("button");
        monitorBtn.className = "ghost";
        monitorBtn.type = "button";
        monitorBtn.textContent = "人工确认进入监测";
        monitorBtn.addEventListener("click", () => overrideSafety(item.userId, item.sessionId, "MONITORING"));
        actions.append(monitorBtn);
      }
      const resolveBtn = document.createElement("button");
      resolveBtn.className = "ghost";
      resolveBtn.type = "button";
      resolveBtn.textContent = "标记已解除";
      resolveBtn.addEventListener("click", () => overrideSafety(item.userId, item.sessionId, "RESOLVED"));
      const fpBtn = document.createElement("button");
      fpBtn.className = "ghost";
      fpBtn.type = "button";
      fpBtn.textContent = "标记误判";
      fpBtn.addEventListener("click", () => overrideSafety(item.userId, item.sessionId, "FALSE_POSITIVE"));
      actions.append(resolveBtn, fpBtn);
    }

    card.append(head, meta, time, actions);
    els.safetyStates.append(card);
  }
}

function renderRiskEvents(events) {
  els.riskEvents.innerHTML = "";
  if (!events.length) {
    els.riskEvents.innerHTML = `<div class="empty small"><strong>暂无风险事件</strong><p>风险检测、复核和状态变更会按时间记录。</p></div>`;
    return;
  }
  for (const item of events) {
    const card = document.createElement("article");
    card.className = "delivery-item";
    const head = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `#${item.id} 用户${item.userId} · ${item.eventKind}`;
    const badge = document.createElement("span");
    badge.className = `status-badge status-${String(item.riskLevel || "").toLowerCase()}`;
    badge.textContent = item.riskLevel;
    head.append(title, badge);
    const meta = document.createElement("small");
    const transition = item.previousStatus ? ` ${item.previousStatus} → ${item.newStatus}` : "";
    meta.textContent = `${item.source}${transition} · ${displayTime(item.createdAt)}`;
    const message = document.createElement("p");
    message.textContent = item.evidenceSummary || "无证据概述";
    card.append(head, meta, message);
    els.riskEvents.append(card);
  }
}

async function loadKnowledgeStatus() {
  try {
    const response = await api("/api/admin/knowledge/status");
    const status = await response.json();
    const vector = status.vectorAvailable ? `向量 ${status.vectorChunks ?? 0}` : "向量不可用";
    els.knowledgeState.textContent = `${status.school} · ${status.documents} 份文件 · ${status.parentChunks} 父块 / ${status.childChunks} 子块 · ${vector}`;
    document.getElementById("policyKnowledgeStatus").textContent = status.activePolicyDocuments > 0
      ? `已接入 ${status.activePolicyDocuments} 份本校有效政策文件。资料未覆盖的问题会提示补充对应文件。`
      : "尚未接入本校有效政策文件。请先上传对应的正式资料，系统不会猜测具体规则或金额。";
    await loadKnowledgeDocuments();
    await loadConversationRuns();
  } catch (error) {
    els.knowledgeState.textContent = `读取失败：${error.message}`;
  }
}

function localDateKey(value) {
  const date = new Date(value);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function renderTrend(reports) {
  const days = [];
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  for (let offset = 6; offset >= 0; offset -= 1) {
    const date = new Date(today);
    date.setDate(today.getDate() - offset);
    days.push({
      date,
      key: localDateKey(date),
      count: 0
    });
  }
  const dayMap = new Map(days.map((item) => [item.key, item]));
  for (const report of reports) {
    const day = dayMap.get(localDateKey(report.createdAt));
    if (day) day.count += 1;
  }
  const max = Math.max(1, ...days.map((item) => item.count));
  const total = days.reduce((sum, item) => sum + item.count, 0);
  els.trendTotal.textContent = `共 ${total} 条`;
  els.activityTrend.innerHTML = "";

  for (const item of days) {
    const column = document.createElement("div");
    column.className = "trend-column";
    column.title = `${formatShortDate(item.date)}：${item.count} 条报告`;

    const value = document.createElement("strong");
    value.textContent = item.count;
    const track = document.createElement("div");
    track.className = "trend-track";
    const bar = document.createElement("span");
    bar.style.height = `${item.count ? Math.max(12, (item.count / max) * 100) : 4}%`;
    track.append(bar);
    const label = document.createElement("small");
    label.textContent = formatShortDate(item.date);

    column.append(value, track, label);
    els.activityTrend.append(column);
  }
}

function countBy(items, key) {
  return items.reduce((result, item) => {
    const value = String(item[key] || "UNKNOWN").toUpperCase();
    result[value] = (result[value] || 0) + 1;
    return result;
  }, {});
}

function renderDistribution(container, definitions, counts) {
  container.innerHTML = "";
  const total = definitions.reduce((sum, item) => sum + (counts[item.key] || 0), 0);
  for (const item of definitions) {
    const count = counts[item.key] || 0;
    const row = document.createElement("div");
    row.className = "distribution-row";

    const label = document.createElement("span");
    label.textContent = item.label;
    const track = document.createElement("div");
    track.className = "distribution-track";
    const fill = document.createElement("span");
    fill.className = `distribution-fill tone-${item.tone}`;
    fill.style.width = `${percentage(count, total)}%`;
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = count;

    row.append(label, track, value);
    container.append(row);
  }
}

function renderAnalytics(reports, excel, alerts) {
  renderDistribution(els.riskDistribution, [
    { key: "LOW", label: "低风险", tone: "teal" },
    { key: "MEDIUM", label: "中风险", tone: "amber" },
    { key: "HIGH", label: "高风险", tone: "red" }
  ], countBy(reports, "riskLevel"));

  renderDistribution(els.emotionDistribution, [
    { key: "NORMAL", label: "平稳", tone: "teal" },
    { key: "ANXIETY", label: "焦虑", tone: "amber" },
    { key: "DEPRESSED", label: "低落", tone: "orange" },
    { key: "HIGH_RISK", label: "高风险", tone: "red" }
  ], countBy(reports, "emotion"));

  renderDistribution(els.intentDistribution, [
    { key: "CHAT", label: "日常交流", tone: "blue" },
    { key: "CONSULT", label: "旧版心理咨询记录", tone: "amber" },
    { key: "RISK", label: "情绪与安全支持", tone: "red" }
  ], countBy(reports, "intent"));

  const toolCounts = {
    EXCEL_SUCCESS: excel.filter((item) => item.status === "SUCCESS").length,
    EXCEL_FAILED: excel.filter((item) => item.status === "FAILED").length,
    ALERT_SUCCESS: alerts.filter((item) => item.status === "SUCCESS").length,
    ALERT_FAILED: alerts.filter((item) => item.status === "FAILED").length
  };
  renderDistribution(els.toolDistribution, [
    { key: "EXCEL_SUCCESS", label: "台账成功", tone: "teal" },
    { key: "EXCEL_FAILED", label: "台账失败", tone: "red" },
    { key: "ALERT_SUCCESS", label: "预警成功", tone: "blue" },
    { key: "ALERT_FAILED", label: "预警失败", tone: "red" }
  ], toolCounts);
}

function renderToolRecords(container, records, kind) {
  container.innerHTML = "";
  if (!records.length) {
    const empty = document.createElement("div");
    empty.className = "delivery-empty";
    empty.textContent = kind === "excel" ? "暂无 Excel 写入记录" : "暂无预警发送记录";
    container.append(empty);
    return;
  }
  for (const item of records.slice(0, 4)) {
    const card = document.createElement("article");
    card.className = "delivery-record";

    const head = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `报告 #${item.reportId}`;
    const badge = document.createElement("span");
    badge.className = `status-badge status-${String(item.status || "").toLowerCase()}`;
    badge.textContent = statusLabel(item.status);
    head.append(title, badge);

    const meta = document.createElement("small");
    meta.textContent = kind === "excel"
      ? `${displayTime(item.createdAt)}${item.filePath ? ` · ${item.filePath}` : ""}`
      : `${item.channel || "通知"} · ${item.recipient || "未配置接收人"} · ${displayTime(item.createdAt)}`;
    const message = document.createElement("p");
    message.textContent = item.message || "无附加信息";
    card.append(head, meta, message);
    container.append(card);
  }
}

function renderCases(cases) {
  els.cases.innerHTML = "";
  if (!cases.length) {
    els.cases.innerHTML = `<div class="empty small"><strong>暂无个案</strong><p>中高风险报告会自动创建风险个案。</p></div>`;
    return;
  }
  for (const item of cases) {
    const card = document.createElement("article");
    card.className = `case-card risk-${item.riskLevel.toLowerCase()}`;

    const head = document.createElement("div");
    head.className = "case-head";
    const titleLine = document.createElement("div");
    titleLine.className = "record-title-line";
    const title = document.createElement("strong");
    title.textContent = `个案 #${item.id}`;
    const badge = document.createElement("span");
    badge.className = `status-badge case-status-${String(item.status || "").toLowerCase()}`;
    badge.textContent = statusLabel(item.status);
    titleLine.append(title, badge);
    const time = document.createElement("span");
    time.textContent = displayTime(item.updatedAt);
    head.append(titleLine, time);

    const meta = document.createElement("small");
    meta.textContent = `报告 #${item.reportId} · ${item.riskLevel} · 负责人：${item.owner || "未分配"}`;
    const summary = document.createElement("p");
    summary.textContent = item.summary || "";
    const handoff = document.createElement("pre");
    handoff.textContent = item.handoffSummary || "";

    card.append(head, meta, summary, handoff);
    els.cases.append(card);
  }
}

function renderReports(reports) {
  els.reports.innerHTML = "";
  if (!reports.length) {
    els.reports.innerHTML = `<div class="empty small"><strong>暂无报告</strong><p>达到安全报告触发条件的记录会在这里展示；普通政策咨询不生成心理报告。</p></div>`;
    return;
  }
  for (const item of reports) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `report risk-${item.riskLevel.toLowerCase()}`;
    card.dataset.sessionId = item.sessionId;

    const head = document.createElement("div");
    head.className = "report-head";
    const titleLine = document.createElement("div");
    titleLine.className = "record-title-line";
    const title = document.createElement("strong");
    title.textContent = item.displayName || item.username || "未知学生";
    const badge = document.createElement("span");
    badge.className = `risk-badge risk-${String(item.riskLevel || "low").toLowerCase()}`;
    badge.textContent = item.riskLevel || "LOW";
    titleLine.append(title, badge);
    const time = document.createElement("span");
    time.textContent = displayTime(item.createdAt);
    head.append(titleLine, time);

    const summary = document.createElement("p");
    summary.textContent = item.summary;
    const content = document.createElement("small");
    content.textContent = `${item.emotion || "NORMAL"} / ${item.intent || "CHAT"} · ${item.content}`;
    const action = document.createElement("span");
    action.className = "report-action";
    action.textContent = "查看会话档案";

    card.append(head, summary, content, action);
    card.addEventListener("click", () => loadConversation(item.sessionId));
    els.reports.append(card);
  }
}

async function loadConversation(sessionId) {
  if (!sessionId) {
    els.conversationState.textContent = "该报告缺少会话 ID";
    return;
  }
  els.conversationState.textContent = "正在读取...";
  els.conversationDetail.innerHTML = `<div class="empty small"><strong>加载中</strong><p>正在读取历史消息。</p></div>`;
  for (const card of els.reports.querySelectorAll(".report")) {
    card.classList.toggle("active", card.dataset.sessionId === sessionId);
  }
  try {
    const response = await api(`/api/admin/conversations/${encodeURIComponent(sessionId)}`);
    const conversation = await response.json();
    renderConversation(conversation);
  } catch (error) {
    els.conversationState.textContent = "读取失败";
    els.conversationDetail.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty small";
    const title = document.createElement("strong");
    title.textContent = "无法查看档案";
    const detail = document.createElement("p");
    detail.textContent = error.message;
    empty.append(title, detail);
    els.conversationDetail.append(empty);
  }
}

function renderConversation(conversation) {
  const messages = conversation.messages || [];
  els.conversationState.textContent = `${conversation.title || conversation.sessionId} · ${messages.length} 条消息`;
  els.conversationDetail.innerHTML = "";
  if (!messages.length) {
    els.conversationDetail.innerHTML = `<div class="empty small"><strong>暂无消息</strong><p>这个会话还没有写入消息记录。</p></div>`;
    return;
  }
  for (const message of messages) {
    const role = (message.role || "").toLowerCase();
    const row = document.createElement("article");
    row.className = `conversation-message ${role}`;

    const meta = document.createElement("div");
    meta.className = "conversation-meta";
    const label = document.createElement("strong");
    label.textContent = roleLabel(message.role);
    const time = document.createElement("span");
    time.textContent = displayTime(message.createdAt);
    meta.append(label, time);

    const bubble = document.createElement("div");
    bubble.className = "conversation-bubble";
    bubble.textContent = message.content || "";

    row.append(meta, bubble);
    if (role === "assistant") {
      appendReplySources(row, message.sources || [], message.traceId);
    }
    els.conversationDetail.append(row);
  }
}

function appendReplySources(container, sources, traceId) {
  const details = document.createElement("details");
  details.className = "source-card";
  const summary = document.createElement("summary");
  summary.textContent = `本条回复来源（${sources.length}）${traceId ? ` · 运行 #${traceId}` : ""}`;
  details.append(summary);
  if (!sources.length) {
    const note = document.createElement("p");
    note.textContent = "本条回复未使用检索资料，或旧记录没有来源关联。";
    details.append(note);
  }
  for (const source of sources) {
    const section = document.createElement("section"); section.className = "source-item";
    const title = document.createElement("strong"); title.textContent = source.title || source.source || "未命名文件";
    const meta = document.createElement("p");
    meta.textContent = [source.source, source.heading, source.version ? `版本 ${source.version}` : "",
      (source.page || source.page_start) ? `第 ${source.page || source.page_start} 页` : ""].filter(Boolean).join(" · ");
    const excerpt = document.createElement("pre"); excerpt.className = "chunk-preview"; excerpt.textContent = source.content || "无片段正文";
    section.append(title, meta, excerpt);
    try {
      const url = new URL(source.url || source.source_url);
      if (["http:", "https:"].includes(url.protocol)) {
        const link = document.createElement("a"); link.href = url.href; link.textContent = "查看原文件";
        link.target = "_blank"; link.rel = "noopener noreferrer"; section.append(link);
      }
    } catch {}
    details.append(section);
  }
  container.append(details);
}

async function uploadKnowledgeFile(event) {
  event.preventDefault();
  const file = els.knowledgeFile.files?.[0];
  if (!file) {
    els.knowledgeUploadState.textContent = "请先选择文件";
    return;
  }
  const data = new FormData();
  data.append("file", file);
  const field = (id) => document.getElementById(id).value.trim();
  data.append("metadata", JSON.stringify({
    title: field("knowledgeTitle"), domain: field("knowledgeDomain"), version: field("knowledgeVersion") || null,
    department: field("knowledgeDepartment"), source_url: field("knowledgeUrl"),
    published_on: field("knowledgePublished") || null, effective_on: field("knowledgeEffective") || null,
    expires_on: field("knowledgeExpires") || null, status: field("knowledgeStatus")
  }));
  els.knowledgeUploadState.textContent = "正在切分入库...";
  try {
    const response = await api("/api/admin/knowledge/file", { method: "POST", body: data });
    const result = await response.json();
    els.knowledgeUploadState.textContent = `${result.source} 已入库 ${result.chunks} 个片段`;
    els.knowledgeFile.value = "";
    loadKnowledgeStatus();
  } catch (error) {
    els.knowledgeUploadState.textContent = `上传失败：${error.message}`;
  }
}

async function runKnowledgeAction(kind) {
  const isBackup = kind === "backup";
  const button = isBackup ? els.backupVector : els.rebuildVector;
  const endpoint = isBackup ? "/api/admin/knowledge/backup" : "/api/admin/knowledge/rebuild-vector";
  const original = button.textContent;
  button.disabled = true;
  els.knowledgeUploadState.textContent = isBackup ? "正在备份向量索引..." : "正在重建向量索引...";
  try {
    const response = await api(endpoint, { method: "POST" });
    const result = await response.json();
    els.knowledgeUploadState.textContent = isBackup
      ? `备份完成：${result.snapshot}`
      : `重建完成：${result.indexedChunks} 个片段`;
    loadKnowledgeStatus();
  } catch (error) {
    els.knowledgeUploadState.textContent = `操作失败：${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function logout() {
  clearAuth();
  window.location.assign("/");
}

els.switchAccount.addEventListener("click", logout);
els.refreshAdmin.addEventListener("click", () => {
  loadAdminDashboard();
  loadKnowledgeStatus();
});
els.knowledgeUploadForm.addEventListener("submit", uploadKnowledgeFile);
els.rebuildVector.addEventListener("click", () => runKnowledgeAction("rebuild"));
els.backupVector.addEventListener("click", () => runKnowledgeAction("backup"));

checkHealth();
loadProfile().then((profile) => {
  if (!profile) return;
  loadPersonalMemoryStatus();
  loadAgentStatus();
  loadAdminDashboard();
  loadKnowledgeStatus();
});

async function loadPersonalMemoryStatus() {
  const box = document.getElementById("personalMemoryStatus");
  if (!box) return;
  try {
    const body = await (await api("/api/admin/long-term-memory/status")).json();
    box.replaceChildren();
    for (const row of body.items) {
      const line = document.createElement("p");
      line.textContent = `用户 #${row.userId} · ${row.state} · 任务 ${row.jobState} · 记忆 ${row.memoryCount} 条 · 待同步 ${row.indexPending} 条 · 召回 ${row.recallCount} 次 · 最近 ${row.recallState} / ${row.lastRecalledCount} 条${row.errorCode ? ` · 故障 ${row.errorCode}` : ""}`;
      box.append(line);
    }
    if (!body.items.length) box.textContent = body.enabled ? "暂无长期记忆处理任务" : "长期记忆未启用";
  } catch {
    box.textContent = "长期记忆状态暂不可用";
  }
}
document.getElementById("refreshPersonalMemory")?.addEventListener("click", loadPersonalMemoryStatus);

async function loadKnowledgeDocuments() {
  const container = document.getElementById("knowledgeDocuments");
  if (!container) return;
  const docs = await (await api("/api/admin/knowledge/documents")).json();
  container.replaceChildren();
  for (const doc of docs) {
    const card = document.createElement("article"); card.className = "knowledge-document";
    const title = document.createElement("strong"); title.textContent = doc.title;
    const meta = document.createElement("p"); meta.className = "hint";
    const label = {ACTIVE: "启用", DRAFT: "草稿", ARCHIVED: "归档"}[doc.status] || doc.status;
    meta.textContent = `${doc.domain === "policy" ? "校园政策" : "情绪支持"} · ${doc.school} · 版本 ${doc.version} · ${label}${doc.isDemo ? " · 演示资料" : ""}${doc.effectiveOn ? " · 生效 " + doc.effectiveOn : ""}`;
    const preview = document.createElement("div"); preview.hidden = true;
    const show = document.createElement("button"); show.type = "button"; show.className = "ghost"; show.textContent = "预览父子块";
    show.addEventListener("click", async () => {
      try {
        if (!preview.childNodes.length) {
          const detail = await (await api(`/api/admin/knowledge/documents/${doc.id}`)).json();
          for (const chunk of detail.chunks) {
            const block = document.createElement("details");
            const head = document.createElement("summary");
            head.textContent = `${chunk.type === "parent" ? "父块" : "子块"} #${chunk.id}${chunk.parentId ? " → 父块 #" + chunk.parentId : ""} · ${chunk.heading}`;
            const body = document.createElement("pre"); body.className = "chunk-preview"; body.textContent = chunk.content;
            block.append(head, body); preview.append(block);
          }
        }
        preview.hidden = !preview.hidden;
      } catch (error) { els.knowledgeUploadState.textContent = error.message; }
    });
    card.append(title, meta, show);
    for (const [status, text] of [["ACTIVE", "启用此版本"], ["ARCHIVED", "停用归档"]]) {
      if (doc.status === status) continue;
      const action = document.createElement("button"); action.type = "button"; action.className = "ghost"; action.textContent = text;
      action.addEventListener("click", async () => {
        action.disabled = true;
        try {
          await api(`/api/admin/knowledge/documents/${doc.id}/status`, {method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({status})});
          await loadKnowledgeStatus();
        } catch (error) { els.knowledgeUploadState.textContent = error.message; action.disabled = false; }
      });
      card.append(action);
    }
    card.append(preview); container.append(card);
  }
}

async function loadConversationRuns() {
  const container = document.getElementById("conversationRuns");
  const runs = await (await api("/api/admin/agent-traces")).json();
  const counts = {CHAT: 0, CONSULT: 0, RISK: 0};
  container.replaceChildren();
  for (const run of runs) {
    const intent = run.assessment?.businessIntent || run.intent;
    counts[intent] = (counts[intent] || 0) + 1;
    const card = document.createElement("article"); card.className = "knowledge-document";
    const title = document.createElement("strong"); title.textContent = run.originalInput;
    const meta = document.createElement("p"); meta.className = "hint";
    meta.textContent = `${run.username} · ${{CHAT: "日常交流", CONSULT: "政策咨询", RISK: "情绪与安全支持"}[intent] || intent} · 资料 ${run.retrievedKnowledge.length} 条 · ${displayTime(run.createdAt)}`;
    const view = document.createElement("button"); view.type = "button"; view.className = "ghost"; view.textContent = "查看会话";
    view.addEventListener("click", () => loadConversation(run.sessionId));
    const trace = document.createElement("details");
    const head = document.createElement("summary"); head.textContent = "检索与协作过程";
    const body = document.createElement("pre"); body.className = "chunk-preview";
    body.textContent = JSON.stringify(run.agentSteps.filter(step => step.kind === "agent_artifact" || step.kind === "agent_event"), null, 2);
    trace.append(head, body); card.append(title, meta, view, trace); container.append(card);
    const reply = run.responseMessages.findLast(message => message.kind === "approved_candidate" || message.kind === "interrupted_candidate");
    if (reply) {
      const answer = document.createElement("p"); answer.textContent = reply.content; card.append(answer);
    }
    appendReplySources(card, run.retrievedKnowledge, run.id);
  }
  document.getElementById("conversationRunsCount").textContent = `最近 ${runs.length} 次：闲聊 ${counts.CHAT} · 政策 ${counts.CONSULT} · 支持 ${counts.RISK}`;
  if (!runs.length) container.textContent = "暂无对话运行记录";
}
