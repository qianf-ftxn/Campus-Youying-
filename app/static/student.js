const AUTH_KEY = "mindbridge.auth";

const state = {
  sessionId: null,
  sending: false,
  composing: false,
  loadingSession: false,
  sessions: [],
  nextOffset: null,
  profile: null,
  modelName: "mock"
};

const els = {
  serviceState: document.querySelector("#serviceState"),
  modelState: document.querySelector("#modelState"),
  activeAccount: document.querySelector("#activeAccount"),
  switchAccount: document.querySelector("#switchAccount"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  newSession: document.querySelector("#newSession"),
  sidebarNewSession: document.querySelector("#sidebarNewSession"),
  sessionList: document.querySelector("#sessionList"),
  sessionListStatus: document.querySelector("#sessionListStatus"),
  moreSessions: document.querySelector("#moreSessions"),
  conversationTitle: document.querySelector("#conversationTitle"),
  sessionBadge: document.querySelector("#sessionBadge"),
  studentGreeting: document.querySelector("#studentGreeting"),
  charCount: document.querySelector("#charCount")
};
let memoryStatusPoll = 0;

async function refreshMemoryStatus() {
  const generation = ++memoryStatusPoll;
  const label = document.querySelector("#memoryState");
  if (!label || !state.profile) return;
  for (let attempt = 0; attempt < 60 && generation === memoryStatusPoll; attempt++) {
    try {
      const body = await (await api("/api/memory/status")).json();
      if (generation !== memoryStatusPoll) return;
      const labels = {UPDATING: "记忆更新中", READY: "记忆已更新", DEGRADED: "记忆已保存，检索同步待完成", ERROR: "记忆更新待重试"};
      label.textContent = labels[body.state] || "";
      label.hidden = !label.textContent;
      if (!["UPDATING", "DEGRADED"].includes(body.state)) return;
      await new Promise((resolve) => setTimeout(resolve, 2000));
    } catch {
      if (generation !== memoryStatusPoll) return;
      label.hidden = false;
      label.textContent = "记忆状态暂不可用";
      return;
    }
  }
}

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
  const response = await fetch(path, { cache: "no-store", ...options, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response;
}

function activeSessionKey() {
  return `campus.activeSession:${state.profile.id}`;
}

function rememberSession(id) {
  state.sessionId = id;
  if (id) sessionStorage.setItem(activeSessionKey(), id);
  else sessionStorage.removeItem(activeSessionKey());
}

function updateBusy() {
  const busy = state.sending || state.loadingSession || !state.profile;
  els.sendButton.disabled = busy;
  els.newSession.disabled = busy;
  els.sidebarNewSession.disabled = busy;
  els.moreSessions.disabled = busy;
  els.sessionList.querySelectorAll("button").forEach((button) => { button.disabled = busy; });
}

function renderSessionList() {
  els.sessionList.replaceChildren();
  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item${session.sessionId === state.sessionId ? " active" : ""}`;
    if (session.sessionId === state.sessionId) button.setAttribute("aria-current", "page");
    const title = document.createElement("strong");
    title.textContent = session.title || "未命名对话";
    const time = document.createElement("small");
    const value = session.updatedAt.endsWith("Z") || /[+-]\d\d:\d\d$/.test(session.updatedAt)
      ? session.updatedAt : session.updatedAt + "Z";
    time.textContent = `${new Date(value).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })} · ${session.messageCount} 条消息`;
    button.append(title, time);
    button.addEventListener("click", () => openSession(session.sessionId));
    els.sessionList.append(button);
  }
  els.sessionListStatus.textContent = state.sessions.length ? "" : "还没有历史对话，发一条消息就会自动保存。";
  els.moreSessions.hidden = state.nextOffset === null;
  updateBusy();
}

async function loadSessions(append = false) {
  const offset = append ? state.nextOffset : 0;
  if (offset === null) return;
  try {
    const response = await api(`/api/chat/sessions?offset=${offset}&limit=50`);
    const body = await response.json();
    state.sessions = append ? [...state.sessions, ...body.items.filter((item) => !state.sessions.some((s) => s.sessionId === item.sessionId))] : body.items;
    state.nextOffset = body.nextOffset;
    renderSessionList();
  } catch {
    els.sessionListStatus.textContent = "历史对话暂时读取失败，请点击重试。";
    els.moreSessions.textContent = "重新读取";
    els.moreSessions.hidden = false;
  }
}

async function openSession(id) {
  if (state.sending || state.loadingSession) return;
  state.loadingSession = true;
  updateBusy();
  setPill(els.sessionBadge, "LOADING", "warn");
  try {
    const response = await api(`/api/chat/sessions/${encodeURIComponent(id)}`);
    const conversation = await response.json();
    rememberSession(conversation.sessionId);
    els.messages.replaceChildren();
    for (const message of conversation.messages) {
      const bubble = addMessage(message.role, message.content);
      if (message.completionStatus === "INTERRUPTED") {
        const note = document.createElement("span");
        note.className = "message-status";
        note.textContent = "回复中断";
        bubble.parentElement.querySelector(".message-role").append(note);
      }
    }
    els.conversationTitle.textContent = conversation.title;
    els.messageInput.value = "";
    updateCharCount();
    renderSessionList();
    setPill(els.sessionBadge, "READY");
    els.messages.scrollTop = els.messages.scrollHeight;
  } catch (error) {
    setPill(els.sessionBadge, "读取失败", "danger");
    els.sessionListStatus.textContent = "无法读取该对话，请重试或选择其他对话。";
  } finally {
    state.loadingSession = false;
    updateBusy();
  }
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

function studentDisplayName(profile) {
  return profile.displayName === "Demo Student" ? "Student" : (profile.displayName || profile.username || "Student");
}

function greetingForNow() {
  const hour = new Date().getHours();
  if (hour < 6) return "夜深了";
  if (hour < 11) return "早上好";
  if (hour < 14) return "中午好";
  if (hour < 18) return "下午好";
  return "晚上好";
}

function updateCharCount() {
  els.charCount.textContent = `${els.messageInput.value.length} / 1000`;
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
    if (isAdmin(profile)) {
      window.location.replace("/admin.html");
      return null;
    }
    state.profile = profile;
    const displayName = studentDisplayName(profile);
    els.activeAccount.textContent = displayName;
    els.studentGreeting.textContent = `${greetingForNow()}，${displayName}`;
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

function clearWelcome() {
  const empty = els.messages.querySelector(".empty");
  if (empty) empty.remove();
}

function addMessage(role, content) {
  clearWelcome();
  const row = document.createElement("article");
  row.className = `message ${role}`;
  row.innerHTML = `
    <div class="message-role">${role === "user" ? "我" : "校园有应"}</div>
    <div class="bubble"></div>
  `;
  row.querySelector(".bubble").textContent = content;
  els.messages.append(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return row.querySelector(".bubble");
}

function parseSse(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop();
  for (const part of parts) {
    const dataLine = part.split("\n").find((line) => line.startsWith("data: "));
    if (!dataLine) continue;
    onEvent(JSON.parse(dataLine.slice(6)));
  }
  return rest;
}

async function sendMessage(event) {
  event.preventDefault();
  if (state.sending || state.loadingSession || !state.profile) return;
  const message = els.messageInput.value.trim();
  if (!message) return;
  state.sending = true;
  updateBusy();
  setPill(els.sessionBadge, "THINKING", "warn");
  els.messageInput.value = "";
  updateCharCount();
  addMessage("user", message);
  const assistant = addMessage("assistant", "");
  let raw = "";

  try {
    const response = await api("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: state.sessionId,
        message,
        clientMessageId: `cm-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
      })
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamFailed = false;
    let receivedDone = false;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSse(buffer, (eventData) => {
        if (eventData.type === "meta") {
          rememberSession(eventData.sessionId);
          els.conversationTitle.textContent = state.sessions.find((s) => s.sessionId === state.sessionId)?.title || message.slice(0, 36);
        }
        if (eventData.type === "token") {
          raw += eventData.content || "";
          assistant.textContent = raw;
          els.messages.scrollTop = els.messages.scrollHeight;
        }
        if (eventData.type === "status" && !raw) assistant.textContent = eventData.message || "正在处理";
        if (eventData.type === "done") receivedDone = true;
        if (eventData.type === "error") {
          streamFailed = true;
          assistant.textContent = raw ? `${raw}\n\n${eventData.message || "本次回复未完成，请重试。"}` : (eventData.message || "本次回复未完成，请重试。");
          setPill(els.sessionBadge, "ERROR", "danger");
        }
      });
    }
    if (!streamFailed && receivedDone) setPill(els.sessionBadge, "DONE", "ok");
    else if (!streamFailed) throw new Error("连接在回复完成前中断，请重试。");
  } catch (error) {
    assistant.textContent = raw ? `${raw}\n\n回复未完成：${error.message}` : `发送失败：${error.message}`;
    setPill(els.sessionBadge, "ERROR", "danger");
  } finally {
    state.sending = false;
    updateBusy();
    await loadSessions();
    refreshMemoryStatus();
  }
}

function resetSession() {
  if (state.sending || state.loadingSession || !state.profile) return;
  rememberSession(null);
  els.conversationTitle.textContent = "新对话 · 从此刻最想了解的问题开始";
  renderSessionList();
  els.messages.innerHTML = `
    <div class="empty student-welcome">
      <span class="welcome-kicker">A FRESH START</span>
      <strong>新会话已经准备好</strong>
      <p>不用延续刚才的话题，你可以从此刻最想说的一件事重新开始。</p>
    </div>
  `;
  els.messageInput.value = "";
  updateCharCount();
  setPill(els.sessionBadge, "READY");
}

function logout() {
  clearAuth();
  window.location.assign("/");
}

document.querySelectorAll("[data-quick]").forEach((button) => {
  button.addEventListener("click", () => {
    els.messageInput.value = button.dataset.quick;
    updateCharCount();
    els.messageInput.focus();
  });
});
els.messageInput.addEventListener("input", updateCharCount);
els.messageInput.addEventListener("compositionstart", () => { state.composing = true; });
els.messageInput.addEventListener("compositionend", () => { state.composing = false; });
els.messageInput.addEventListener("keydown", (event) => {
  // 中文输入法中的 Enter 是确认候选词，不能提交。
  if (event.key !== "Enter" || event.shiftKey || event.isComposing || state.composing || event.keyCode === 229) return;
  event.preventDefault();
  if (event.repeat || state.sending || state.loadingSession || !els.messageInput.value.trim()) return;
  els.chatForm.requestSubmit();
});
els.chatForm.addEventListener("submit", sendMessage);
els.newSession.addEventListener("click", resetSession);
els.sidebarNewSession.addEventListener("click", resetSession);
els.moreSessions.addEventListener("click", () => {
  const retry = els.moreSessions.textContent === "重新读取";
  els.moreSessions.textContent = "加载更多";
  loadSessions(!retry);
});
els.switchAccount.addEventListener("click", logout);

checkHealth();
loadProfile().then((profile) => {
  if (profile) {
    updateBusy();
    refreshMemoryStatus();
    loadAgentStatus().catch(() => setPill(els.modelState, "模型状态暂不可用", "warn"));
    loadSessions().then(() => {
      const previous = sessionStorage.getItem(activeSessionKey());
      if (previous) openSession(previous);
    });
  }
});
updateCharCount();
updateBusy();
