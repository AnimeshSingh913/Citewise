// Citewise web client. No framework, no build step.

const $ = (id) => document.getElementById(id);
const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* private mode */ } },
};

const state = {
  access: store.get("cw.access"),
  refresh: store.get("cw.refresh"),
  user: null,
  conversations: [], convTotal: 0,
  documents: [], docTotal: 0,
  activeConv: null,
  selectedDocs: new Set(),
  streaming: false,
  pollTimer: null,
  registerMode: false,
};

// ---------- API ----------

class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : detail?.message || `Request failed (${status})`);
    this.status = status;
    this.detail = detail;
  }
}

function setTokens(access, refresh) {
  state.access = access; state.refresh = refresh;
  store.set("cw.access", access); store.set("cw.refresh", refresh);
}

let refreshing = null;
async function refreshTokens() {
  if (!state.refresh) return false;
  refreshing ??= fetch("/api/auth/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: state.refresh }),
  }).then(async (r) => {
    if (!r.ok) { setTokens(null, null); return false; }
    const t = await r.json();
    setTokens(t.access_token, t.refresh_token);
    return true;
  }).finally(() => { refreshing = null; });
  return refreshing;
}

async function request(path, { method = "GET", body, raw = false, retry = true } = {}) {
  const headers = {};
  if (state.access) headers.Authorization = `Bearer ${state.access}`;
  let payload = body;
  if (body && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(path, { method, headers, body: payload });
  if (res.status === 401 && retry && (await refreshTokens())) {
    return request(path, { method, body, raw, retry: false });
  }
  if (res.status === 401) { showAuth(); throw new ApiError(401, "Please sign in again"); }
  if (raw) {
    if (!res.ok) throw new ApiError(res.status, (await res.json().catch(() => ({}))).detail);
    return res;
  }
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, data.detail);
  return data;
}

// ---------- Utilities ----------

function toast(message, ms = 3500) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { el.hidden = true; }, ms);
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(c));
  return node;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Small, safe Markdown subset. Input is escaped first, so only the tags we add exist.
function renderMarkdown(text) {
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[(\d{1,3})\]/g, '<button type="button" class="cite" data-n="$1">$1</button>');
  const blocks = escapeHtml(text.trim()).split(/\n{2,}/);
  return blocks.map((block) => {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) {
      return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*•]\s+/, ""))}</li>`).join("")}</ul>`;
    }
    if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
      return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`;
    }
    if (lines.every((l) => /^&gt;\s?/.test(l))) {
      return `<blockquote>${inline(lines.map((l) => l.replace(/^&gt;\s?/, "")).join("<br>"))}</blockquote>`;
    }
    const heading = block.match(/^#{1,6}\s+(.*)$/);
    if (heading && lines.length === 1) return `<p><strong>${inline(heading[1])}</strong></p>`;
    return `<p>${inline(lines.join("<br>"))}</p>`;
  }).join("");
}

function sourceLabel(s) {
  if (s.page_number) return `${s.filename} · p. ${s.page_number}`;
  return s.heading ? `${s.filename} · ${s.heading}` : s.filename;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// ---------- Auth ----------

function showAuth() {
  $("app-view").hidden = true;
  $("auth-view").hidden = false;
  stopPolling();
  $("auth-email").focus();
}

function setAuthMode(register) {
  state.registerMode = register;
  $("auth-submit").textContent = register ? "Create account" : "Sign in";
  $("auth-toggle").textContent = register ? "Have an account? Sign in" : "New here? Create an account";
  $("auth-password").autocomplete = register ? "new-password" : "current-password";
  $("auth-error").hidden = true;
}

async function submitAuth(event) {
  event.preventDefault();
  const email = $("auth-email").value.trim();
  const password = $("auth-password").value;
  const err = $("auth-error");
  err.hidden = true;
  if (password.length < 8) { err.textContent = "Password must be at least 8 characters."; err.hidden = false; return; }
  $("auth-submit").disabled = true;
  try {
    if (state.registerMode) await request("/api/auth/register", { method: "POST", body: { email, password }, retry: false });
    const t = await request("/api/auth/login", { method: "POST", body: { email, password }, retry: false });
    setTokens(t.access_token, t.refresh_token);
    $("auth-password").value = "";
    await boot();
  } catch (e) {
    err.textContent = e.status === 422 ? "Enter a valid email address." : e.message;
    err.hidden = false;
  } finally {
    $("auth-submit").disabled = false;
  }
}

async function logout() {
  if (state.refresh) await fetch("/api/auth/logout", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ refresh_token: state.refresh }),
  }).catch(() => {});
  setTokens(null, null);
  state.activeConv = null;
  showAuth();
}

// ---------- Conversations ----------

async function loadConversations(append = false) {
  const offset = append ? state.conversations.length : 0;
  const page = await request(`/api/conversations?limit=30&offset=${offset}`);
  state.conversations = append ? state.conversations.concat(page.items) : page.items;
  state.convTotal = page.total;
  renderConversations();
}

function renderConversations() {
  const list = $("conversation-list");
  list.replaceChildren(...state.conversations.map((c) => el("li", {},
    el("div", { class: `conv${c.id === state.activeConv?.id ? " active" : ""}`, onclick: () => openConversation(c) },
      el("span", { class: "conv-title", title: c.title || "Untitled" }, c.title || "Untitled"),
      el("button", { class: "icon-btn", title: "Rename", "aria-label": "Rename", onclick: (e) => { e.stopPropagation(); renameConversation(c); } }, "✎"),
      el("button", { class: "icon-btn", title: "Delete", "aria-label": "Delete", onclick: (e) => { e.stopPropagation(); deleteConversation(c); } }, "🗑"),
    ))));
  if (!state.conversations.length) list.append(el("li", { class: "muted small" }, "No chats yet"));
  $("more-conversations").hidden = state.conversations.length >= state.convTotal;
}

async function openConversation(conv) {
  if (state.streaming) return;
  state.activeConv = conv;
  $("chat-title").textContent = conv.title || "Untitled";
  renderConversations();
  closeSidebar();
  const thread = $("thread");
  thread.replaceChildren();
  let offset = 0, total = Infinity;
  while (offset < total) {
    const page = await request(`/api/conversations/${conv.id}/messages?limit=100&offset=${offset}`);
    total = page.total;
    offset += page.items.length;
    for (const m of page.items) thread.append(m.role === "user" ? userMessage(m.content) : assistantMessage(m.content, m.sources));
    if (!page.items.length) break;
  }
  if (!thread.children.length) thread.append(emptyState());
  thread.scrollTop = thread.scrollHeight;
}

function newChat() {
  if (state.streaming) return;
  state.activeConv = null;
  $("chat-title").textContent = "New chat";
  $("thread").replaceChildren(emptyState());
  renderConversations();
  closeSidebar();
  $("question").focus();
}

async function renameConversation(conv) {
  const title = prompt("Rename chat", conv.title || "");
  if (!title?.trim()) return;
  const updated = await request(`/api/conversations/${conv.id}`, { method: "PATCH", body: { title: title.trim() } });
  Object.assign(conv, updated);
  if (state.activeConv?.id === conv.id) $("chat-title").textContent = updated.title;
  renderConversations();
}

async function deleteConversation(conv) {
  if (!confirm(`Delete “${conv.title || "Untitled"}”? This can't be undone.`)) return;
  await request(`/api/conversations/${conv.id}`, { method: "DELETE" });
  if (state.activeConv?.id === conv.id) newChat();
  await loadConversations();
}

// ---------- Documents ----------

async function loadDocuments(append = false) {
  const offset = append ? state.documents.length : 0;
  const page = await request(`/api/documents?limit=50&offset=${offset}`);
  state.documents = append ? state.documents.concat(page.items) : page.items;
  state.docTotal = page.total;
  const ids = new Set(state.documents.map((d) => d.id));
  for (const id of state.selectedDocs) if (!ids.has(id)) state.selectedDocs.delete(id);
  renderDocuments();
  schedulePolling();
}

function renderDocuments() {
  const list = $("document-list");
  list.replaceChildren(...state.documents.map((d) => {
    const busy = d.status === "PENDING" || d.status === "PROCESSING";
    const sub = d.status === "READY"
      ? [d.page_count ? `${d.page_count} pages` : null, formatSize(d.size_bytes)].filter(Boolean).join(" · ")
      : d.status === "FAILED" ? (d.error || "Processing failed") : busy ? "Processing…" : d.status;
    const checkbox = el("input", {
      type: "checkbox", title: "Only search this document", "aria-label": `Only search ${d.filename}`,
      disabled: d.status !== "READY", checked: state.selectedDocs.has(d.id),
      onchange: (e) => { e.target.checked ? state.selectedDocs.add(d.id) : state.selectedDocs.delete(d.id); renderScope(); },
    });
    return el("li", {}, el("div", { class: "doc" },
      checkbox,
      el("div", { class: "doc-main" },
        el("div", { class: "doc-name", title: d.filename }, d.filename),
        el("div", { class: "doc-sub", title: sub }, sub)),
      el("div", { class: "doc-actions" },
        el("span", { class: `badge ${d.status}` }, d.status === "READY" ? "Ready" : d.status === "FAILED" ? "Failed" : "…"),
        d.status === "FAILED" || d.status === "READY"
          ? el("button", { class: "icon-btn", title: d.status === "FAILED" ? "Retry" : "Re-index", "aria-label": "Reprocess", onclick: () => reprocess(d) }, "↻")
          : null,
        el("button", { class: "icon-btn", title: "Delete", "aria-label": `Delete ${d.filename}`, onclick: () => deleteDocument(d) }, "🗑"),
      )));
  }));
  if (!state.documents.length) list.append(el("li", { class: "muted small" }, "Upload a document to get started"));
  $("doc-count").textContent = state.docTotal ? `(${state.docTotal})` : "";
  $("more-documents").hidden = state.documents.length >= state.docTotal;
  renderScope();
}

function renderScope() {
  const pill = $("scope-pill");
  const n = state.selectedDocs.size;
  pill.hidden = n === 0;
  pill.textContent = n === 1 ? "Searching 1 document" : `Searching ${n} documents`;
}

async function uploadFiles(files) {
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      await request("/api/documents", { method: "POST", body: form });
      toast(`Uploaded ${file.name}. Processing…`);
    } catch (e) {
      toast(e.status === 413 ? `${file.name} is too large` : `${file.name}: ${e.message}`, 5000);
    }
  }
  await loadDocuments();
}

async function reprocess(doc) {
  try {
    await request(`/api/documents/${doc.id}/reprocess`, { method: "POST" });
    await loadDocuments();
  } catch (e) { toast(e.message); }
}

async function deleteDocument(doc) {
  if (!confirm(`Delete “${doc.filename}” and its index? This can't be undone.`)) return;
  await request(`/api/documents/${doc.id}`, { method: "DELETE" });
  state.selectedDocs.delete(doc.id);
  await loadDocuments();
}

function schedulePolling() {
  stopPolling();
  if (state.documents.some((d) => d.status === "PENDING" || d.status === "PROCESSING")) {
    state.pollTimer = setTimeout(() => loadDocuments().catch(() => {}), 2000);
  }
}

function stopPolling() { clearTimeout(state.pollTimer); state.pollTimer = null; }

// ---------- Chat ----------

function emptyState() {
  return el("div", { class: "empty" },
    el("h2", {}, "Ask anything about your documents"),
    el("p", { class: "muted" }, "Every answer links each claim to the passage and page it came from."));
}

function userMessage(text) {
  return el("div", { class: "msg msg-user" }, el("div", { class: "bubble" }, text));
}

function sourcesRow(sources) {
  return el("div", { class: "sources" }, sources.map((s) =>
    el("button", { type: "button", class: "source-chip", onclick: () => openSource(s) },
      el("span", { class: "n" }, String(s.n)), el("span", { class: "label" }, sourceLabel(s)))));
}

function assistantMessage(text, sources = []) {
  const bubble = el("div", { class: "bubble" });
  bubble.innerHTML = renderMarkdown(text);
  wireCitations(bubble, sources);
  return el("div", { class: "msg msg-assistant" }, bubble, sources.length ? sourcesRow(sources) : null);
}

function wireCitations(container, sources) {
  const byN = new Map(sources.map((s) => [String(s.n), s]));
  for (const btn of container.querySelectorAll(".cite")) {
    const s = byN.get(btn.dataset.n);
    if (!s) { btn.replaceWith(document.createTextNode(`[${btn.dataset.n}]`)); continue; }
    btn.title = sourceLabel(s);
    btn.onclick = () => openSource(s);
  }
}

async function* readSse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message", data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      yield { event, data: data ? JSON.parse(data) : null };
    }
  }
}

async function ask(event) {
  event.preventDefault();
  const input = $("question");
  const question = input.value.trim();
  if (!question || state.streaming) return;

  state.streaming = true;
  $("send").disabled = true;
  input.value = "";
  autosize();

  const thread = $("thread");
  thread.querySelector(".empty")?.remove();
  thread.append(userMessage(question));
  const status = el("div", { class: "status-line" }, el("span", { class: "dots" }, "Searching your documents"));
  const bubble = el("div", { class: "bubble caret" });
  const msg = el("div", { class: "msg msg-assistant" }, status, bubble);
  thread.append(msg);
  thread.scrollTop = thread.scrollHeight;

  let text = "";
  let sources = [];
  let frame = 0;
  const paint = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      bubble.innerHTML = renderMarkdown(text);
      wireCitations(bubble, sources);
      const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
      if (nearBottom) thread.scrollTop = thread.scrollHeight;
    });
  };

  try {
    if (!state.activeConv) {
      state.activeConv = await request("/api/conversations", { method: "POST", body: {} });
    }
    const body = { content: question };
    if (state.selectedDocs.size) body.document_ids = [...state.selectedDocs];
    const res = await request(`/api/conversations/${state.activeConv.id}/messages`, { method: "POST", body, raw: true });

    for await (const { event: name, data } of readSse(res)) {
      if (name === "retrieved") {
        status.replaceChildren(el("span", { class: "dots" },
          data.count ? `Reading ${data.count} passage${data.count === 1 ? "" : "s"}` : "No matching passages"));
      } else if (name === "token") {
        status.remove();
        text += data;
        paint();
      } else if (name === "cite") {
        sources = data.sources;
        text += data.numbers.map((n) => `[${n}]`).join("");
        paint();
      } else if (name === "reset") {
        text = ""; sources = []; paint();
      } else if (name === "error") {
        status.remove();
        msg.classList.add("msg-error");
        bubble.textContent = data.message;
      } else if (name === "done") {
        sources = data.sources || sources;
        paint();
        if (sources.length) msg.append(sourcesRow(sources));
      }
    }
  } catch (e) {
    status.remove();
    msg.classList.add("msg-error");
    bubble.textContent = e.message;
  } finally {
    bubble.classList.remove("caret");
    state.streaming = false;
    $("send").disabled = false;
    input.focus();
    loadConversations().then(() => {
      const fresh = state.conversations.find((c) => c.id === state.activeConv?.id);
      if (fresh) { state.activeConv = fresh; $("chat-title").textContent = fresh.title || "Untitled"; renderConversations(); }
    }).catch(() => {});
  }
}

// ---------- Source drawer ----------

let currentSource = null;

function openSource(s) {
  currentSource = s;
  $("source-file").textContent = s.filename;
  $("source-meta").textContent = [s.page_number ? `Page ${s.page_number}` : null, s.heading].filter(Boolean).join(" · ");
  $("source-text").textContent = s.excerpt + (s.excerpt.length >= 280 ? "…" : "");
  $("open-source").hidden = false;
  $("source-drawer").classList.add("open");
  $("source-drawer").setAttribute("aria-hidden", "false");
}

function closeSource() {
  $("source-drawer").classList.remove("open");
  $("source-drawer").setAttribute("aria-hidden", "true");
}

async function openSourceDocument() {
  if (!currentSource) return;
  const tab = window.open("", "_blank");
  try {
    const res = await request(`/api/documents/${currentSource.document_id}/file`, { raw: true });
    const url = URL.createObjectURL(await res.blob());
    const target = currentSource.page_number ? `${url}#page=${currentSource.page_number}` : url;
    if (tab) tab.location.href = target; else window.open(target, "_blank");
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  } catch (e) {
    tab?.close();
    toast(e.status === 404 ? "That document has been deleted." : e.message);
  }
}

// ---------- Layout ----------

function openSidebar() { $("sidebar").classList.add("open"); }
function closeSidebar() { $("sidebar").classList.remove("open"); }

function autosize() {
  const t = $("question");
  t.style.height = "auto";
  t.style.height = `${Math.min(t.scrollHeight, 200)}px`;
}

// ---------- Boot ----------

async function boot() {
  if (!state.access && !state.refresh) return showAuth();
  try {
    state.user = await request("/api/auth/me");
  } catch {
    return showAuth();
  }
  $("auth-view").hidden = true;
  $("app-view").hidden = false;
  $("user-email").textContent = state.user.email;
  newChat();
  await Promise.all([loadConversations(), loadDocuments()]);
}

function wire() {
  $("auth-form").addEventListener("submit", submitAuth);
  $("auth-toggle").addEventListener("click", () => setAuthMode(!state.registerMode));
  $("logout").addEventListener("click", logout);
  $("new-chat").addEventListener("click", newChat);
  $("more-conversations").addEventListener("click", () => loadConversations(true));
  $("more-documents").addEventListener("click", () => loadDocuments(true));
  $("composer").addEventListener("submit", ask);
  $("open-sidebar").addEventListener("click", openSidebar);
  $("close-sidebar").addEventListener("click", closeSidebar);
  $("close-drawer").addEventListener("click", closeSource);
  $("open-source").addEventListener("click", openSourceDocument);

  const q = $("question");
  q.addEventListener("input", autosize);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSource(); });
  // Pick up documents uploaded from another tab or via the API.
  window.addEventListener("focus", () => { if (state.user && !state.pollTimer) loadDocuments().catch(() => {}); });

  const zone = $("dropzone");
  const input = $("file-input");
  input.addEventListener("change", () => { uploadFiles([...input.files]); input.value = ""; });
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
  });
}

wire();
boot();
