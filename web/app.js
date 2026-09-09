"use strict";

// The browser side is deliberately dumb: it renders whatever /api/state returns
// and posts user intents back. All chess truth stays on the server.

let state = null;
const history = []; // {role: 'user'|'tutor'|'error', text}

const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const opts = { method: body === undefined ? "GET" : "POST" };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  return res.json();
}

// ---- screen routing --------------------------------------------------------

function show(id) {
  for (const el of document.querySelectorAll(".screen")) {
    el.hidden = el.id !== id;
  }
}

function render() {
  renderBadges();
  if (!state || !state.loaded) {
    show("loader");
    return;
  }
  if (!state.player_color) {
    $("color-meta").textContent =
      `${state.white} (White) vs ${state.black} (Black) — result ${state.result}.`;
    show("color-select");
    return;
  }
  show("review");
  renderReview();
}

function renderBadges() {
  const badges = $("badges");
  if (!state) { badges.innerHTML = ""; return; }
  const parts = [];
  if (!state.api_key_present) parts.push('<span class="badge warn">no API key — chat disabled</span>');
  if (!state.engine_available) parts.push('<span class="badge warn">no engine — analysis disabled</span>');
  badges.innerHTML = parts.join("");
}

// ---- review screen ---------------------------------------------------------

function renderReview() {
  $("board").innerHTML = state.board_svg || `<div class="error">${state.svg_error || "No board."}</div>`;

  const ply = state.current_ply;
  const total = state.moves.length;
  $("board-info").textContent =
    `Ply ${ply} / ${total} · ${state.side_to_move} to move` +
    (state.last_move_san ? ` · last: ${state.last_move_san}` : "");

  renderMoveList();
}

function renderMoveList() {
  const list = $("move-list");
  const moves = state.moves;
  let html = "";
  for (let i = 0; i < moves.length; i += 2) {
    const w = moves[i];
    const b = moves[i + 1];
    const num = w.move_number;
    html += `<div class="move-row"><span class="num">${num}.</span>`;
    html += moveCell(w);
    html += b ? moveCell(b) : "<span></span>";
    html += "</div>";
  }
  list.innerHTML = html || '<span class="muted">No moves.</span>';
  for (const el of list.querySelectorAll(".move")) {
    el.addEventListener("click", () => goto(parseInt(el.dataset.ply, 10)));
  }
  const sel = list.querySelector(".move.selected");
  if (sel) sel.scrollIntoView({ block: "nearest" });
}

function moveCell(m) {
  const selected = m.ply === state.current_ply ? " selected" : "";
  return `<span class="move${selected}" data-ply="${m.ply}">${m.san}</span>`;
}

// ---- actions ---------------------------------------------------------------

async function loadGame(pgn, sample) {
  const err = $("loader-error");
  err.hidden = true;
  const result = await api("/api/load-game", sample ? { sample: true } : { pgn });
  if (!result.ok) {
    err.textContent = result.error;
    err.hidden = false;
    return;
  }
  state = result;
  render();
}

async function setColor(color) {
  const result = await api("/api/set-color", { color });
  if (result.ok) { state = result; render(); }
}

async function goto(ply) {
  const clamped = Math.max(0, Math.min(ply, state.moves.length));
  const result = await api("/api/goto", { ply: clamped });
  if (result.ok) { state = result; render(); }
}

function nav(which) {
  const ply = state.current_ply;
  const total = state.moves.length;
  if (which === "start") goto(0);
  else if (which === "prev") goto(ply - 1);
  else if (which === "next") goto(ply + 1);
  else if (which === "end") goto(total);
}

// ---- chat ------------------------------------------------------------------

function pushMessage(role, text) {
  history.push({ role, text });
  renderChat();
}

function renderChat(loading) {
  const box = $("chat-history");
  let html = history
    .map((m) => `<div class="msg ${m.role}">${escapeHtml(m.text)}</div>`)
    .join("");
  if (loading) html += `<div class="msg tutor loading">thinking…</div>`;
  box.innerHTML = html;
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function explainQuestion() {
  // "this move" = the move that produced the position the user is viewing;
  // at the start, fall back to explaining the game's first move.
  const moves = state.moves;
  if (!moves.length) return "Explain the opening.";
  const idx = state.current_ply > 0 ? state.current_ply - 1 : 0;
  const m = moves[idx];
  const sep = m.side === "white" ? "." : "...";
  return `Explain ${m.move_number}${sep}${m.san}`;
}

async function ask(question) {
  if (!question.trim()) return;
  pushMessage("user", question);
  $("send-btn").disabled = true;
  renderChat(true);
  let result;
  try {
    result = await api("/api/ask", { question });
  } catch (e) {
    result = { ok: false, error: `Network error: ${e}` };
  }
  $("send-btn").disabled = false;
  if (result.ok) pushMessage("tutor", result.answer);
  else pushMessage("error", result.error);
}

// ---- wiring ----------------------------------------------------------------

function wire() {
  $("load-game").addEventListener("click", () => loadGame($("pgn-input").value, false));
  $("load-sample").addEventListener("click", () => loadGame(null, true));
  $("pick-white").addEventListener("click", () => setColor("white"));
  $("pick-black").addEventListener("click", () => setColor("black"));
  $("back-to-loader").addEventListener("click", () => { state = null; show("loader"); });

  for (const btn of document.querySelectorAll(".nav button")) {
    btn.addEventListener("click", () => nav(btn.dataset.nav));
  }
  for (const btn of $("quick-questions").querySelectorAll("button")) {
    btn.addEventListener("click", () => {
      const q = btn.dataset.q === "__explain__" ? explainQuestion() : btn.dataset.q;
      ask(q);
    });
  }
  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("chat-text").value;
    $("chat-text").value = "";
    ask(text);
  });

  document.addEventListener("keydown", (e) => {
    if (!state || !state.loaded || !state.player_color) return;
    if (document.activeElement && document.activeElement.tagName === "INPUT") return;
    if (e.key === "ArrowLeft") nav("prev");
    else if (e.key === "ArrowRight") nav("next");
  });
}

async function init() {
  wire();
  state = await api("/api/state");
  render();
}

init();
