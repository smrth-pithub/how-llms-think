// Next-token viewer: the model's top guesses for the next token, live as you type.
// The server returns raw log-probabilities for the top 100 tokens; temperature and
// softmax happen here, with the same softmax() the gesture demo uses.

topbar("/llm");

const $ = id => document.getElementById(id);
const ta = $("text"), backdrop = $("backdrop");
const K = 100;
const params = new URLSearchParams(location.search);
const AUTO_DELAY = +(params.get("delay") || 650);    // ms to show the bars before each auto pick
const AUTO_MAX = 80;

const PRESETS = [
  ["The cat sat on the", "warm-up: an easy guess"],
  ["The capital of France is", "a fact it has seen many times"],
  ["Once upon a time", "temperature: write with low, then high"],
  ["BITS Pilani is famous for its", "something local"],
  ["The best thing about Mondays is", "no right answer"],
  ["The first person to walk on Mars was", "hallucination: it still has to pick"],
  ["The winner of the 2031 Cricket World Cup was", "hallucination: it can't know this"],
];

let current = null;          // {text, tokens: [{id, token, bytes, logprob}], ms, model}
let probs = null;
let genStart = null;         // where the model's own text starts (for the highlight)
let lastPick = null;         // [start, end] of the newest token it wrote
let history = [];
let decoder = new TextDecoder("utf-8");
let auto = false;

const bars = new Bars($("bars"), {
  rows: 10,
  maxFont: 34,
  formatLabel: (label, item) => tokenNode(item.tok),
  onClick: item => pick(item.index),
});
const temp = setupTemperature($("temp"), () => { if (current) applyTemp(); });

function isSpecial(tok) { return /^<\|.*\|>$/.test(tok); }

function tokenNode(t) {
  if (t.token.includes("�") && t.bytes && t.bytes.length) {   // part of a character
    const s = document.createElement("span");
    s.className = "ws";
    s.textContent = t.bytes.map(b => b.toString(16).padStart(2, "0")).join(" ");
    return s;
  }
  return showToken(t.token);
}

// ---------- predictions ----------
let inflight = null, inflightText = null, again = false;
function predict() {
  const text = ta.value;
  if (!text) { current = null; probs = null; bars.update([]); return Promise.resolve(null); }
  if (current && current.text === text) return Promise.resolve(current);
  if (inflight && inflightText === text) return inflight;
  if (inflight) { again = true; return inflight.then(() => predict()); }
  inflightText = text;
  inflight = postJSON("/api/llm/next", { text, k: K })
    .then(d => {
      hideOverlay();
      current = { text, ...d };
      showLatency(d.ms);
      if (ta.value === text) applyTemp();
      return current;
    })
    .catch(e => {
      if (e.status === 503) {
        showOverlay(e.data && e.data.state === "error" ? "Model failed to load: " + e.data.error : "Loading the model…");
        return new Promise(r => setTimeout(r, 1500)).then(() => null);
      }
      showOverlay("Error: " + e.message);
      return null;
    })
    .finally(() => {
      inflight = null;
      if (again || (current && current.text !== ta.value)) { again = false; predict(); }
    });
  return inflight;
}

function applyTemp() {
  probs = softmax(current.tokens.map(t => t.logprob), temp.value);
  bars.update(current.tokens.map((t, i) => ({ key: t.id, label: t.token, value: probs[i], tok: t, index: i })),
              { forceOrder: true });
}

function showLatency(ms) {
  const el = $("lat");
  el.textContent = `${Math.round(ms)} ms`;
  el.classList.toggle("slow", ms > 500);
}

// ---------- picking: the loop ----------
async function pick(index = null) {
  let cur = await predict();
  if (!cur && ta.value) cur = await predict();
  if (!cur || !cur.tokens.length) return false;
  const p = softmax(cur.tokens.map(t => t.logprob), temp.value);
  const i = index ?? sample(p);
  const tok = cur.tokens[i];
  const rank = [...p].sort((a, b) => b - a).indexOf(p[i]) + 1;
  showPicked(tok, p[i], rank);
  bars.flash(tok.id);
  if (isSpecial(tok.token)) { stopAuto(); toast("The model picked an end-of-text token: it wants to stop here"); return false; }
  history.push({ text: ta.value, genStart, lastPick });
  const start = ta.value.length;
  if (genStart === null) genStart = start;
  ta.value += decoder.decode(new Uint8Array(tok.bytes || []), { stream: true }) || (tok.bytes ? "" : tok.token);
  lastPick = [start, ta.value.length];
  afterTextChange(true);
  return true;
}

function showPicked(tok, prob, rank) {
  const el = $("picked");
  el.innerHTML = "";
  el.append("picked ");
  const b = document.createElement("b"); b.appendChild(tokenNode(tok)); el.append(b);
  el.append(` · ${fmtPct(prob)} · #${rank}${rank > 10 ? " (not even in the top 10)" : ""}`);
}

async function autoWrite() {
  if (auto) { stopAuto(); return; }
  auto = true;
  $("autoBtn").textContent = "Stop ■";
  $("autoBtn").classList.add("on");
  for (let n = 0; auto && n < AUTO_MAX; n++) {
    await predict();
    await new Promise(r => setTimeout(r, AUTO_DELAY));
    if (!auto) break;
    if (!(await pick())) break;
  }
  stopAuto();
}
function stopAuto() {
  auto = false;
  $("autoBtn").textContent = "Auto-write ▶";
  $("autoBtn").classList.remove("on");
}

function undo() {
  const h = history.pop();
  if (!h) { toast("Nothing to undo"); return; }
  ta.value = h.text; genStart = h.genStart; lastPick = h.lastPick;
  decoder = new TextDecoder("utf-8");
  afterTextChange(true);
}

// ---------- text box ----------
function esc(s) { return s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function updateBackdrop() {
  const t = ta.value;
  let html = esc(t);
  if (genStart !== null && genStart <= t.length) {
    const ls = lastPick ? Math.max(genStart, lastPick[0]) : t.length;
    const a = t.slice(0, genStart), b = t.slice(genStart, ls), c = t.slice(ls);
    html = esc(a) + (b ? `<mark>${esc(b)}</mark>` : "") + (c ? `<mark class="last">${esc(c)}</mark>` : "");
  }
  backdrop.innerHTML = html + "\n";
  backdrop.scrollTop = ta.scrollTop;
}

let tokTimer;
function afterTextChange(scrollToEnd) {
  updateBackdrop();
  if (scrollToEnd) { ta.scrollTop = ta.scrollHeight; backdrop.scrollTop = ta.scrollTop; }
  predict();
  clearTimeout(tokTimer);
  tokTimer = setTimeout(tokenizeStrip, 150);
}

async function tokenizeStrip() {
  const text = ta.value;
  if (!text) { $("chips").innerHTML = ""; $("count").textContent = ""; return; }
  try {
    const d = await postJSON("/api/llm/tokenize", { text });
    if (ta.value !== text) return;
    const chips = d.tokens.slice(-40).map(t => {
      const c = document.createElement("span"); c.className = "chip";
      if (Array.isArray(t.piece)) { c.classList.add("bytes"); c.textContent = t.piece.map(b => b.toString(16).padStart(2, "0")).join(" "); }
      else c.appendChild(showToken(t.piece));
      return c;
    });
    $("chips").replaceChildren(...chips);
    $("count").textContent = `What the model reads (${d.tokens.length} tokens):`;
  } catch (e) { /* the bars will show any error */ }
}

ta.addEventListener("input", () => {
  genStart = null; lastPick = null; decoder = new TextDecoder("utf-8");
  afterTextChange(false);
});
ta.addEventListener("scroll", () => { backdrop.scrollTop = ta.scrollTop; });
ta.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey) { e.preventDefault(); pick(); }
  else if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); autoWrite(); }
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { stopAuto(); return; }
  if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "Enter") { e.preventDefault(); pick(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); temp.nudge(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); temp.nudge(1); }
  else if (e.key === "1" || e.key === "0") temp.set(1);
});

$("pickBtn").addEventListener("click", () => pick());
$("autoBtn").addEventListener("click", autoWrite);
$("undoBtn").addEventListener("click", undo);

// ---------- presets and model ----------
$("preset").innerHTML = `<option value="">example prompts…</option>` +
  PRESETS.map(([p, why], i) => `<option value="${i}">${esc(p)}${why ? "  ·  " + esc(why) : ""}</option>`).join("");
$("preset").addEventListener("change", e => {
  if (e.target.value === "") return;
  stopAuto();
  ta.value = PRESETS[+e.target.value][0];
  genStart = null; lastPick = null; history = []; decoder = new TextDecoder("utf-8");
  e.target.value = "";
  afterTextChange(true);
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
});

function showOverlay(t) { $("overlay").textContent = t; $("overlay").classList.remove("hidden"); }
function hideOverlay() { $("overlay").classList.add("hidden"); }

async function loadModel(name) {
  showOverlay(`Loading ${name || "the model"} onto the GPU… (about 10 s)`);
  current = null;
  try {
    const s = await postJSON("/api/llm/load", { model: name || null });
    fillModels(s);
    if (s.state !== "ready") { showOverlay("Model failed to load: " + (s.error || s.state)); return; }
    hideOverlay();
    afterTextChange(false);
  } catch (e) { showOverlay("Model failed to load: " + e.message); }
}

function fillModels(s) {
  const sel = $("model");
  sel.innerHTML = s.models.map(m => `<option value="${m.name}" ${m.available ? "" : "disabled"}>${m.name}${m.available ? "" : " (missing)"}</option>`).join("");
  if (s.model) sel.value = s.model;
}
$("model").addEventListener("change", e => { stopAuto(); loadModel(e.target.value); });

fetch("/api/llm/status").then(r => r.json()).then(s => { fillModels(s); loadModel(s.model); });
updateBackdrop();
ta.focus();
ta.setSelectionRange(ta.value.length, ta.value.length);
