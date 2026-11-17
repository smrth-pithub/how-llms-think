// Gesture demo: the server sends raw scores (logits) for every camera frame.
// Here: softmax with the audience's temperature -> smooth over a few frames -> ranked bars.

topbar("/gesture");

const $ = id => document.getElementById(id);
const canvas = $("cam"), ctx = canvas.getContext("2d");
const params = new URLSearchParams(location.search);
const SMOOTH_SECONDS = +(params.get("smooth") || 0.15);   // bigger = steadier, slower bars

let labels = [];
let lastLogits = null, lastFrameAt = 0;
let probs = null;          // smoothed percentages shown on screen
let scores = null;         // smoothed raw scores (for "Show the numbers")
let mirror = store("mirror") !== "0";
let showNumbers = false;
let info = null;

const bars = new Bars($("bars"), { rows: 7, reorderMs: 300 });
const temp = setupTemperature($("temp"), onTemperature);

$("code").textContent = softmax.toString();

function onTemperature(T) {
  if (!lastLogits) return;
  // if frames are flowing, the next frame picks the new temperature up;
  // if not (paused camera), redraw now so the slider still feels live
  if (performance.now() - lastFrameAt > 250) {
    probs = softmax(lastLogits, T);
    render();
  }
}

function blend(prev, next, a) { return prev ? prev.map((v, i) => v + a * (next[i] - v)) : next.slice(); }

let lastT = performance.now();
function onState(s) {
  const now = performance.now();
  const dt = Math.min(0.5, (now - lastT) / 1000);
  lastT = now;
  updateMessage(s);
  $("stats").textContent = s.status === "running"
    ? `${s.fps.toFixed(0)} fps · pose + model ${s.infer_ms.toFixed(0)} ms` : "";
  if (s.model && s.model.labels) labels = s.model.labels;

  if (s.logits) {
    lastLogits = s.logits;
    lastFrameAt = now;
    const p = softmax(s.logits, temp.value);
    const a = 1 - Math.exp(-dt / Math.max(SMOOTH_SECONDS, 1e-3));   // smoothing: average the last few frames
    probs = blend(probs, p, a);
    scores = blend(scores, s.logits, a);
    $("bars").classList.remove("dim");
    render();
  } else {
    $("bars").classList.toggle("dim", !!probs);
  }
  if (showNumbers) updateFlow(s);
}

function render() {
  if (!probs) return;
  bars.update(labels.map((label, i) => ({
    key: label, label, value: probs[i],
    extra: scores ? "score " + scores[i].toFixed(1) : "",
  })));
}

function updateMessage(s) {
  const msg = $("msg");
  let text = "";
  if (s.status === "error") text = s.message;
  else if (s.status === "starting" || s.status === "stopped") text = s.message || "Starting camera…";
  else if (s.status === "running" && s.message) text = s.message;
  else if (s.status === "running" && !s.has_person) text = "Step into view: stand 2–3 m from the camera";
  msg.textContent = text;
  msg.classList.toggle("hidden", !text);

  const trained = s.model && s.model.trained;
  const empty = $("barsEmpty");
  if (!trained) {
    empty.innerHTML = 'No gesture model yet.<br>Record a few people on <a href="/record">Record &amp; train</a>, then press Train.';
  }
  empty.classList.toggle("hidden", !!trained);
  setModelInfo(s.model);
}

let modelInfoKey = "";
function setModelInfo(m) {
  const key = JSON.stringify(m && m.report || null);
  if (key === modelInfoKey) return;
  modelInfoKey = key;
  const r = m && m.report;
  if (!r || !r.people) { $("modelInfo").textContent = ""; return; }
  let t = `Trained on ${r.people.length} ${r.people.length === 1 ? "person" : "people"}`;
  if (r.extra && r.extra.people) t += ` + ${r.extra.people} from MPOSE2021`;
  if (r.heldout && r.heldout.kind === "unseen people") t += ` · right ${r.heldout.overall.toFixed(0)}% of the time on people it never saw`;
  $("modelInfo").textContent = t;
}

function updateFlow(s) {
  const w = s.wrist ? `right wrist x <b>${s.wrist[0].toFixed(2)}</b> y <b>${s.wrist[1].toFixed(2)}</b>` : "nobody in view";
  const n = info ? info.n_points : 17;
  $("flow").innerHTML = `Body → numbers: ${w} · ${n} points × (x, y) over the last ` +
    `${info ? info.window_seconds : 1} s (<b>${s.n_in_window}</b> frames) → small neural net → <b>7 raw scores</b> → softmax:`;
}

// ---------- camera frames ----------
function draw(bmp) {
  if (canvas.width !== bmp.width || canvas.height !== bmp.height) { canvas.width = bmp.width; canvas.height = bmp.height; }
  ctx.save();
  if (mirror) { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }
  ctx.drawImage(bmp, 0, 0);
  ctx.restore();
  bmp.close();
}

let decoding = false;
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/gesture`);
  ws.binaryType = "blob";
  ws.onmessage = ev => {
    if (typeof ev.data === "string") { onState(JSON.parse(ev.data)); return; }
    if (decoding) return;                       // drop a frame rather than fall behind
    decoding = true;
    createImageBitmap(ev.data).then(draw).catch(() => {}).finally(() => { decoding = false; });
  };
  ws.onclose = () => {
    $("msg").textContent = "Lost connection to the demo app. Is it still running? Retrying…";
    $("msg").classList.remove("hidden");
    setTimeout(connect, 1500);
  };
}

// ---------- sampling ----------
let pickTimer;
function samplePick() {
  if (!probs) { toast("Nothing to sample yet"); return; }
  const i = sample(probs);
  $("pickV").textContent = labels[i];
  $("pickP").textContent = `it had ${fmtPct(probs[i])} on this`;
  $("pick").classList.add("show");
  bars.flash(labels[i]);
  clearTimeout(pickTimer);
  pickTimer = setTimeout(() => $("pick").classList.remove("show"), 2600);
}

function toggleNumbers() {
  showNumbers = !showNumbers;
  $("numbers").classList.toggle("hidden", !showNumbers);
  $("numbersBtn").classList.toggle("on", showNumbers);
  bars.setExtra(showNumbers);
  setTimeout(() => bars.layout(), 0);
}

$("sampleBtn").addEventListener("click", samplePick);
$("numbersBtn").addEventListener("click", toggleNumbers);
document.addEventListener("keydown", e => {
  if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (e.key === " ") { e.preventDefault(); samplePick(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); temp.nudge(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); temp.nudge(1); }
  else if (k === "1" || k === "0") temp.set(1);
  else if (k === "n") toggleNumbers();
  else if (k === "m") { mirror = !mirror; store("mirror", mirror ? "1" : "0"); toast(mirror ? "Mirrored view" : "Camera view (not mirrored)"); }
});

fetch("/api/gesture/info").then(r => r.json()).then(d => {
  info = d;
  if (!labels.length) labels = d.gestures.map(g => g.label);
}).catch(() => {});
connect();
