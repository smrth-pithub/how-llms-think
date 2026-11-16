topbar("/record");

const $ = id => document.getElementById(id);
const canvas = $("cam"), ctx = canvas.getContext("2d");
let gestures = [];
let hints = {};
let mirror = store("mirror") !== "0";
let lastRecSaved = 0;
let recording = false;

async function loadInfo() {
  const d = await (await fetch("/api/gesture/info")).json();
  gestures = d.gestures;
  hints = Object.fromEntries(gestures.map(g => [g.key, g.hint]));
  $("gchecks").innerHTML = gestures.map(g =>
    `<label><input type="checkbox" value="${g.key}" checked> ${g.label}</label>`).join("");
}

async function loadCounts() {
  const d = await (await fetch("/api/record/summary")).json();
  const people = Object.keys(d.people);
  if (!people.length) { $("counts").textContent = "Nothing recorded yet."; return; }
  const short = { waving: "Wave", thumbs_up: "Thumb", hands_up: "Up", clapping: "Clap", jumping: "Jump", arms_crossed: "Cross", standing_still: "Still" };
  let html = `<table class="counts"><tr><th>Person</th>${d.gestures.map(g => `<th title="${g.label}">${short[g.key] || g.label}</th>`).join("")}</tr>`;
  for (const p of people) {
    html += `<tr><td>${escapeHtml(p)}</td>${d.gestures.map(g => {
      const n = d.people[p][g.key] || 0;
      return `<td class="${n ? "" : "zero"}">${n || "–"}</td>`;
    }).join("")}</tr>`;
  }
  html += `</table><div class="note">${people.length} ${people.length === 1 ? "person" : "people"}. Aim for 5–6.</div>`;
  $("counts").innerHTML = html;
}

function escapeHtml(s) { return s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function onState(s) {
  const msg = $("msg");
  let text = "";
  if (s.status === "error") text = s.message;
  else if (s.status !== "running") text = s.message || "Starting camera…";
  else if (s.message) text = s.message;
  else if (!s.has_person) text = "Nobody in view";
  msg.textContent = text;
  msg.classList.toggle("hidden", !text);

  const r = s.rec;
  const prompt = $("prompt");
  const active = r && r.phase !== "done";
  if (active !== recording) {
    recording = active;
    $("startBtn").classList.toggle("hidden", active);
    $("stopBtn").classList.toggle("hidden", !active);
  }
  if (!active) {
    prompt.classList.add("hidden");
    $("progress").style.width = "0";
    if (r && r.phase === "done" && r.saved.length !== lastRecSaved) {
      lastRecSaved = r.saved.length;
      toast(`Saved ${r.saved.length} clips for ${r.person}`);
      loadCounts();
    }
  } else {
    prompt.classList.remove("hidden");
    prompt.classList.toggle("countdown", r.phase === "countdown");
    prompt.classList.toggle("recording", r.phase === "recording");
    $("pPhase").innerHTML = r.phase === "countdown"
      ? `Get ready · ${r.index + 1} of ${r.total}` : `<span class="recdot"></span>Recording · ${r.index + 1} of ${r.total}`;
    $("pWhat").textContent = r.label;
    $("pHint").textContent = r.phase === "countdown" ? (hints[r.gesture] || "") : "";
    $("pNum").textContent = Math.ceil(r.remaining);
    const secs = +$("seconds").value;
    $("progress").style.width = r.phase === "recording" ? (100 * (1 - r.remaining / secs)).toFixed(1) + "%" : "0";
    if (r.saved.length !== lastRecSaved) { lastRecSaved = r.saved.length; loadCounts(); }
  }
  $("msgs").textContent = r && r.messages && r.messages.length ? "⚠ " + r.messages.join(" · ") : "";
}

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
    if (decoding) return;
    decoding = true;
    createImageBitmap(ev.data).then(draw).catch(() => {}).finally(() => { decoding = false; });
  };
  ws.onclose = () => { setTimeout(connect, 1500); };
}

$("startBtn").addEventListener("click", async () => {
  const person = $("person").value.trim();
  if (!person) { $("person").focus(); toast("Type the person's name first"); return; }
  const chosen = [...document.querySelectorAll("#gchecks input:checked")].map(i => i.value);
  if (!chosen.length) { toast("Pick at least one gesture"); return; }
  lastRecSaved = 0;
  await postJSON("/api/record/start", { person, gestures: chosen, seconds: +$("seconds").value, countdown: 3 });
});
$("stopBtn").addEventListener("click", async () => { await postJSON("/api/record/stop"); toast("Stopped. The clip in progress was not saved."); loadCounts(); });
$("discardBtn").addEventListener("click", async () => {
  const d = await postJSON("/api/record/discard-last");
  toast(d.discarded ? `Deleted ${d.discarded}` : "No clips from this session to discard");
  loadCounts();
});

$("trainBtn").addEventListener("click", async () => {
  const btn = $("trainBtn"), log = $("log");
  btn.disabled = true; btn.textContent = "Training…";
  log.classList.remove("hidden"); log.textContent = "";
  try {
    const r = await fetch("/api/train", { method: "POST" });
    const reader = r.body.getReader(), dec = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      log.textContent += dec.decode(value, { stream: true });
      log.scrollTop = log.scrollHeight;
    }
    toast(log.textContent.includes("[exit code 0]") ? "Model trained and loaded into the demo" : "Training failed: see the log", 3500);
  } catch (e) {
    log.textContent += "\n" + e;
  } finally {
    btn.disabled = false; btn.textContent = "Train";
  }
});

loadInfo().then(loadCounts);
connect();
