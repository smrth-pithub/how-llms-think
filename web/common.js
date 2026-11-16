// Shared by every page. The softmax below is THE softmax: the gesture demo and the
// next-token viewer both call this exact function, and "Show the numbers" prints its source.

function softmax(scores, temperature) {
  const scaled = scores.map(s => s / temperature);   // temperature: <1 sharpens, >1 flattens
  const biggest = Math.max(...scaled);               // subtract the max so exp() can't overflow
  const exps = scaled.map(s => Math.exp(s - biggest));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / total);                   // percentages that add up to 100%
}

// Pick one option at random, in proportion to its probability.
function sample(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

// ---------- temperature slider: log scale, 0.1 .. 10, with 1.0 in the middle ----------
const TEMP_MIN = 0.1, TEMP_MAX = 10;
function sliderToTemp(v) { return Math.pow(10, (v - 50) / 50); }
function tempToSlider(t) { return 50 + 50 * Math.log10(t); }
function fmtTemp(t) { return t < 1 ? t.toFixed(2) : t < 10 ? t.toFixed(1) : t.toFixed(0); }

function setupTemperature(root, onChange) {
  root.innerHTML = `
    <div class="temp-label">Temperature <span class="temp-value">1.0</span></div>
    <input type="range" min="0" max="100" step="0.5" value="50" aria-label="Temperature">
    <div class="temp-ends"><span>&larr; sharp</span><span>flat &rarr;</span></div>`;
  const input = root.querySelector("input");
  const value = root.querySelector(".temp-value");
  const api = {
    value: 1,
    set(t) {
      t = Math.min(TEMP_MAX, Math.max(TEMP_MIN, t));
      input.value = tempToSlider(t);
      update();
    },
    nudge(dir) { input.value = Math.min(100, Math.max(0, +input.value + dir * 5)); update(); },
  };
  function update() {
    let t = sliderToTemp(+input.value);
    if (Math.abs(t - 1) < 0.04) t = 1;                // snap to 1.0 near the middle
    api.value = t;
    value.textContent = fmtTemp(t);
    const pct = +input.value;
    input.style.setProperty("--fill", pct + "%");
    onChange(t);
  }
  input.addEventListener("input", update);
  input.addEventListener("dblclick", () => api.set(1));
  update();
  return api;
}

// ---------- ranked bars ----------
// items: [{key, label, value (0..1), extra (optional small text)}]
class Bars {
  constructor(el, opts = {}) {
    this.el = el;
    this.rows = opts.rows || 7;
    this.onClick = opts.onClick || null;
    this.reorderMs = opts.reorderMs ?? 0;
    this.formatLabel = opts.formatLabel || (s => document.createTextNode(s));
    this.maxFont = opts.maxFont || 40;      // text size follows the row height, so it fits any screen
    this.map = new Map();
    this.order = [];
    this.lastReorder = 0;
    this.showExtra = false;
    window.addEventListener("resize", () => this.layout());
    if (window.ResizeObserver) new ResizeObserver(() => this.layout()).observe(el);
  }
  rowHeight() {
    const h = this.el.clientHeight / this.rows;
    this.el.style.fontSize = Math.max(14, Math.min(this.maxFont, h * 0.58)).toFixed(1) + "px";
    return h;
  }
  setExtra(on) { this.showExtra = on; this.el.classList.toggle("with-extra", on); this.map.forEach(r => r.extra.classList.toggle("hidden", !on)); }

  _make(item) {
    const row = document.createElement("div");
    row.className = "bar-row" + (this.onClick ? " clickable" : "");
    row.style.opacity = "0";
    const label = document.createElement("div"); label.className = "bar-label";
    const track = document.createElement("div"); track.className = "bar-track";
    const fill = document.createElement("div"); fill.className = "bar-fill";
    const right = document.createElement("div"); right.className = "bar-right";
    const pct = document.createElement("div"); pct.className = "bar-pct";
    const extra = document.createElement("div"); extra.className = "bar-extra" + (this.showExtra ? "" : " hidden");
    track.appendChild(fill); right.append(pct, extra);
    row.append(label, track, right);
    label.appendChild(this.formatLabel(item.label, item));
    if (this.onClick) row.addEventListener("click", () => this.onClick(row._item));
    this.el.appendChild(row);
    const r = { row, label, fill, pct, extra, gone: false };
    this.map.set(item.key, r);
    return r;
  }

  update(items, { forceOrder = false } = {}) {
    const now = performance.now();
    const byKey = new Map(items.map(i => [i.key, i]));
    let order = this.order.filter(k => byKey.has(k));
    const sorted = [...items].sort((a, b) => b.value - a.value).slice(0, this.rows).map(i => i.key);
    if (forceOrder || now - this.lastReorder >= this.reorderMs || order.length !== sorted.length ||
        sorted.some(k => !order.includes(k))) {
      if (order.join("|") !== sorted.join("|")) this.lastReorder = now;
      order = sorted;
    }
    this.order = order;
    const keep = new Set(order);
    for (const [k, r] of this.map) {
      if (!keep.has(k) && !r.gone) {
        r.gone = true; r.row.style.opacity = "0";
        setTimeout(() => { if (r.gone) { r.row.remove(); this.map.delete(k); } }, 260);
      }
    }
    const h = this.rowHeight();
    order.forEach((k, rank) => {
      const item = byKey.get(k);
      let r = this.map.get(k);
      const isNew = !r;
      if (!r) r = this._make(item);
      if (r.gone) { r.gone = false; }
      r.row._item = item;
      r.row.style.height = h + "px";
      if (isNew) {
        r.row.style.transition = "none";
        r.row.style.transform = `translateY(${rank * h}px)`;
        r.row.getBoundingClientRect();
        r.row.style.transition = "";
      }
      r.row.style.transform = `translateY(${rank * h}px)`;
      r.row.style.opacity = "1";
      r.row.classList.toggle("top", rank === 0);
      r.fill.style.width = (Math.max(0, Math.min(1, item.value)) * 100).toFixed(2) + "%";
      r.pct.textContent = fmtPct(item.value);
      if (item.extra !== undefined) r.extra.textContent = item.extra;
    });
  }

  layout() {
    const h = this.rowHeight();
    this.order.forEach((k, rank) => {
      const r = this.map.get(k);
      if (r) { r.row.style.height = h + "px"; r.row.style.transform = `translateY(${rank * h}px)`; }
    });
  }

  flash(key) {
    const r = this.map.get(key);
    if (!r) return;
    r.row.classList.remove("flash"); void r.row.offsetWidth; r.row.classList.add("flash");
    setTimeout(() => r.row.classList.remove("flash"), 1400);
  }
}

function fmtPct(p) {
  const v = p * 100;
  if (v >= 99.95) return "100%";
  if (v >= 9.95) return v.toFixed(0) + "%";
  if (v >= 0.995) return v.toFixed(1) + "%";
  if (v >= 0.01) return v.toFixed(2) + "%";
  return "<0.01%";
}

// ---------- page chrome: theme, fullscreen, hide top bar, toasts ----------
function store(key, val) {
  try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; }
}
(function initTheme() {
  const saved = store("theme");
  if (saved) document.documentElement.dataset.theme = saved;
})();
function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  store("theme", next);
}
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => {});
}
function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === "INPUT" && t.type !== "range" || t.tagName === "TEXTAREA" || t.isContentEditable);
}
document.addEventListener("keydown", e => {
  if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "d") toggleTheme();
  else if (k === "f") toggleFullscreen();
  else if (k === "h") document.body.classList.toggle("chrome-hidden");
});

let toastTimer;
function toast(msg, ms = 2200) {
  let el = document.querySelector(".toast");
  if (!el) { el = document.createElement("div"); el.className = "toast"; document.body.appendChild(el); }
  el.textContent = msg; el.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove("show"), ms);
}

function topbar(current) {
  const links = [["/", "Home"], ["/gesture", "1 · Gesture demo"], ["/tokenizer", "2 · Tokenizer"],
                 ["/llm", "3 · Next token"], ["/record", "Record & train"]];
  const bar = document.createElement("header");
  bar.className = "topbar";
  bar.innerHTML = `<span class="brand">How LLMs Think</span><nav>${links.map(([h, t]) =>
    `<a href="${h}"${h === current ? ' aria-current="page"' : ""}>${t}</a>`).join("")}</nav>
    <span class="spacer"></span><span class="keys"><kbd>F</kbd> fullscreen · <kbd>H</kbd> hide bar · <kbd>D</kbd> dark</span>`;
  document.body.prepend(bar);
}

async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const err = new Error(data.error || data.detail || r.statusText); err.data = data; err.status = r.status; throw err; }
  return data;
}

// Make spaces and newlines visible in a token: " mat" -> "·mat"
function showToken(tok) {
  const frag = document.createDocumentFragment();
  if (tok === "") { frag.append("∅"); return frag; }
  const special = tok.match(/^<\|(.+)\|>$/);
  if (special) { const s = document.createElement("span"); s.className = "ws"; s.textContent = "⟨" + special[1] + "⟩"; frag.append(s); return frag; }
  let buf = "";
  const flush = () => { if (buf) { frag.append(buf); buf = ""; } };
  for (const ch of tok) {
    if (ch === " " || ch === "\n" || ch === "\t") {
      flush();
      const s = document.createElement("span"); s.className = "ws";
      s.textContent = ch === " " ? "·" : ch === "\n" ? "↵" : "→";
      frag.append(s);
    } else buf += ch;
  }
  flush();
  return frag;
}
