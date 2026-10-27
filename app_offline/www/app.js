// Cek Strawberry: deteksi penyakit dan kematangan langsung di HP (onnxruntime-web), tanpa laptop/server.
// Pra/pasca-proses (letterbox persegi, NMS per kelas IoU 0.7, ambang warna merah) sama dengan versi Python.
const $ = s => document.querySelector(s);
const video = $("#video"), still = $("#still"), overlay = $("#overlay"), octx = overlay.getContext("2d");
const lb = document.createElement("canvas"), lbctx = lb.getContext("2d", { willReadFrequently: true });
const grab = document.createElement("canvas"), gctx = grab.getContext("2d");
const COLORS = { ripe: "#e5484d", unripe: "#3fb950", penyakit: "#ff9f1a" };
const IOU = 0.7, MAX_DET = 300, RED_LEVELS = [0.3, 0.7];

let META = null, IMGSZ = 640, ENGINE = "cpu";
const SESS = {};  // task -> {session, names, labels, input, output}
let mode = localStorage.getItem("mode") || "penyakit";
let conf = parseFloat(localStorage.getItem("conf") || "0.35");
let facing = "environment", stream = null, track = null, paused = false, torchOn = false;
let last = null, busy = false, times = [], stillMode = false, wakeLock = null, queue = Promise.resolve();

// ---------- UI kecil ----------
function toast(text, ms = 1800) {
  const t = $("#toast"); t.textContent = text; t.classList.add("show");
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove("show"), ms);
}
function showMsg(title, text) { $("#msgTitle").textContent = title; $("#msgText").textContent = text; $("#msg").classList.add("show"); }
function hideMsg() { $("#msg").classList.remove("show"); }
function loading(text, frac) {
  $("#loadText").textContent = text;
  if (frac != null) $("#progress i").style.width = `${Math.round(frac * 100)}%`;
}

// ---------- service worker: offline + cross-origin isolation (WASM multi-thread) ----------
async function setupServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  try {
    await navigator.serviceWorker.register("sw.js");
    await navigator.serviceWorker.ready;
    if (!window.crossOriginIsolated && !sessionStorage.getItem("coiReload")) {
      if (!navigator.serviceWorker.controller) {
        await new Promise(r => { navigator.serviceWorker.addEventListener("controllerchange", r, { once: true }); setTimeout(r, 3000); });
      }
      sessionStorage.setItem("coiReload", "1");  // sekali saja, supaya tidak berputar-putar
      location.reload();
      await new Promise(() => {});
    }
    navigator.storage?.persist?.();
  } catch (e) { console.warn("Service worker gagal:", e); }
}

async function fetchProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = +res.headers.get("Content-Length") || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(), out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got); got += value.length; onProgress(got / total);
  }
  return out;
}

async function loadModels() {
  META = await (await fetch("models.json", { cache: "no-store" })).json();
  IMGSZ = META.imgsz; lb.width = lb.height = IMGSZ;
  ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
  ort.env.wasm.numThreads = window.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const want = localStorage.getItem("engine") || (navigator.gpu ? "gpu" : "cpu");
  const tasks = Object.keys(META.models);
  const total = tasks.reduce((a, t) => a + META.models[t].bytes, 0);
  let before = 0;
  for (const task of tasks) {
    const m = META.models[task];
    const bytes = await fetchProgress(m.file, f => loading(`Memuat model ${task}... ${Math.round((before + f * m.bytes) / 2 ** 20)} / ${Math.round(total / 2 ** 20)} MB`, (before + f * m.bytes) / total));
    before += m.bytes;
    loading(`Menyiapkan model ${task}...`, before / total);
    let session = null;
    if (want === "gpu" && navigator.gpu) {
      try { session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] }); ENGINE = "gpu"; }
      catch (e) { console.warn("WebGPU gagal, pindah ke CPU:", e); }
    }
    if (!session) { session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] }); ENGINE = "cpu"; }
    SESS[task] = { session, names: m.names, labels: m.labels, input: session.inputNames[0], output: session.outputNames[0] };
  }
  $("#engine").value = ENGINE;
  if (want === "gpu" && ENGINE === "cpu") {  // HP/browser ini tidak punya WebGPU
    $("#engine").querySelector('[value="gpu"]').disabled = true;
    localStorage.setItem("engine", "cpu");
  }
}

async function checkOffline() {
  const c = $("#offline");
  if (!("caches" in window) || !navigator.serviceWorker?.controller) { c.textContent = "Offline: belum"; c.className = "chip warn"; return; }
  const need = ["index.html", "app.js", "models.json", "ort/ort.min.js", "ort/ort-wasm-simd-threaded.jsep.mjs",
    "ort/ort-wasm-simd-threaded.jsep.wasm", ...Object.values(META.models).map(m => m.file)];
  const ok = (await Promise.all(need.map(u => caches.match(new URL(u, location.href).href, { ignoreSearch: true })))).every(Boolean);
  c.textContent = ok ? "Siap offline" : "Offline: belum lengkap"; c.className = "chip " + (ok ? "ok" : "warn");
}

// ---------- deteksi ----------
function preprocess(src, w, h) {
  const r = Math.min(IMGSZ / w, IMGSZ / h), nw = Math.round(w * r), nh = Math.round(h * r);
  const dx = Math.floor((IMGSZ - nw) / 2), dy = Math.floor((IMGSZ - nh) / 2);
  lbctx.fillStyle = "rgb(114,114,114)"; lbctx.fillRect(0, 0, IMGSZ, IMGSZ);
  lbctx.drawImage(src, 0, 0, w, h, dx, dy, nw, nh);
  const px = lbctx.getImageData(0, 0, IMGSZ, IMGSZ).data, n = IMGSZ * IMGSZ;
  const data = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) { data[i] = px[p] / 255; data[n + i] = px[p + 1] / 255; data[2 * n + i] = px[p + 2] / 255; }
  return { data, px, r, dx, dy, nw, nh, w, h };
}

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]), x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
}

function decode(out, nc, minConf) {
  const N = out.length / (4 + nc), cands = [];
  for (let j = 0; j < N; j++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) { const s = out[(4 + c) * N + j]; if (s > best) { best = s; cls = c; } }
    if (best < minConf) continue;
    const cx = out[j], cy = out[N + j], w = out[2 * N + j], h = out[3 * N + j];
    cands.push({ cls, score: best, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] });
  }
  cands.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of cands) {  // NMS per kelas
    if (keep.length >= MAX_DET) break;
    if (!keep.some(k => k.cls === d.cls && iou(k.box, d.box) >= IOU)) keep.push(d);
  }
  return keep;
}

// Fraksi piksel merah di elips tengah kotak (koordinat letterbox); rumus HSV ala OpenCV (H 0..180).
function redRatio(px, b) {
  const x0 = Math.max(0, Math.floor(b[0])), y0 = Math.max(0, Math.floor(b[1]));
  const x1 = Math.min(IMGSZ, Math.ceil(b[2])), y1 = Math.min(IMGSZ, Math.ceil(b[3]));
  const w = x1 - x0, h = y1 - y0;
  if (w < 6 || h < 6) return null;
  const cx = x0 + w / 2, cy = y0 + h / 2, ax = Math.max(1, w * 0.35), ay = Math.max(1, h * 0.35);
  let fruit = 0, red = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const ex = (x + 0.5 - cx) / ax, ey = (y + 0.5 - cy) / ay;
    if (ex * ex + ey * ey > 1) continue;
    const p = (y * IMGSZ + x) * 4, R = px[p], G = px[p + 1], B = px[p + 2];
    const V = Math.max(R, G, B);
    if (V < 40) continue;
    fruit++;
    const mn = Math.min(R, G, B), S = V === 0 ? 0 : (V - mn) * 255 / V;
    if (S < 90) continue;
    let H = 0; const d = V - mn;
    if (d > 0) { H = V === R ? 60 * (G - B) / d : V === G ? 120 + 60 * (B - R) / d : 240 + 60 * (R - G) / d; if (H < 0) H += 360; }
    H /= 2;
    if (H <= 10 || H >= 160) red++;
  }
  return fruit ? red / fruit : null;
}
const levelOf = r => r == null ? null : r < RED_LEVELS[0] ? "mentah" : r < RED_LEVELS[1] ? "setengah matang" : "matang";

function lightOf(pre) {
  let sum = 0, bright = 0, n = 0;
  for (let y = pre.dy; y < pre.dy + pre.nh; y += 4) for (let x = pre.dx; x < pre.dx + pre.nw; x += 4) {
    const p = (y * IMGSZ + x) * 4, V = Math.max(pre.px[p], pre.px[p + 1], pre.px[p + 2]);
    sum += V; bright += V >= 250; n++;
  }
  return { cahaya: n ? sum / n / 255 : 0, silau: n ? bright / n : 0 };
}

async function detect(src, w, h) {
  const t0 = performance.now();
  const pre = preprocess(src, w, h);
  const tensor = new ort.Tensor("float32", pre.data, [1, 3, IMGSZ, IMGSZ]);
  const tasks = mode === "keduanya" ? Object.keys(SESS) : [mode];
  const dets = [];
  for (const task of tasks) {
    const s = SESS[task]; if (!s) continue;
    const res = await s.session.run({ [s.input]: tensor });
    const out = res[s.output].data;
    for (const d of decode(out, s.names.length, conf)) {
      const name = s.names[d.cls];
      const box = [(d.box[0] - pre.dx) / pre.r, (d.box[1] - pre.dy) / pre.r, (d.box[2] - pre.dx) / pre.r, (d.box[3] - pre.dy) / pre.r]
        .map((v, i) => Math.max(0, Math.min(i % 2 ? h : w, v)));
      const det = { task, name, label: s.labels[d.cls], conf: d.score, box };
      if (task === "kematangan") { det.red = redRatio(pre.px, d.box); det.level = levelOf(det.red); }
      dets.push(det);
    }
  }
  return { dets, w, h, ...lightOf(pre), ms: Math.round(performance.now() - t0) };
}
// satu inferensi pada satu waktu (session ONNX tidak boleh dipanggil bersamaan)
function detectQueued(src, w, h) { const p = queue.then(() => detect(src, w, h)); queue = p.catch(() => {}); return p; }

// ---------- kamera ----------
async function startCamera() {
  hideMsg();
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showMsg("Kamera tidak bisa dibuka", "Aplikasi harus dibuka lewat https://. Anda tetap bisa menguji foto dari Galeri.");
    return;
  }
  stream?.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } } });
  } catch (e) {
    showMsg("Izin kamera ditolak", "Izinkan akses kamera untuk aplikasi ini di pengaturan browser, lalu tekan Coba lagi. (" + e.name + ")");
    return;
  }
  track = stream.getVideoTracks()[0];
  video.srcObject = stream;
  await video.play().catch(() => {});
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  $("#torch").hidden = !caps.torch; torchOn = false; $("#torch").textContent = "Senter";
  if (!stillMode) leaveStill();  // jangan menutup foto galeri yang sedang dilihat
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (_) {}
}

async function loop() {
  if (!stillMode && !paused && !busy && video.videoWidth && document.visibilityState === "visible") {
    busy = true;
    const t0 = performance.now();
    try {
      const r = await detectQueued(video, video.videoWidth, video.videoHeight);
      if (!stillMode) { last = r; draw(); }
      times.push(performance.now() - t0); if (times.length > 10) times.shift();
      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      $("#stat").textContent = `${(1000 / avg).toFixed(1)} fps · ${ENGINE.toUpperCase()}`;
    } catch (e) {
      console.error(e); $("#stat").textContent = "Gagal: " + e.message;
      await new Promise(r => setTimeout(r, 1000));
    }
    busy = false;
  }
  requestAnimationFrame(loop);
}

// ---------- gambar hasil ----------
function sourceRect() {
  const vw = stillMode ? still.naturalWidth : video.videoWidth, vh = stillMode ? still.naturalHeight : video.videoHeight;
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  if (!vw || !vh) return null;
  const s = Math.min(cw / vw, ch / vh);  // object-fit: contain
  return { x: (cw - vw * s) / 2, y: (ch - vh * s) / 2, w: vw * s, h: vh * s };
}
function lightChip(r) {
  const c = $("#light");
  if (!r) { c.textContent = "Cahaya: -"; c.className = "chip"; return; }
  let cls = "ok", txt = `Cahaya ${Math.round(r.cahaya * 100)}%`;
  if (r.silau > 0.15) { cls = "warn"; txt += " · silau"; }
  else if (r.cahaya < 0.25) { cls = "bad"; txt += " · gelap"; }
  else if (r.cahaya < 0.38) { cls = "warn"; txt += " · redup"; }
  c.textContent = txt; c.className = "chip " + cls;
}
const colorOf = d => d.task === "kematangan" ? (COLORS[d.name] || "#4ea1ff") : COLORS.penyakit;
function textOf(d) {
  let t = `${d.label} ${(d.conf * 100).toFixed(0)}%`;
  if (d.task === "kematangan" && d.red != null) t += ` · merah ${Math.round(d.red * 100)}%`;
  return t;
}
function draw() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(overlay.clientWidth * dpr); overlay.height = Math.round(overlay.clientHeight * dpr);
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
  lightChip(last);
  renderList();
  const rect = sourceRect();
  if (!last || !rect) return;
  const sx = rect.w / last.w, sy = rect.h / last.h;
  octx.font = "600 13px system-ui, sans-serif"; octx.textBaseline = "top";
  const placed = [];  // label yang sudah digambar, supaya label kotak berdekatan tidak saling menimpa
  for (const d of last.dets) {
    const [x0, y0, x1, y1] = d.box;
    const x = rect.x + x0 * sx, y = rect.y + y0 * sy, w = (x1 - x0) * sx, h = (y1 - y0) * sy;
    const col = colorOf(d), text = textOf(d);
    octx.lineWidth = 2.5; octx.strokeStyle = col; octx.strokeRect(x, y, w, h);
    const tw = octx.measureText(text).width + 10;
    let ty = y >= 22 ? y - 21 : y + h + 1;
    for (let k = 0; k < 8 && placed.some(p => x < p.x + p.w && x + tw > p.x && ty < p.y + 20 && ty + 20 > p.y); k++) ty += 21;
    placed.push({ x, y: ty, w: tw });
    octx.fillStyle = col; octx.fillRect(x - 1.25, ty, tw, 20);
    octx.fillStyle = "#fff"; octx.fillText(text, x + 4, ty + 3);
  }
}
function renderList() {
  const list = $("#list"); list.innerHTML = "";
  if (!last) return;
  if (!last.dets.length) {
    const row = document.createElement("div"); row.className = "row";
    row.innerHTML = "<b></b><span></span>";
    row.querySelector("b").textContent = mode === "kematangan" ? "Tidak ada buah terdeteksi" : mode === "penyakit" ? "Tidak ada penyakit terdeteksi" : "Tidak ada deteksi";
    row.querySelector("span").textContent = `ambang ${conf.toFixed(2)}`;
    list.append(row); return;
  }
  const groups = new Map();
  for (const d of last.dets) {
    const key = d.task === "kematangan" && d.level ? `${d.label} (warna: ${d.level})` : d.label;
    const g = groups.get(key) || { n: 0, best: 0, color: colorOf(d) };
    g.n++; g.best = Math.max(g.best, d.conf); groups.set(key, g);
  }
  for (const [key, g] of [...groups].sort((a, b) => b[1].best - a[1].best)) {
    const row = document.createElement("div"); row.className = "row";
    row.innerHTML = `<b><i class="dot" style="background:${g.color}"></i></b><span>×${g.n} · ${(g.best * 100).toFixed(0)}%</span>`;
    row.querySelector("b").append(key);
    list.append(row);
  }
}
window.addEventListener("resize", draw);

// ---------- kontrol ----------
function setMode(m) {
  mode = m; localStorage.setItem("mode", m);
  document.querySelectorAll("#modes button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
  last = null; draw();
  if (stillMode) analyzeStill();
}
document.querySelectorAll("#modes button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
$("#conf").value = conf; $("#confv").textContent = conf.toFixed(2);
$("#conf").oninput = e => { conf = parseFloat(e.target.value); $("#confv").textContent = conf.toFixed(2); localStorage.setItem("conf", conf); };
$("#conf").onchange = () => { if (stillMode) analyzeStill(); };
$("#engine").onchange = e => { localStorage.setItem("engine", e.target.value); location.reload(); };
$("#retry").onclick = startCamera;
$("#flip").onclick = () => { facing = facing === "environment" ? "user" : "environment"; startCamera(); };
$("#torch").onclick = async () => {
  try { torchOn = !torchOn; await track.applyConstraints({ advanced: [{ torch: torchOn }] }); $("#torch").textContent = torchOn ? "Senter: on" : "Senter"; }
  catch (_) { toast("Senter tidak didukung kamera ini"); }
};
$("#pause").onclick = () => {
  if (stillMode) { leaveStill(); return; }
  paused = !paused; $("#pause").textContent = paused ? "Lanjut" : "Jeda";
  paused ? video.pause() : video.play();
};
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible" && wakeLock !== null) { try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (_) {} }
});

// simpan foto resolusi penuh ke HP (folder Download), untuk dilabeli dan melatih ulang model
$("#save").onclick = async () => {
  try {
    let blob;
    if (stillMode) blob = await (await fetch(still.src)).blob();
    else {
      grab.width = video.videoWidth; grab.height = video.videoHeight; gctx.drawImage(video, 0, 0);
      blob = await new Promise(r => grab.toBlob(r, "image/jpeg", 0.92));
    }
    const now = new Date(), pad = n => String(n).padStart(2, "0");
    const name = `strawberry_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}_${mode}.jpg`;
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    toast(`Disimpan: ${name}`);
  } catch (e) { toast("Gagal menyimpan: " + e.message); }
};

$("#gallery").onclick = () => $("#file").click();
$("#msgGallery").onclick = () => { hideMsg(); $("#file").click(); };
$("#file").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  if (still.src) URL.revokeObjectURL(still.src);
  const ok = await new Promise(res => { still.onload = () => res(true); still.onerror = () => res(false); still.src = URL.createObjectURL(f); });
  e.target.value = "";
  if (!ok) { toast("Foto tidak bisa dibuka"); return; }
  stillMode = true; still.style.display = "block"; video.style.visibility = "hidden";
  $("#pause").textContent = "Kamera live";
  analyzeStill();
};
async function analyzeStill() {
  try { last = await detectQueued(still, still.naturalWidth, still.naturalHeight); draw(); $("#stat").textContent = `${last.ms} ms · ${ENGINE.toUpperCase()}`; }
  catch (e) { console.error(e); toast("Gagal mendeteksi: " + e.message); }
}
function leaveStill() {
  stillMode = false; still.style.display = "none"; video.style.visibility = "visible";
  paused = false; $("#pause").textContent = "Jeda"; last = null; draw();
}

// ---------- mulai ----------
(async () => {
  setMode(mode);
  try {
    loading("Menyiapkan penyimpanan offline...");
    await setupServiceWorker();
    await loadModels();
  } catch (e) {
    console.error(e);
    loading("Gagal memuat model: " + e.message + ". Pastikan ada internet saat membuka pertama kali, lalu muat ulang.");
    return;
  }
  $("#loading").classList.remove("show");
  checkOffline();
  startCamera();
  requestAnimationFrame(loop);
})();
