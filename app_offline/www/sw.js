// Service worker: menyimpan aplikasi + model di HP supaya jalan tanpa internet, dan menambahkan header
// cross-origin isolation (COOP/COEP) agar onnxruntime boleh memakai beberapa inti CPU.
// VERSION diisi otomatis oleh build.py; versi baru = cache baru, cache lama dihapus.
const VERSION = "0d95120bbc8b";
const CACHE = "strawberry-" + VERSION;
const SHELL = ["./", "index.html", "app.js", "manifest.webmanifest", "icon-192.png", "icon-512.png", "models.json",
  "ort/ort.min.js", "ort/ort-wasm-simd-threaded.jsep.mjs"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith("strawberry-") && k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

function isolate(res) {
  if (!res || res.status === 0 || res.type === "opaque") return res;
  const h = new Headers(res.headers);
  h.set("Cross-Origin-Embedder-Policy", "require-corp");
  h.set("Cross-Origin-Opener-Policy", "same-origin");
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    // models.json selalu dicoba dari jaringan dulu agar model baru terdeteksi; sisanya dari cache dulu
    if (req.url.endsWith("models.json")) {
      try {
        const res = await fetch(req, { cache: "no-store" });
        if (res.ok) cache.put(req, res.clone());
        return isolate(res);
      } catch (_) { /* offline */ }
    }
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) return isolate(hit);
    const res = await fetch(req);
    if (res.ok && res.type === "basic") cache.put(req, res.clone());
    return isolate(res);
  })());
});
