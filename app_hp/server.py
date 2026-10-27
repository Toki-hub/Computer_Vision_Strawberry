"""
Aplikasi HP untuk menguji model strawberry lewat kamera HP.

Laptop menjalankan kedua model (penyakit dan kematangan). HP cukup membuka halaman web di browser
(Chrome di Android, Safari di iPhone), lewat Wi-Fi yang SAMA dengan laptop. Tidak perlu install apa pun di HP.

    python app_hp/server.py

Lalu buka alamat https://IP-LAPTOP:8443 yang dicetak di terminal dari HP. Browser HP hanya mengizinkan kamera
di halaman HTTPS, jadi skrip ini membuat sertifikat sendiri (self-signed) di app_hp/sertifikat/ saat pertama
kali dijalankan. Akibatnya HP akan menampilkan peringatan "koneksi tidak aman / not private": pilih
Lanjutan (Advanced) -> Lanjutkan ke situs (Proceed). Itu aman di sini karena alamatnya laptop Anda sendiri.

Foto yang disimpan lewat tombol "Simpan" di HP masuk ke app_hp/tangkapan/ (beserta hasil deteksinya dalam
.json), untuk dilabeli dan dipakai melatih ulang model dengan foto dari kamera sendiri.
"""
import argparse
import json
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DISEASE_WEIGHTS = ROOT / "Diseases Detection" / "results" / "best.pt"
RIPENESS_WEIGHTS = ROOT / "kematangan" / "results" / "best.pt"
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/manifest.json": ("manifest.json", "application/manifest+json"),
          "/icon.svg": ("icon.svg", "image/svg+xml")}
MAX_BODY = 12 * 2**20
OPENSSL_CANDIDATES = (r"C:\Program Files\Git\usr\bin\openssl.exe", r"C:\Program Files\Git\mingw64\bin\openssl.exe")

DISEASE_ID = {
    "angular_leafspot": "bercak daun bersudut",
    "anthracnose_fruit_rot": "busuk buah antraknosa",
    "blossom_blight": "hawar bunga",
    "gray_mold": "kapang abu-abu",
    "leaf_spot": "bercak daun",
    "powdery_mildew_fruit": "embun tepung buah",
    "powdery_mildew_leaf": "embun tepung daun",
}
RIPENESS_ID = {"ripe": "matang", "unripe": "mentah"}


def as_pt(path):
    """Ultralytics butuh ekstensi .pt; zip yang sebenarnya bobot PyTorch (mis. best.zip) disalin ke temp sebagai .pt."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"File bobot tidak ditemukan: {path}\nBeri path-nya lewat --disease / --ripeness.")
    if path.suffix.lower() != ".zip":
        return str(path)
    with zipfile.ZipFile(path) as z:
        if not any(n.endswith("data.pkl") for n in z.namelist()):
            raise SystemExit(f'"{path}" adalah arsip zip biasa: ekstrak dulu, lalu arahkan ke file .pt di dalamnya.')
    dest = Path(tempfile.gettempdir()) / "strawberry_app_weights" / f"{path.parent.name}_{path.stem}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    return str(dest)


def red_ratio(crop):
    """Fraksi piksel merah di elips tengah kotak (BGR); rumus sama dengan kematangan/realtime_kematangan.py."""
    import cv2
    import numpy as np

    h, w = crop.shape[:2]
    if h < 6 or w < 6:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, int(w * 0.35)), max(1, int(h * 0.35))), 0, 0, 360, 255, -1)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fruit = (mask > 0) & (V >= 40)
    n = int(fruit.sum())
    if n == 0:
        return None
    red = fruit & ((H <= 10) | (H >= 160)) & (S >= 90)
    return float(red.sum()) / n


def level_of(ratio, levels):
    if ratio is None:
        return None
    return "mentah" if ratio < levels[0] else ("setengah matang" if ratio < levels[1] else "matang")


class Detector:
    def __init__(self, disease, ripeness, imgsz, levels):
        from ultralytics import YOLO

        self.imgsz, self.levels = imgsz, levels
        self.models = {}
        for task, w in (("penyakit", disease), ("kematangan", ripeness)):
            if w:
                self.models[task] = YOLO(as_pt(w))
                print(f"Model {task}: {w} | kelas: {list(self.models[task].names.values())}", flush=True)
        if not self.models:
            raise SystemExit("Tidak ada model yang dimuat.")
        self.lock = threading.Lock()  # satu inferensi pada satu waktu (beberapa HP boleh terhubung)

    def warmup(self):
        import numpy as np

        self.detect(np.zeros((480, 640, 3), np.uint8), "keduanya", 0.5)

    def detect(self, img, mode, conf):
        import cv2

        tasks = list(self.models) if mode == "keduanya" else [mode]
        dets = []
        t0 = time.perf_counter()
        with self.lock:
            for task in tasks:
                model = self.models.get(task)
                if model is None:
                    continue
                r = model.predict(img, imgsz=self.imgsz, conf=conf, verbose=False)[0]
                for b in r.boxes:
                    name = model.names[int(b.cls)]
                    x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
                    d = {"task": task, "name": name, "conf": round(float(b.conf), 3), "box": [x0, y0, x1, y1]}
                    if task == "kematangan":
                        red = red_ratio(img[max(0, y0):y1, max(0, x0):x1])
                        d.update(label=RIPENESS_ID.get(name, name), red=None if red is None else round(red, 3),
                                 level=level_of(red, self.levels))
                    else:
                        d["label"] = DISEASE_ID.get(name, name)
                    dets.append(d)
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
        return {
            "dets": dets, "w": img.shape[1], "h": img.shape[0],
            "cahaya": round(float(v.mean()) / 255, 3),         # rata-rata kecerahan 0..1
            "silau": round(float((v >= 250).mean()), 3),        # fraksi piksel yang terlalu terang
            "ms": round((time.perf_counter() - t0) * 1000),
        }


def make_handler(det, save_dir):
    import cv2
    import numpy as np

    class Handler(BaseHTTPRequestHandler):
        server_version = "StrawberryApp/1.0"

        def log_message(self, fmt, *args):  # cukup cetak error, bukan setiap frame
            pass

        def send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/info":
                return self.send_json({"models": {t: list(m.names.values()) for t, m in det.models.items()},
                                       "imgsz": det.imgsz, "levels": det.levels})
            if path not in STATIC:
                return self.send_error(404)
            name, ctype = STATIC[path]
            body = (HERE / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def read_image(self):
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BODY:
                self.send_json({"error": "ukuran gambar tidak valid"}, 413)
                return None, None
            data = self.rfile.read(n)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self.send_json({"error": "gambar tidak bisa dibaca"}, 400)
                return None, None
            return img, data

        def do_POST(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            mode = q.get("mode", "penyakit")
            if mode not in ("penyakit", "kematangan", "keduanya"):
                return self.send_json({"error": f"mode tidak dikenal: {mode}"}, 400)
            try:
                conf = min(max(float(q.get("conf", 0.35)), 0.05), 0.95)
            except ValueError:
                return self.send_json({"error": "conf harus angka"}, 400)
            if url.path not in ("/deteksi", "/simpan"):
                return self.send_error(404)
            img, data = self.read_image()
            if img is None:
                return None
            try:
                res = det.detect(img, mode, conf)
            except Exception as e:  # jangan hentikan server karena satu frame
                print("Gagal mendeteksi:", repr(e), flush=True)
                return self.send_json({"error": str(e)}, 500)
            if url.path == "/simpan":
                save_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3] + f"_{mode}"
                (save_dir / f"{stem}.jpg").write_bytes(data)
                (save_dir / f"{stem}.json").write_text(
                    json.dumps({"mode": mode, "conf": conf, **res}, indent=2, ensure_ascii=False), encoding="utf-8")
                res["saved"] = f"{stem}.jpg"
                print(f"Disimpan: {save_dir / stem}.jpg ({len(res['dets'])} deteksi)", flush=True)
            return self.send_json(res)

    return Handler


def lan_ips():
    ips = []
    try:  # alamat yang dipakai untuk keluar ke jaringan (tidak mengirim paket apa pun)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        ips += [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    return [ip for ip in dict.fromkeys(ips) if not ip.startswith("127.")]


def ensure_cert(ips):
    d = HERE / "sertifikat"
    crt, key = d / "cert.pem", d / "key.pem"
    if crt.is_file() and key.is_file():
        return crt, key
    exe = shutil.which("openssl") or next((p for p in OPENSSL_CANDIDATES if Path(p).is_file()), None)
    if not exe:
        raise SystemExit("openssl tidak ditemukan (biasanya ikut Git for Windows). Install Git, atau beri path "
                         "sertifikat sendiri lewat --cert dan --key.")
    d.mkdir(exist_ok=True)
    san = ",".join([f"IP:{ip}" for ip in [*ips, "127.0.0.1"]] + ["DNS:localhost"])
    subprocess.run([exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825", "-keyout", str(key),
                    "-out", str(crt), "-subj", "/CN=strawberry-app", "-addext", f"subjectAltName={san}"],
                   check=True, capture_output=True)
    print(f"Sertifikat dibuat: {crt}", flush=True)
    return crt, key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--disease", default=str(DISEASE_WEIGHTS), help="bobot model penyakit ('' untuk tidak memakai)")
    ap.add_argument("--ripeness", default=str(RIPENESS_WEIGHTS), help="bobot model kematangan ('' untuk tidak memakai)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--levels", type=float, nargs=2, default=(0.3, 0.7), metavar=("BAWAH", "ATAS"),
                    help="ambang fraksi merah untuk tingkat mentah / setengah matang / matang")
    ap.add_argument("--save-dir", default=str(HERE / "tangkapan"))
    ap.add_argument("--cert")
    ap.add_argument("--key")
    ap.add_argument("--no-https", action="store_true", help="HTTP biasa (kamera hanya jalan bila dibuka dari laptop ini)")
    args = ap.parse_args()

    det = Detector(args.disease, args.ripeness, args.imgsz, tuple(args.levels))
    det.warmup()
    ips = lan_ips()
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(det, Path(args.save_dir)))
    scheme = "http" if args.no_https else "https"
    if not args.no_https:
        crt, key = (args.cert, args.key) if args.cert and args.key else ensure_cert(ips)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(crt, key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print("\nBuka salah satu alamat ini di browser HP (HP dan laptop harus di Wi-Fi yang sama):", flush=True)
    for ip in ips or ["<IP-laptop>"]:
        print(f"    {scheme}://{ip}:{args.port}", flush=True)
    print(f"Di laptop sendiri: {scheme}://localhost:{args.port}", flush=True)
    if not args.no_https:
        print("Peringatan sertifikat di HP: pilih Lanjutan/Advanced -> Lanjutkan/Proceed.", flush=True)
    print("Jika Windows menanyakan izin firewall untuk Python, centang 'Private networks'. Ctrl+C untuk berhenti.\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Dihentikan.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
