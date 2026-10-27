"""
Deteksi penyakit strawberry secara real time dari kamera.

Secara default skrip memakai model terbaru: results/best.pt (diekstrak dari "results (2).zip",
YOLO26s, 7 kelas, mAP50 test 0.865). Kalau folder results/ tidak ada, skrip mencari best.pt di folder
yang sama dengan realtime.py atau di subfolder (mis. folder hasil 'Extract All'), lalu best.zip
(file .pt yang ekstensinya berubah, model lama). Lalu jalankan:

    # webcam laptop / USB (memakai results/best.pt)
    python realtime.py

    # model lama (best.zip)
    python realtime.py --weights best.zip

    # file bobot di tempat lain
    python realtime.py --weights "C:/proyek/best.pt" --source 0

    # kamera IP / ESP32-CAM (stream MJPEG atau RTSP), tanpa jendela (perangkat headless)
    python realtime.py --source http://192.168.1.50:81/stream --fps 1 --no-show

    # uji dengan file video (semua frame diproses, berhenti saat video habis)
    python realtime.py --source contoh.mp4

    # kirim event ke broker MQTT (pip install paho-mqtt)
    python realtime.py --source 0 --mqtt-host 192.168.1.10

Agar tidak membunyikan alarm dari satu frame yang keliru, sebuah penyakit baru dianggap
"terdeteksi" jika muncul di >= --min-hits dari --window frame terakhir, dan dianggap "hilang"
jika tidak muncul sama sekali dalam --window frame terakhir. Event dikirim hanya saat status berubah.
"""
import argparse
import json
import shutil
import tempfile
import time
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path

# Model terbaru (best.pt dari "results (2).zip"), relatif ke folder realtime.py
DEFAULT_WEIGHTS = "results/best.pt"


class Debouncer:
    """Mengubah deteksi per-frame menjadi event 'muncul' / 'hilang' yang stabil."""

    def __init__(self, window=8, min_hits=3):
        self.min_hits = min_hits
        self.history = deque(maxlen=window)
        self.active = set()

    def update(self, found):
        """found: kumpulan nama kelas yang terdeteksi pada frame ini. Mengembalikan (muncul, hilang)."""
        self.history.append(set(found))
        counts = {}
        for frame in self.history:
            for name in frame:
                counts[name] = counts.get(name, 0) + 1
        appeared = {n for n, c in counts.items() if c >= self.min_hits and n not in self.active}
        cleared = {n for n in self.active if counts.get(n, 0) == 0}
        self.active = (self.active | appeared) - cleared
        return appeared, cleared


def make_publisher(mqtt_host=None, mqtt_port=1883, mqtt_topic="strawberry/disease"):
    """Mengembalikan fungsi publish(event: dict). Tanpa --mqtt-host, event hanya dicetak."""
    if not mqtt_host:
        return lambda event: print(json.dumps(event, ensure_ascii=False), flush=True)

    import paho.mqtt.client as mqtt

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # paho-mqtt 2.x
    except AttributeError:
        client = mqtt.Client()  # paho-mqtt 1.x
    client.connect(mqtt_host, mqtt_port, keepalive=60)
    client.loop_start()

    def publish(event):
        payload = json.dumps(event, ensure_ascii=False)
        print(payload, flush=True)
        client.publish(mqtt_topic, payload, qos=1)

    return publish


def zip_kind(path):
    """'torch' jika zip itu sebenarnya file bobot PyTorch, 'arsip' jika zip biasa, None jika bukan zip valid."""
    try:
        with zipfile.ZipFile(path) as z:
            return "torch" if any(n.endswith("data.pkl") for n in z.namelist()) else "arsip"
    except (zipfile.BadZipFile, OSError):
        return None


def find_weights(value):
    """Cari file bobot (harus berupa FILE, bukan folder).

    Urutan: path apa adanya -> relatif ke folder realtime.py -> jika path itu folder, file .pt di dalamnya
    -> subfolder dari folder realtime.py (mis. folder best/ hasil 'Extract All' yang berisi best.pt).
    """
    here = Path(__file__).resolve().parent
    p = Path(value).expanduser()

    def pick(cands):
        cands = sorted({c.resolve() for c in cands if c.is_file()})
        if len(cands) > 1:
            raise SystemExit(
                "Ditemukan beberapa file bobot:\n  " + "\n  ".join(map(str, cands))
                + '\nPilih salah satu lewat --weights, mis. --weights "C:/proyek/best.pt".'
            )
        return str(cands[0]) if cands else None

    for base in (p, here / p):
        if base.is_file():
            return str(base)
        if base.is_dir():  # path menunjuk ke folder: pakai best.pt (atau satu-satunya .pt) di dalamnya
            found = pick(base.rglob("best.pt")) or pick(base.rglob("*.pt"))
            if found:
                return found

    if not p.is_absolute() and len(p.parts) == 1:  # hanya nama (mis. default best.pt): cari di subfolder
        found = pick(f for pattern in (f"*/{p.name}", f"*/*/{p.name}") for f in here.glob(pattern))
        if found:
            return found
        # zip yang sebenarnya file bobot PyTorch (mis. best.zip karena ekstensi berubah saat diunduh)
        zips = sorted({*here.glob("*.zip"), *Path.cwd().glob("*.zip")})[:20]
        found = pick(z for z in zips if z.stem.lower().startswith(p.stem.lower()) and zip_kind(z) == "torch")
        if found:
            return found

    hint = ""
    for z in sorted({*here.glob("*.zip"), *Path.cwd().glob("*.zip")})[:10]:
        kind = zip_kind(z)
        if kind == "torch":
            hint += (
                f'\n"{z.name}" sebenarnya sudah file bobot PyTorch (format .pt memang berupa zip), jadi JANGAN diekstrak. '
                f'Pakai langsung: --weights "{z}" (skrip menyalinnya ke folder temp sebagai .pt), '
                "atau ganti namanya menjadi best.pt."
            )
        elif kind == "arsip":
            hint += f'\n"{z.name}" adalah arsip zip biasa: ekstrak dulu, lalu pastikan best.pt hasil ekstraknya ada di folder ini.'
    raise SystemExit(
        f"File bobot tidak ditemukan: {value}\n"
        f"Folder yang dicari: {here}\n"
        "Taruh best.pt (hasil training di Kaggle) di folder yang sama dengan realtime.py, "
        'atau beri path lengkapnya, mis. --weights "C:/proyek/best.pt".' + hint
    )


def as_pt(path):
    """Ultralytics menentukan format model dari ekstensi file, jadi zip yang sebenarnya bobot PyTorch
    (mis. best.zip) disalin ke folder temp sebagai .pt. File asli tidak diubah dan tidak perlu di-rename."""
    path = Path(path)
    if path.suffix.lower() == ".pt":
        return str(path)
    kind = zip_kind(path) if path.suffix.lower() == ".zip" else None
    if kind == "arsip":
        raise SystemExit(
            f'"{path.name}" adalah arsip zip biasa, bukan file bobot. '
            "Ekstrak dulu, lalu arahkan --weights ke file .pt di dalamnya."
        )
    if kind != "torch":
        return str(path)  # format hasil export lain (onnx, dll.) dibiarkan apa adanya
    dest = Path(tempfile.gettempdir()) / "strawberry_disease_weights" / f"{path.stem}.pt"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    except OSError as e:
        raise SystemExit(f"Gagal menyalin {path} ke {dest}: {e}")
    print(f'Catatan: "{path.name}" sebenarnya file bobot PyTorch ber-ekstensi .zip; dipakai lewat salinan {dest}', flush=True)
    return str(dest)


def resolve_weights(value):
    """Cari file bobot, lalu pastikan berekstensi .pt agar bisa dimuat Ultralytics."""
    return as_pt(find_weights(value))


def is_live_source(source):
    """True untuk webcam (angka) atau stream jaringan; False untuk file video lokal."""
    return isinstance(source, int) or str(source).lower().startswith(
        ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help=f"file bobot hasil training (default: {DEFAULT_WEIGHTS}, atau best.pt di folder ini)")
    ap.add_argument("--source", default="0", help="indeks webcam (0, 1, ...), URL stream (http/rtsp), atau file video")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35, help="naikkan jika terlalu banyak alarm palsu")
    ap.add_argument("--fps", type=float, default=5, help="batas inferensi per detik untuk webcam/stream (hemat CPU/daya)")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="tampilkan jendela video (--no-show untuk headless)")
    ap.add_argument("--mqtt-host")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--mqtt-topic", default="strawberry/disease")
    args = ap.parse_args()
    if args.weights is None:  # pakai model terbaru jika ada, kalau tidak cari best.pt seperti biasa
        newest = Path(__file__).resolve().parent / DEFAULT_WEIGHTS
        args.weights = str(newest) if newest.is_file() else "best.pt"
    weights = resolve_weights(args.weights)  # sebelum import berat, supaya pesan error langsung muncul

    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names
    source = int(args.source) if args.source.isdigit() else args.source
    live = is_live_source(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Tidak bisa membuka sumber video: {args.source}")

    publish = make_publisher(args.mqtt_host, args.mqtt_port, args.mqtt_topic)
    debouncer = Debouncer(args.window, args.min_hits)
    min_gap = 1.0 / args.fps if live else 0.0  # file video: semua frame diproses
    last = 0.0
    print(f"Model dimuat: {weights} | {len(names)} kelas: {list(names.values())}", flush=True)
    print("Berhenti: tekan q di jendela video, atau Ctrl+C di terminal.", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if not live:  # file video sudah habis
                    print("Video selesai.", flush=True)
                    break
                print("Frame gagal dibaca, menyambungkan ulang...", flush=True)  # stream putus
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(source)
                continue

            now = time.time()
            if now - last < min_gap:
                # Frame tetap dibaca agar buffer kamera tidak menumpuk, tapi tidak diproses.
                # Jendela tetap dipantau supaya tidak "Not Responding" saat --fps kecil.
                if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last = now

            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            stats = {}
            for box in result.boxes:
                name = names[int(box.cls)]
                s = stats.setdefault(name, {"count": 0, "max_conf": 0.0})
                s["count"] += 1
                s["max_conf"] = round(max(s["max_conf"], float(box.conf)), 3)

            appeared, cleared = debouncer.update(stats.keys())
            stamp = datetime.now().isoformat(timespec="seconds")
            for name in sorted(appeared):
                publish({"time": stamp, "event": "detected", "disease": name, **stats.get(name, {})})
            for name in sorted(cleared):
                publish({"time": stamp, "event": "cleared", "disease": name})

            if args.show:
                cv2.imshow("Strawberry disease (tekan q untuk keluar)", result.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Dihentikan.", flush=True)
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()