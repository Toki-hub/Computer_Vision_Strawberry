"""
Deteksi tingkat kematangan strawberry secara real time dari kamera.

Model: results/best.pt di folder skrip ini (atau best.zip hasil unduhan Kaggle di folder yang sama), hasil
notebook strawberry_ripeness_kaggle.ipynb (kelas ripe / unripe). Skrip ini mandiri, tidak bergantung pada
realtime.py model penyakit. Jalankan dari folder proyek:

    # webcam laptop / USB
    python kematangan/realtime_kematangan.py

    # kamera IP / ESP32-CAM, tanpa jendela (perangkat headless)
    python kematangan/realtime_kematangan.py --source http://192.168.1.50:81/stream --fps 1 --no-show

    # kirim event ke broker MQTT (pip install paho-mqtt)
    python kematangan/realtime_kematangan.py --source 0 --mqtt-host 192.168.1.10

Selain kelas dari model (matang / mentah), tiap buah diberi persentase piksel merah di bagian tengah
kotaknya, lalu dijadikan tingkat: mentah (< ambang bawah), setengah matang, atau matang (>= ambang atas).
Dataset tidak punya label "setengah matang", jadi tingkat itu hanya perkiraan dari warna; atur ambangnya
lewat --levels sesuai hasil sel 8 notebook.

Event dikirim seperti realtime.py: sebuah kelas dianggap "terdeteksi" jika muncul di >= --min-hits dari
--window frame terakhir, dan "hilang" jika tidak muncul sama sekali dalam --window frame terakhir.
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

HERE = Path(__file__).resolve().parent
# results/best.pt, atau best.zip di folder ini (file .pt yang ekstensinya berubah saat diunduh dari Kaggle)
DEFAULT_WEIGHTS = (HERE / "results" / "best.pt", HERE / "best.zip")
LABEL_ID = {"ripe": "matang", "unripe": "mentah"}
BOX_COLOR = {"ripe": (40, 40, 230), "unripe": (60, 190, 60)}  # BGR


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


def make_publisher(mqtt_host=None, mqtt_port=1883, mqtt_topic="strawberry/ripeness"):
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


def is_live_source(source):
    """True untuk webcam (angka) atau stream jaringan; False untuk file video lokal."""
    return isinstance(source, int) or str(source).lower().startswith(
        ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")
    )


def resolve_weights(value):
    """Path file bobot yang bisa dimuat Ultralytics. Zip yang sebenarnya bobot PyTorch (mis. best.zip)
    disalin ke folder temp sebagai .pt; file aslinya tidak diubah."""
    path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f"File bobot tidak ditemukan: {path}")
    if path.suffix.lower() != ".zip":
        return str(path)
    try:
        with zipfile.ZipFile(path) as z:
            is_torch = any(n.endswith("data.pkl") for n in z.namelist())
    except zipfile.BadZipFile:
        raise SystemExit(f'"{path.name}" bukan zip yang valid.')
    if not is_torch:
        raise SystemExit(f'"{path.name}" adalah arsip zip biasa: ekstrak dulu, lalu arahkan --weights ke file .pt di dalamnya.')
    dest = Path(tempfile.gettempdir()) / "strawberry_ripeness_weights" / f"{path.stem}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    return str(dest)


def red_ratio(crop):
    """Fraksi piksel merah di elips tengah kotak (BGR). None jika kotak terlalu kecil.
    Rumusnya sama dengan sel 8 notebook, supaya ambang --levels bisa dicek di sana."""
    import cv2
    import numpy as np

    h, w = crop.shape[:2]
    if h < 6 or w < 6:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, int(w * 0.35)), max(1, int(h * 0.35))), 0, 0, 360, 255, -1)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fruit = (mask > 0) & (V >= 40)  # abaikan bayangan yang terlalu gelap
    n = int(fruit.sum())
    if n == 0:
        return None
    red = fruit & ((H <= 10) | (H >= 160)) & (S >= 90)
    return float(red.sum()) / n


def level_of(ratio, levels):
    if ratio is None:
        return None
    return "mentah" if ratio < levels[0] else ("setengah matang" if ratio < levels[1] else "matang")


def draw(frame, dets, counts):
    import cv2

    img = frame.copy()
    for d in dets:
        x0, y0, x1, y1 = d["xyxy"]
        color = BOX_COLOR.get(d["name"], (200, 160, 40))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        text = f"{LABEL_ID.get(d['name'], d['name'])} {d['conf']:.2f}"
        if d["red"] is not None:
            text += f" | merah {d['red']:.0%} ({d['level']})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y0 - 4 if y0 - th - 8 >= 0 else y1 + th + 4
        cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 4, ty + 3), color, -1)
        cv2.putText(img, text, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    header = "  ".join(f"{LABEL_ID.get(k, k)}: {v}" for k, v in counts.items()) or "tidak ada buah"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="file bobot model kematangan (default: results/best.pt, lalu best.zip di folder ini)")
    ap.add_argument("--source", default="0", help="indeks webcam (0, 1, ...), URL stream (http/rtsp), atau file video")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4, help="naikkan jika ada benda lain yang dikira buah")
    ap.add_argument("--fps", type=float, default=5, help="batas inferensi per detik untuk webcam/stream (hemat CPU/daya)")
    ap.add_argument("--levels", type=float, nargs=2, default=(0.3, 0.7), metavar=("BAWAH", "ATAS"),
                    help="ambang fraksi merah: < BAWAH mentah, < ATAS setengah matang, sisanya matang")
    ap.add_argument("--color", action=argparse.BooleanOptionalAction, default=True, help="hitung persentase warna merah (--no-color untuk mematikan)")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="tampilkan jendela video (--no-show untuk headless)")
    ap.add_argument("--mqtt-host")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--mqtt-topic", default="strawberry/ripeness")
    args = ap.parse_args()
    if args.weights is None:
        found = [p for p in DEFAULT_WEIGHTS if p.is_file()]
        if not found:
            raise SystemExit(
                "Model kematangan belum ada di " + " atau ".join(map(str, DEFAULT_WEIGHTS)) + "\n"
                "Latih dulu dengan strawberry_ripeness_kaggle.ipynb di Kaggle, unduh best.pt dari panel Output, "
                'lalu taruh di folder results/ (atau beri path-nya lewat --weights "C:/.../best.pt").'
            )
        args.weights = str(found[0])
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
                if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last = now

            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            dets, stats = [], {}
            for box in result.boxes:
                name = names[int(box.cls)]
                x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
                red = red_ratio(frame[max(0, y0):y1, max(0, x0):x1]) if args.color else None
                dets.append({"name": name, "conf": float(box.conf), "xyxy": (x0, y0, x1, y1),
                             "red": red, "level": level_of(red, args.levels)})
                s = stats.setdefault(name, {"count": 0, "max_conf": 0.0, "levels": {}})
                s["count"] += 1
                s["max_conf"] = round(max(s["max_conf"], float(box.conf)), 3)
                if red is not None:
                    s["levels"][dets[-1]["level"]] = s["levels"].get(dets[-1]["level"], 0) + 1

            appeared, cleared = debouncer.update(stats.keys())
            stamp = datetime.now().isoformat(timespec="seconds")
            for name in sorted(appeared):
                publish({"time": stamp, "event": "detected", "ripeness": name, "label": LABEL_ID.get(name, name), **stats.get(name, {})})
            for name in sorted(cleared):
                publish({"time": stamp, "event": "cleared", "ripeness": name, "label": LABEL_ID.get(name, name)})

            if args.show:
                counts = {n: stats[n]["count"] for n in names.values() if n in stats}
                cv2.imshow("Kematangan strawberry (tekan q untuk keluar)", draw(frame, dets, counts))
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
