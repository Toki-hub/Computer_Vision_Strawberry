"""
Menyusun folder www/ untuk aplikasi HP offline (PWA): model ONNX + onnxruntime-web + halaman aplikasi.
Jalankan ulang setiap kali model dilatih ulang, lalu unggah ulang isi www/ ke hosting.

    pip install ultralytics onnx onnxslim
    python app_offline/build.py

Kebutuhan: Node.js (untuk `npm pack onnxruntime-web`), dan bobot model di:
    Diseases Detection/results/best.pt   (penyakit)
    kematangan/results/best.pt           (kematangan)
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WWW = HERE / "www"
ORT_VERSION = "1.30.0"
ORT_FILES = ("ort.min.js", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm")
MODELS = {
    "penyakit": ROOT / "Diseases Detection" / "results" / "best.pt",
    "kematangan": ROOT / "kematangan" / "results" / "best.pt",
}
LABELS = {
    "angular_leafspot": "bercak daun bersudut",
    "anthracnose_fruit_rot": "busuk buah antraknosa",
    "blossom_blight": "hawar bunga",
    "gray_mold": "kapang abu-abu",
    "leaf_spot": "bercak daun",
    "powdery_mildew_fruit": "embun tepung buah",
    "powdery_mildew_leaf": "embun tepung daun",
    "ripe": "matang",
    "unripe": "mentah",
}


def export_models(imgsz, work):
    from ultralytics import YOLO

    out = {}
    for task, src in MODELS.items():
        if not src.is_file():
            raise SystemExit(f"Bobot model {task} tidak ditemukan: {src}")
        tmp = work / f"{task}.pt"
        shutil.copyfile(src, tmp)  # export ditulis di samping file .pt, jadi pakai salinan
        model = YOLO(str(tmp))
        # dynamic=True: satu file model bisa dijalankan di beberapa ukuran input (pilihan "Kecepatan" di aplikasi)
        onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=True))
        dest = WWW / "models" / f"{task}.onnx"
        shutil.copyfile(onnx_path, dest)
        names = [model.names[i] for i in sorted(model.names)]
        out[task] = {"file": f"models/{task}.onnx", "names": names, "labels": [LABELS.get(n, n) for n in names],
                     "bytes": dest.stat().st_size}
        print(f"  {task}: {dest.stat().st_size / 2**20:.1f} MB, kelas {names}")
    return out


def fetch_ort(work):
    subprocess.run(f"npm pack onnxruntime-web@{ORT_VERSION} --silent", cwd=work, shell=True, check=True, capture_output=True)
    with tarfile.open(work / f"onnxruntime-web-{ORT_VERSION}.tgz") as t:
        t.extractall(work, filter="data")
    for f in ORT_FILES:
        shutil.copyfile(work / "package" / "dist" / f, WWW / "ort" / f)
    print(f"  onnxruntime-web {ORT_VERSION}: {', '.join(ORT_FILES)}")


def make_icons():
    from PIL import Image, ImageDraw

    for size in (192, 512):
        s = size / 64
        im = Image.new("RGB", (size, size), (14, 15, 16))
        d = ImageDraw.Draw(im)
        d.ellipse([15 * s, 22 * s, 49 * s, 56 * s], fill=(229, 72, 77))
        d.polygon([(p[0] * s, p[1] * s) for p in [(32, 26), (20, 15), (27, 22), (17, 25), (32, 27), (47, 25), (37, 22), (44, 15)]],
                  fill=(63, 185, 80))
        for cx, cy in [(26, 34), (38, 34), (32, 40), (25, 44), (39, 44), (32, 49)]:
            d.ellipse([(cx - 1.3) * s, (cy - 1.3) * s, (cx + 1.3) * s, (cy + 1.3) * s], fill=(255, 217, 160))
        im.save(WWW / f"icon-{size}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imgsz", type=int, default=416, help="ukuran input bawaan di aplikasi (model dilatih di 640)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 416, 640],
                    help="pilihan ukuran input di aplikasi: terkecil = Cepat, terbesar = Akurat (kelipatan 32)")
    ap.add_argument("--skip-ort", action="store_true", help="jangan unduh ulang onnxruntime-web jika sudah ada")
    args = ap.parse_args()
    sizes = sorted(set(args.sizes) | {args.imgsz})
    if any(s % 32 for s in sizes):
        raise SystemExit(f"Ukuran input harus kelipatan 32: {sizes}")

    (WWW / "models").mkdir(parents=True, exist_ok=True)
    (WWW / "ort").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="app_offline_") as tmp:
        work = Path(tmp)
        print("1) Export model ke ONNX")
        models = export_models(max(sizes), work)
        if not (args.skip_ort and all((WWW / "ort" / f).is_file() for f in ORT_FILES)):
            print("2) Mengambil onnxruntime-web")
            fetch_ort(work)
    make_icons()

    # versi = hash isi model + mesin + halaman, supaya HP mengunduh ulang hanya jika ada yang berubah
    h = hashlib.sha256()
    for f in sorted(WWW.rglob("*")):
        if f.is_file() and f.name not in ("models.json", "sw.js"):
            h.update(f.relative_to(WWW).as_posix().encode())
            h.update(f.read_bytes())
    version = h.hexdigest()[:12]
    meta = {"version": version, "imgsz": args.imgsz, "sizes": sizes, "ort": ORT_VERSION, "models": models}
    (WWW / "models.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    sw = WWW / "sw.js"
    text = sw.read_text(encoding="utf-8")
    start = text.index('const VERSION = "') + len('const VERSION = "')
    sw.write_text(text[:start] + version + text[text.index('"', start):], encoding="utf-8")
    total = sum(f.stat().st_size for f in WWW.rglob("*") if f.is_file())
    print(f"Selesai: {WWW} ({total / 2**20:.0f} MB), versi {version}")


if __name__ == "__main__":
    main()
