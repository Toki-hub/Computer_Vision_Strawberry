"""
Latih detektor penyakit strawberry dengan Ultralytics YOLO.

Pemakaian (disarankan di GPU, mis. Google Colab / Kaggle):
    pip install -U ultralytics
    python train.py --data-dir "Strawberry Disease Data"

Versi lebih ringan untuk perangkat edge:
    python train.py --data-dir "Strawberry Disease Data" --model yolo26n.pt --imgsz 416

Catatan:
- Label dataset berformat poligon (segmentasi). Untuk model deteksi, Ultralytics otomatis
  mengubah poligon menjadi bounding box saat memuat label.
- Dataset tidak menyertakan data.yaml, jadi skrip ini membuatnya sendiri.
"""
import argparse
from pathlib import Path

import yaml

# ID kelas di file label mengikuti urutan alfabet nama kelas (sudah dicocokkan dengan
# awalan nama file gambar). Jangan ubah urutan ini.
CLASS_NAMES = [
    "angular_leafspot",       # 0
    "anthracnose_fruit_rot",  # 1
    "blossom_blight",         # 2
    "gray_mold",              # 3
    "leaf_spot",              # 4
    "powdery_mildew_fruit",   # 5
    "powdery_mildew_leaf",    # 6
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="Strawberry Disease Data", help="folder berisi train/ val/ test/")
    ap.add_argument("--model", default="yolo26s.pt", help="yolo26n.pt (ringan) ... yolo26s.pt / yolo26m.pt (lebih akurat)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="gambar asli 419 px; 416 lebih cepat, 640 biasanya lebih teliti")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 = GPU pertama, cpu = CPU; kosong = otomatis")
    ap.add_argument("--name", default="strawberry")
    args = ap.parse_args()

    root = Path(args.data_dir).resolve()
    for sub in ("train/images", "val/images", "test/images"):
        if not (root / sub).is_dir():
            raise SystemExit(f"Folder tidak ditemukan: {root / sub}")

    data_yaml = Path("data.yaml").resolve()
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),  # path absolut supaya tidak tergantung dari mana skrip dijalankan
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": dict(enumerate(CLASS_NAMES)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    from ultralytics import YOLO  # diimpor di sini supaya `--help` tetap cepat

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=30,   # berhenti lebih awal kalau val tidak membaik 30 epoch
        flipud=0.5,    # daun/buah tidak punya orientasi atas-bawah yang tetap
        name=args.name,
    )

    # Evaluasi akhir di split test (dipakai hanya sekali, setelah training selesai)
    best = Path(model.trainer.best)
    print(f"\nBobot terbaik: {best}")
    metrics = YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device)
    print(f"\n[TEST] mAP50-95 = {metrics.box.map:.3f} | mAP50 = {metrics.box.map50:.3f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:24s} mAP50-95 = {metrics.box.maps[i]:.3f}")


if __name__ == "__main__":  # wajib di Windows agar dataloader tidak error
    main()
