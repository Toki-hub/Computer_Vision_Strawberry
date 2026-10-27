#!/usr/bin/env python3
"""
Tambahan untuk paket strawberry_extra: negatif strawberry sehat (terutama yang mentah) dan patokan alarm palsu.
Paket strawberry_extra yang lama TIDAK diubah; dua paket baru dibuat di sampingnya:

  strawberry_extra_v2/     (strawberry_extra_v2.zip)
    train|val|test/        neg_ripeness2_*.jpg: foto strawberry matang/mentah tanpa label, dibagi 60/15/25
    split_frozen.csv       pembagian train/val/test tetap untuk SEMUA gambar (dari split_assignment.csv run sebelumnya
                           + negatif baru), supaya test set sama di setiap run
  benchmark_healthy/       (benchmark_healthy.zip)
    images/                foto strawberry sehat yang TIDAK pernah dipakai training, untuk mengukur alarm palsu
                           dengan cara yang sama di setiap run (notebook menemukannya otomatis)

Foto yang kembar (termasuk versi cermin/putar) dikelompokkan: satu kelompok tidak pernah sekaligus berada di patokan
dan di data training.

Contoh:
    python extend_extra.py --extra strawberry_extra.zip --ripeness strawberry_v1i_yolov8_1.zip \
        --split-csv split_assignment.csv --out data_extra_v2
"""
import argparse
import collections
import csv
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

T = getattr(Image, "Transpose", Image)
TRANS = [None, T.FLIP_LEFT_RIGHT, T.FLIP_TOP_BOTTOM, T.ROTATE_90, T.ROTATE_180, T.ROTATE_270, T.TRANSPOSE, T.TRANSVERSE]
DUP_T = 20                      # jarak Hamming (dari 256 bit) <= ini: foto kembar
SPLITS = ("train", "val", "test")


def open_source(path, workdir):
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        dest = Path(workdir) / p.stem
        if not dest.exists():
            print(f"  mengekstrak {p.name} ...", flush=True)
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
        return dest
    if p.is_dir():
        return p
    raise SystemExit(f"Tidak ditemukan: {path}")


def find_root(base, need_yaml=True):
    for d in sorted([Path(base), *Path(base).rglob("*")]):
        if d.is_dir() and (d / "train" / "images").is_dir() and (d / "test" / "images").is_dir() \
                and ((d / "val" / "images").is_dir() or (d / "valid" / "images").is_dir()) \
                and (not need_yaml or (d / "data.yaml").is_file()):
            return d
    raise SystemExit(f"Struktur train/val/test tidak ditemukan di {base}")


def dhash(im):
    a = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def hash1(p):
    with Image.open(p) as im:
        return dhash(im).astype(np.float32) * 2 - 1


def hash8(p):
    with Image.open(p) as im:
        im = im.convert("RGB")
        im.thumbnail((640, 640))
        return np.stack([dhash(im if t is None else im.transpose(t)) for t in TRANS]).astype(np.float32) * 2 - 1


def min_dist8(Ha, Hb):
    """Jarak Hamming terkecil antar 8 orientasi, simetris. Ha (n,8,256), Hb (m,8,256) -> (n,m)."""
    def one(X, Y):
        D = np.full((len(X), len(Y)), 256, np.float32)
        for t in range(X.shape[1]):
            np.minimum(D, (256 - X[:, t] @ Y[:, 0].T) / 2, out=D)
        return D
    return np.minimum(one(Ha, Hb), one(Hb, Ha).T)


def union_groups(D, thr):
    n = len(D)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in zip(*np.nonzero(np.triu(D <= thr, 1))):
        par[find(a)] = find(b)
    return [find(i) for i in range(n)]


def ripeness_classes(img):
    lp = img.parent.parent / "labels" / (img.stem + ".txt")
    if not lp.is_file():
        return set()
    return {int(float(l.split()[0])) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", required=True, help="paket strawberry_extra lama (zip atau folder)")
    ap.add_argument("--ripeness", required=True, help="dataset strawberry ripe/unripe YOLO (zip atau folder)")
    ap.add_argument("--split-csv", required=True, help="split_assignment.csv dari run sebelumnya (sel 2b notebook)")
    ap.add_argument("--n-new", type=int, default=300, help="jumlah negatif baru (satu foto per kelompok kembar)")
    ap.add_argument("--n-bench", type=int, default=600, help="jumlah calon foto patokan sebelum dibersihkan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data_extra_v2")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    pack, bench_root = out / "strawberry_extra_v2", out / "benchmark_healthy"
    for d in (pack, bench_root):
        shutil.rmtree(d, ignore_errors=True)
        (out / f"{d.name}.zip").unlink(missing_ok=True)
    work = Path(tempfile.mkdtemp(prefix="extend_extra_"))
    try:
        print("1) Membaca sumber")
        old_root = find_root(open_source(args.extra, work))
        rp_root = find_root(open_source(args.ripeness, work), need_yaml=False)
        old_neg = sorted(old_root.glob("*/images/neg_ripeness_*.jpg"))
        rp_imgs = sorted(rp_root.rglob("images/*.jpg"))
        names = [l.strip() for l in (rp_root / "data.yaml").read_text().splitlines() if l.startswith("names:")] if (rp_root / "data.yaml").is_file() else []
        print(f"  paket lama: {sum(1 for _ in old_root.glob('*/images/*'))} gambar ({len(old_neg)} negatif ripeness) | dataset ripe/unripe: {len(rp_imgs)} foto {names}")

        print("2) Memilih calon patokan (foto yang belum pernah dipakai)")
        U1 = np.stack([hash1(p) for p in old_neg])
        R1 = np.stack([hash1(p) for p in rp_imgs])
        d1 = ((256 - R1 @ U1.T) / 2).min(axis=1)
        unseen = [p for p, x in zip(rp_imgs, d1) if x > DUP_T]
        random.Random(args.seed).shuffle(unseen)
        cand = unseen[: args.n_bench]

        print("3) Mengelompokkan foto kembar (8 orientasi)")
        H = np.stack([hash8(p) for p in rp_imgs])
        HU = np.stack([hash8(p) for p in old_neg])
        gid = union_groups(min_dist8(H, H), DUP_T)
        used = min_dist8(H, HU).min(axis=1) <= DUP_T            # foto yang kembar dengan negatif lama
        used_g = {gid[i] for i in np.nonzero(used)[0]}
        idx = {p: i for i, p in enumerate(rp_imgs)}
        bench = [p for p in cand if gid[idx[p]] not in used_g]
        bench_g = {gid[idx[p]] for p in cand}
        print(f"  calon patokan {len(cand)} -> {len(bench)} setelah membuang yang sekelompok dengan negatif lama")

        print("4) Memilih negatif baru (satu foto per kelompok, utamakan yang berisi buah mentah)")
        pool = collections.defaultdict(list)
        for i, p in enumerate(rp_imgs):
            if gid[i] not in used_g and gid[i] not in bench_g:
                pool[gid[i]].append(p)
        rng = random.Random(args.seed + 1)
        UNRIPE = 1                                              # names: ['ripe', 'unripe']
        reps, with_unripe, without = {}, [], []
        for g, members in sorted(pool.items()):
            members = sorted(members)
            rng.shuffle(members)
            un = [m for m in members if UNRIPE in ripeness_classes(m)]
            reps[g] = un[0] if un else members[0]
            (with_unripe if un else without).append(g)
        rng.shuffle(with_unripe)
        rng.shuffle(without)
        chosen = [reps[g] for g in (with_unripe + without)[: args.n_new]]
        n_un = sum(1 for p in chosen if UNRIPE in ripeness_classes(p))
        print(f"  kolam: {sum(map(len, pool.values()))} foto dalam {len(pool)} kelompok | terpilih {len(chosen)} ({n_un} berisi buah mentah)")
        if len(chosen) < args.n_new:
            print(f"  PERINGATAN: kolam hanya cukup untuk {len(chosen)} negatif")

        print("5) Menulis paket")
        for s in SPLITS:
            (pack / s / "images").mkdir(parents=True, exist_ok=True)
            (pack / s / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_root / "data.yaml", pack / "data.yaml")
        rng2 = random.Random(args.seed + 2)
        order = list(range(len(chosen)))
        rng2.shuffle(order)
        n = len(chosen)
        n_tr, n_va = round(n * 0.60), round(n * 0.15)
        new_rows = []
        for rank, k in enumerate(order):
            split = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
            name = f"neg_ripeness2_{k:04d}.jpg"
            shutil.copyfile(chosen[k], pack / split / "images" / name)
            (pack / split / "labels" / f"neg_ripeness2_{k:04d}.txt").write_text("", encoding="utf-8")
            new_rows.append((name, split, chosen[k].name))

        bdir = bench_root / "images"
        bdir.mkdir(parents=True)
        with open(bench_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "sumber", "label_kematangan"])
            for k, p in enumerate(bench):
                name = f"bench_healthy_{k:04d}.jpg"
                shutil.copyfile(p, bdir / name)
                cls = sorted(ripeness_classes(p))
                w.writerow([name, p.name, "+".join({0: "ripe", 1: "unripe"}.get(c, str(c)) for c in cls)])

        # pembagian tetap: run sebelumnya (berdasarkan nama file) + negatif baru
        frozen = {}
        with open(args.split_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                name = Path(r["image"].replace("\\", "/")).name
                if name in frozen and frozen[name] != r["split_baru"]:
                    raise SystemExit(f"Nama file ganda dengan split berbeda: {name}")
                frozen[name] = r["split_baru"]
        for name, split, _ in new_rows:
            frozen[name] = split
        with open(pack / "split_frozen.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "split"])
            for name in sorted(frozen):
                w.writerow([name, frozen[name]])

        rep = {}
        rep["v2"] = {
            "negatif_baru": {s: sum(1 for _, x, _ in new_rows if x == s) for s in SPLITS},
            "negatif_baru_berisi_buah_mentah": n_un,
            "patokan_sehat": len(bench),
            "split_frozen_entri": len(frozen),
            "seed": args.seed,
        }
        (pack / "REPORT.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        att = ("Sumber: dataset strawberry ripe/unripe, Roboflow aliazz/strawberry-fowax v1 (CC BY 4.0), "
               "https://universe.roboflow.com/aliazz/strawberry-fowax . Foto disalin tanpa diubah, label dihapus.\n")
        (pack / "ATTRIBUTION.txt").write_text(att, encoding="utf-8")
        (bench_root / "ATTRIBUTION.txt").write_text(att, encoding="utf-8")
        print(f"  negatif baru: {rep['v2']['negatif_baru']} | patokan: {len(bench)} | split_frozen.csv: {len(frozen)} entri")

        if not args.no_zip:
            for d in (pack, bench_root):
                shutil.make_archive(str(out / d.name), "zip", root_dir=out, base_dir=d.name)
                print(f"Zip: {out / d.name}.zip ({(out / f'{d.name}.zip').stat().st_size / 2**20:.1f} MB)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
