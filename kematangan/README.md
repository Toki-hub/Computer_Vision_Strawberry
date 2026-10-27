# Model kematangan strawberry

Model kedua, terpisah dari model penyakit (`../realtime.py`, `../results/best.pt`). Kelas: `ripe` (matang) dan `unripe` (mentah).

| File | Isi |
|---|---|
| `strawberry_ripeness_kaggle.ipynb` | notebook training untuk Kaggle |
| `realtime_kematangan.py` | deteksi kematangan realtime dari webcam / ESP32-CAM / video |
| `results/` | taruh `best.pt` dan `run_summary.json` hasil notebook di sini |

## Langkah

1. Kaggle → Create → New Dataset, unggah `strawberry.v1i.yolov8.zip` (ada di folder Downloads).
2. Buat notebook baru dari `strawberry_ripeness_kaggle.ipynb`. Lampirkan dataset tadi, ditambah paket `strawberry_extra` yang sudah ada (opsional, sebagai foto daun dan tanaman lain tanpa buah). Atur GPU dan Internet, lalu jalankan Save & Run All.
3. Unduh `best.pt` dan `run_summary.json` dari Output, lalu taruh di `kematangan/results/`.
4. Dari folder proyek:

```
python kematangan/realtime_kematangan.py
```

Tiap buah diberi kelas dari model, persentase warna merah, dan tingkat (mentah / setengah matang / matang) dari warna itu. Ambang tingkat diatur dengan `--levels 0.3 0.7`. Pada uji awal dengan label asli, median merah buah `ripe` 93% dan `unripe` 0%.
