# Computer Vision Strawberry

Deteksi **penyakit** dan **tingkat kematangan** strawberry dengan YOLO26, dari kamera laptop, ESP32-CAM, atau langsung di HP.

| Folder | Isi |
|---|---|
| `Diseases Detection/` | notebook training model penyakit (Kaggle), `realtime.py` untuk kamera/ESP32, model di `results/best.pt` (7 kelas penyakit) |
| `kematangan/` | notebook training model kematangan (`ripe`/`unripe`), `realtime_kematangan.py`, model di `results/best.pt` |
| `app_hp/` | server di laptop + halaman web untuk menguji kedua model lewat kamera HP (Wi-Fi yang sama) |
| `app_offline/` | aplikasi HP mandiri (PWA, ONNX di browser) yang jalan tanpa laptop dan tanpa internet; folder `www/` siap diunggah ke Netlify |

Penjelasan lengkap ada di README tiap folder.

## Menjalankan cepat

```
pip install -U ultralytics paho-mqtt
python "Diseases Detection/realtime.py"            # penyakit, webcam laptop
python kematangan/realtime_kematangan.py           # kematangan, webcam laptop
python app_hp/server.py                            # uji lewat kamera HP
```

## Data

Dataset dan arsip hasil training (`*.zip`) tidak ikut di repo karena ukurannya besar. Sumbernya:
- Penyakit: Roboflow [strawberry-qnt3p](https://universe.roboflow.com/strawberry-for-japs/strawberry-qnt3p) (Public Domain)
- Kematangan: Roboflow [aliazz/strawberry-fowax](https://universe.roboflow.com/aliazz/strawberry-fowax) (CC BY 4.0)
- Daun/tanaman lain (negatif): PlantDoc
