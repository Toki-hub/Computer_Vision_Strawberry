# Aplikasi HP offline: Cek Strawberry

Deteksi penyakit dan kematangan strawberry **langsung di HP**, tanpa laptop dan tanpa internet (setelah dibuka sekali). Kedua model dijalankan di browser HP dengan onnxruntime-web: memakai GPU HP (WebGPU) kalau ada, dan otomatis pindah ke CPU kalau tidak.

| Isi | Fungsi |
|---|---|
| `www/` | **folder yang diunggah ke hosting** (aplikasi, model ONNX, onnxruntime-web; total ±100 MB) |
| `build.py` | membuat ulang `www/` dari model terbaru |

## 1. Unggah ke hosting (sekali)

Browser HP hanya mengizinkan kamera dan mode offline di alamat **https://**, jadi folder `www/` perlu diunggah ke hosting gratis.

**Netlify (paling mudah):**
1. Buka https://app.netlify.com/drop dan masuk/daftar (gratis). Tanpa akun, situsnya dihapus setelah 1 jam.
2. Seret folder `app_offline/www` ke halaman itu, lalu tunggu unggahan ±100 MB selesai.
3. Anda mendapat alamat seperti `https://nama-acak.netlify.app`. Nama situs bisa diganti di Site settings.

**GitHub Pages** juga bisa, tapi lewat `git push` (unggah lewat web dibatasi 25 MB per file, sedangkan model 36 MB).

Siapa pun yang tahu alamatnya bisa membuka aplikasi dan mengunduh file modelnya.

## 2. Pasang di HP Android

1. Buka alamat tadi di **Chrome**, dengan internet (Wi-Fi disarankan, ±100 MB).
2. Tunggu sampai model selesai dimuat dan penanda di atas berubah hijau: **Siap offline**.
3. Menu Chrome (⋮) → **Instal aplikasi** / **Tambahkan ke layar utama**.
4. Setelah itu aplikasi bisa dibuka dari layar utama, termasuk di kebun tanpa sinyal.

## Fitur

- **Penyakit / Kematangan / Keduanya**: model yang dipakai.
- **Ambang**: batas keyakinan minimal.
- **Ukuran**: ukuran gambar yang dianalisis. 320 Cepat, **416 Sedang (bawaan)**, 640 Akurat. Lihat tabel di bawah.
- **Area**: **Kotak tengah (bawaan)** hanya menganalisis kotak persegi yang ditandai bingkai di tengah layar, jadi arahkan daun/buah ke dalam bingkai. Objek jadi ±1,8× lebih besar bagi model, sehingga lebih akurat tanpa tambahan waktu. **Layar penuh** menganalisis seluruh frame, tapi objek jadi kecil.
- **Mesin**: GPU atau CPU. GPU otomatis dinonaktifkan kalau HP tidak mendukung WebGPU. Di CPU, model jalan di thread terpisah supaya tampilan tetap lancar.
- Pojok kanan atas: lama satu analisis, mesin, jumlah thread CPU, dan ukuran (mis. `1.8 s · CPU×4 · 416px`). Kalau tertulis `CPU×1`, mode multi-thread tidak aktif dan analisis jadi sekitar 3–4× lebih lambat.
- Saat kamera diarahkan ke objek lain, kotak lama disembunyikan ("Kamera bergeser, menganalisis ulang...") sampai hasil baru keluar. Goyangan tangan dan perubahan cahaya kamera tidak memicunya.
- **Cahaya**: indikator kecerahan (gelap/redup/silau).
- **Senter**: lampu kilat, kalau kamera mendukung.
- **Galeri**: uji foto yang sudah ada di HP.
- **Simpan foto ke HP**: foto resolusi penuh disimpan ke folder Download, untuk dilabeli dan melatih ulang model.

Mode **Keduanya** menjalankan model kematangan juga pada buah yang sakit, jadi buah busuk bisa ikut berlabel "mentah"/"matang". Untuk memeriksa penyakit, pakai mode **Penyakit**.

## Memperbarui model

Setelah model dilatih ulang dan `best.pt` baru ditaruh di `Diseases Detection/results/` atau `kematangan/results/`:

```
pip install ultralytics onnx onnxslim
python app_offline/build.py --skip-ort
```

Lalu unggah ulang folder `www/` ke situs yang sama (di Netlify: tab Deploys → seret folder lagi). HP yang sudah memasang aplikasi akan mengunduh versi baru saat dibuka dengan internet.

## Pilihan ukuran dan area

Waktu diukur di laptop (CPU 4 thread). Di HP angkanya lebih besar, tapi perbandingannya serupa. Akurasi (mAP50) dihitung untuk HP posisi tegak (frame 720×1280) dengan objek memenuhi lebar layar:

| Pengaturan | Waktu 1 model | Keduanya | mAP50 penyakit | mAP50 kematangan |
|---|---|---|---|---|
| Kotak tengah, 320 | 170 ms | 340 ms | 0.921 | 0.905 |
| **Kotak tengah, 416 (bawaan)** | 273 ms | 533 ms | **0.932** | **0.931** |
| Kotak tengah, 640 | 606 ms | 1.211 ms | 0.933 | 0.951 |
| Layar penuh, 416 | 273 ms | 533 ms | 0.844 | 0.846 |
| Layar penuh, 640 (versi pertama) | 606 ms | 1.211 ms | 0.928 | 0.911 |

Nilai mAP50 diukur pada foto dataset yang mungkin pernah dipakai saat training, jadi angkanya terlalu optimis. Yang penting dibaca adalah selisih antar pengaturan. **Layar penuh, 416** turun jauh karena di frame tegak objeknya hanya tampil ±234 px bagi model. Untuk bercak kecil atau objek yang jauh, pakai Kotak tengah + 640.

## Catatan teknis

- Hasil deteksi sudah dibandingkan dengan model `.pt` di Python pada foto yang sama, di ukuran 320, 416, dan 640: kotak dan kelasnya sama, dan keyakinan hanya berbeda ±0,02.
- Satu file model dipakai untuk semua ukuran (ONNX dengan ukuran input dinamis). Pilihan ukuran diatur di `build.py` (`--sizes`, `--imgsz`).
- Model "dipanaskan" di layar loading, karena analisis pertama selalu paling lama (terutama di GPU).
- Semua dijalankan di HP dan tidak ada foto yang dikirim ke mana pun.
