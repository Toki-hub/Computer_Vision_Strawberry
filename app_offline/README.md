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
- **Mesin**: GPU atau CPU. GPU otomatis dinonaktifkan kalau HP tidak mendukung WebGPU.
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

## Catatan teknis

- Hasil deteksi sudah dibandingkan dengan model `.pt` di Python pada foto yang sama: kotak dan kelasnya sama, dan keyakinan hanya berbeda ±0,02.
- Kecepatan bergantung pada HP: dengan WebGPU biasanya beberapa frame per detik; dengan CPU lebih lambat (sekitar 1 detik per frame).
- Semua dijalankan di HP dan tidak ada foto yang dikirim ke mana pun.
- Versi yang lebih cepat (kotak tengah 416 px, model di thread terpisah) tersimpan di riwayat git, commit `09c72a8`.
