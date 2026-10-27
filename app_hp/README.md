# Aplikasi HP: uji model lewat kamera HP

Laptop menjalankan kedua model (penyakit dan kematangan). HP cukup membuka halaman web di browser, tanpa install apa pun. HP dan laptop harus tersambung ke **Wi-Fi yang sama** (atau HP menjadi hotspot untuk laptop).

## Menjalankan

Dari folder proyek (`Comvis`):

```
python app_hp/server.py
```

Terminal akan menampilkan alamat seperti `https://10.37.53.249:8443`. Buka alamat itu di Chrome (Android) atau Safari (iPhone).

1. **Peringatan "koneksi tidak pribadi / not private"**: muncul karena laptop memakai sertifikat buatan sendiri. Pilih **Lanjutan → Lanjutkan ke situs**. Ini wajib karena browser HP hanya mengizinkan kamera di halaman HTTPS.
2. **Izinkan kamera** saat diminta.
3. Pertama kali dijalankan, Windows mungkin menanyakan izin firewall untuk Python: centang **Private networks**. Kalau HP tetap tidak bisa membuka halaman, pastikan jaringan Wi-Fi laptop berprofil *Private*, bukan *Public*.

Agar terasa seperti aplikasi, pilih menu browser → **Tambahkan ke layar utama**.

## Fitur

| Bagian | Fungsi |
|---|---|
| Penyakit / Kematangan / Keduanya | model yang dipakai |
| Ambang | batas keyakinan minimal (naikkan jika banyak deteksi palsu) |
| Cahaya | kecerahan gambar; merah = terlalu gelap, kuning = redup atau silau |
| Senter | menyalakan lampu kilat HP (hanya muncul jika kamera mendukung, umumnya Android) |
| Galeri | uji foto yang sudah ada di HP |
| Simpan foto untuk dataset | foto resolusi penuh + hasil deteksinya (`.json`) disimpan di laptop, di `app_hp/tangkapan/` |

Simpan foto yang salah terdeteksi atau yang tidak terdeteksi. Foto-foto itu nantinya bisa dilabeli (mis. di Roboflow) lalu dipakai melatih ulang model dengan foto dari kamera sendiri, dalam berbagai kondisi cahaya.

## Catatan

- Kecepatan bergantung pada CPU laptop: sekitar 3 frame/detik untuk satu model dan lebih lambat untuk "Keduanya".
- Siapa pun di Wi-Fi yang sama bisa membuka alamat itu selama server berjalan. Matikan server (Ctrl+C) setelah selesai, terutama di Wi-Fi umum.
- Opsi lain: `--port`, `--imgsz`, `--levels 0.3 0.7`, `--disease ''` (hanya model kematangan), `--ripeness ''` (hanya model penyakit). Lihat `python app_hp/server.py --help`.
