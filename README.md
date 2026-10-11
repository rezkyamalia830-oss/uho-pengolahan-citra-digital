
# Mini Project: Deteksi Tanda Tangan Kepala Sekolah

## Deskripsi
Program ini mendeteksi keberadaan tanda tangan kepala sekolah pada area tertentu dari gambar ijazah menggunakan OpenCV. Deteksi dilakukan berdasarkan jumlah dan rasio piksel foreground setelah thresholding dan morphological operation.

## Metode
1. Crop area tanda tangan (ROI).
2. Konversi gambar menjadi grayscale.
3. Global Threshold untuk memisahkan piksel gelap dari latar belakang.
4. Otsu Threshold untuk menentukan ambang secara otomatis.
5. Adaptive Threshold untuk menangani pencahayaan yang tidak merata.
6. Morphological Opening untuk mengurangi noise kecil.
7. Morphological Closing untuk menyambungkan goresan yang terputus.
8. Menghitung piksel foreground dan menentukan status tanda tangan.

## Struktur Folder
```text
segmentasi/
├── deteksi_tanda_tangan.py
├── contoh_ijazah/
│   ├── foto 1.jpg
│   ├── foto 2.jpg
│   ├── foto 3.jpg
│   └── foto 4.jpg
└── hasil_tanda_tangan/
```

## Persiapan
Pastikan Python sudah terpasang. Instal pustaka yang diperlukan melalui terminal VS Code:

```bash
python -m pip install opencv-python numpy matplotlib
```

Masukkan gambar pengujian ke folder `contoh_ijazah`.

## Cara Menjalankan
1. Buka folder proyek di Visual Studio Code.
2. Buka `deteksi_tanda_tangan.py`.
3. Sesuaikan nilai `ROI` dengan posisi tanda tangan pada gambar.
4. Sesuaikan nama file dan label pengujian pada `DATA_UJI`.
5. Sesuaikan `BATAS_MIN` dan `BATAS_MAKS` berdasarkan hasil pengujian.
6. Jalankan program dengan tombol **Run Python File** atau melalui terminal:

```bash
python deteksi_tanda_tangan.py
```

## Output
Program menampilkan perbandingan crop asli, grayscale, dan hasil segmentasi dari tiga metode thresholding setelah morphology.

Hasil gambar disimpan pada folder `hasil_tanda_tangan`. Terminal menampilkan jumlah piksel foreground, rasio foreground, status deteksi, dan akurasi pengujian Otsu.

## Analisis
Thresholding diperlukan untuk memisahkan tinta tanda tangan dari latar belakang sehingga area foreground dapat dihitung.

Ambang yang terlalu tinggi pada global threshold dengan inversi dapat membuat terlalu banyak piksel menjadi foreground. Ambang yang terlalu rendah dapat menghilangkan goresan tanda tangan. Adaptive threshold membantu ketika pencahayaan tidak merata, sedangkan Otsu menentukan ambang otomatis berdasarkan distribusi intensitas.

Opening mengurangi noise kecil, sedangkan closing membantu menghubungkan goresan yang terputus. Namun, hasil deteksi tetap dipengaruhi oleh kualitas gambar, posisi crop, dan nilai batas rasio foreground.

## Batasan
Program menggunakan aturan sederhana berdasarkan rasio foreground, bukan model pengenalan tanda tangan. Teks, cap, garis, atau noda di dalam ROI dapat menyebabkan deteksi keliru. Nilai ROI dan ambang harus diuji menggunakan beberapa gambar bertanda tangan dan tanpa tanda tangan.




# Mini Project: Prototype Verifikasi Ijazah

## Deskripsi
Prototype ini memproses citra ijazah untuk membaca nomor ijazah menggunakan OCR dan menentukan keberadaan tanda tangan kepala sekolah menggunakan thresholding, morphology, dan analisis piksel foreground.

## Pipeline
1. Membaca citra ijazah.
2. Mengubah citra menjadi grayscale.
3. Memisahkan area nomor ijazah dan tanda tangan.
4. Menguji empat metode enhancement untuk area nomor:
   - Grayscale
   - CLAHE
   - Contrast Stretching
   - Histogram Equalization
5. Membaca nomor ijazah menggunakan Tesseract OCR.
6. Menghitung CER jika nomor referensi tersedia.
7. Menerapkan Otsu thresholding dan morphology pada area tanda tangan.
8. Menghasilkan status `PRESENT` atau `ABSENT`.

## Struktur Folder
```text
prototype-verifikasi-ijazah/
├── prototype_verifikasi_ijazah.py
├── contoh_ijazah/
│   └── ijazah_001.jpg
├── hasil_verifikasi/
└── README.md
```

## Persiapan
Instal Python dan Tesseract OCR. Instal pustaka Python:

```bash
python -m pip install opencv-python numpy pytesseract
```

Jika Tesseract tidak terdeteksi otomatis, sesuaikan lokasi `tesseract.exe` di dalam kode.

## Cara Menjalankan
1. Simpan gambar ijazah dalam folder `contoh_ijazah`.
2. Buka file `prototype_verifikasi_ijazah.py`.
3. Sesuaikan `PATH_GAMBAR` dengan gambar yang ingin diuji.
4. Sesuaikan koordinat `ROI_NOMOR` dan `ROI_TANDA_TANGAN`.
5. Isi `NOMOR_BENAR` dengan nomor ijazah yang sebenarnya agar CER dapat dihitung.
6. Jalankan file melalui tombol Run Python File di VS Code atau terminal:

```bash
python prototype_verifikasi_ijazah.py
```

## Output
Program menampilkan nomor ijazah hasil OCR, status tanda tangan, metode enhancement yang dipilih, nilai CER, dan jumlah piksel foreground.

Hasil citra tersimpan di folder `hasil_verifikasi`.

## Evaluasi CER
Character Error Rate dihitung menggunakan:

CER = (S + D + I) / N

Keterangan:
- S: jumlah karakter yang salah diganti.
- D: jumlah karakter yang terhapus.
- I: jumlah karakter yang disisipkan.
- N: jumlah karakter pada nomor referensi.

Semakin kecil CER, semakin baik hasil OCR. CER sebesar 0 berarti hasil OCR sama persis dengan nomor referensi. Metode enhancement paling efektif ditentukan dari CER terkecil pada data uji yang sama.

## Analisis Metode
CLAHE meningkatkan kontras lokal dan dapat membantu ketika pencahayaan tidak merata. Contrast Stretching memperluas rentang intensitas, sedangkan Histogram Equalization menyebarkan intensitas berdasarkan histogram. Namun, tidak ada satu metode yang selalu terbaik untuk semua citra.

Otsu digunakan untuk menentukan ambang segmentasi secara otomatis. Morphological Opening mengurangi noise kecil, sedangkan Closing membantu menyambungkan goresan tanda tangan yang terputus.

## Batasan
Akurasi OCR dipengaruhi kualitas gambar, koordinat ROI, jenis huruf, dan konfigurasi OCR. Deteksi tanda tangan menggunakan rasio foreground merupakan aturan sederhana sehingga cap, tulisan, atau noda dapat menyebabkan kesalahan. Hasil harus dievaluasi menggunakan beberapa gambar dengan label yang diketahui.
