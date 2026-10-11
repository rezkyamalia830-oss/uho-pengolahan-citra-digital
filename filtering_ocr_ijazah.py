
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import re
from pathlib import Path

# ==========================================
# PENGATURAN
# ==========================================

PATH_GAMBAR = "contoh_ijazah/foto 1.jpg"

# Sesuaikan dengan posisi nomor ijazah pada gambar
# Format: x1, y1, x2, y2
ROI = (300, 100, 600, 180)

# Masukkan nomor ijazah yang sebenarnya
GROUND_TRUTH = "GANTI_DENGAN_NOMOR_IJAZAH"

FOLDER_OUTPUT = Path("hasil_filtering_ocr")

# Jika Tesseract tidak terdeteksi di Windows,
# aktifkan baris berikut dan sesuaikan lokasinya:
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )


# ==========================================
# 1. FILTERING DAN SHARPENING
# ==========================================

def mean_filter(gray):
    return cv2.blur(gray, (3, 3))


def median_filter(gray):
    return cv2.medianBlur(gray, 3)


def gaussian_filter(gray):
    return cv2.GaussianBlur(gray, (3, 3), 0)


def sharpening(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Unsharp masking untuk mempertegas tepi karakter
    hasil = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    return hasil


# ==========================================
# 2. OCR
# ==========================================

def jalankan_ocr(gray):
    # Perbesar citra agar karakter kecil lebih mudah dibaca
    tinggi, lebar = gray.shape

    gambar_besar = cv2.resize(
        gray,
        (lebar * 3, tinggi * 3),
        interpolation=cv2.INTER_CUBIC
    )

    konfigurasi = "--oem 3 --psm 7"

    teks = pytesseract.image_to_string(
        gambar_besar,
        config=konfigurasi
    )

    return teks.strip()


# ==========================================
# 3. PERHITUNGAN AKURASI OCR
# ==========================================

def normalisasi(teks):
    return re.sub(r"[^A-Z0-9]", "", teks.upper())


def hitung_akurasi(prediksi, ground_truth):
    prediksi = normalisasi(prediksi)
    ground_truth = normalisasi(ground_truth)

    if not ground_truth:
        return 0, 0, 0

    # Menghitung jarak edit untuk mengetahui kesalahan karakter
    dp = list(range(len(prediksi) + 1))

    for i, karakter_benar in enumerate(ground_truth, start=1):
        baris_baru = [i]

        for j, karakter_prediksi in enumerate(prediksi, start=1):
            biaya = 0 if karakter_benar == karakter_prediksi else 1

            baris_baru.append(min(
                dp[j] + 1,
                baris_baru[j - 1] + 1,
                dp[j - 1] + biaya
            ))

        dp = baris_baru

    jarak = dp[-1]
    karakter_benar = max(0, len(ground_truth) - jarak)

    akurasi = max(
        0,
        (1 - jarak / len(ground_truth)) * 100
    )

    return karakter_benar, len(ground_truth), akurasi


# ==========================================
# 4. MEMBACA DAN CROP GAMBAR
# ==========================================

gambar = cv2.imread(PATH_GAMBAR)

if gambar is None:
    raise FileNotFoundError(
        f"Gambar tidak ditemukan: {PATH_GAMBAR}"
    )

tinggi, lebar = gambar.shape[:2]
x1, y1, x2, y2 = ROI

if not (0 <= x1 < x2 <= lebar and 0 <= y1 < y2 <= tinggi):
    raise ValueError(
        f"ROI tidak sesuai ukuran gambar {lebar}x{tinggi}. "
        "Silakan sesuaikan koordinat."
    )

crop = gambar[y1:y2, x1:x2]

gray = cv2.cvtColor(
    crop,
    cv2.COLOR_BGR2GRAY
)


# ==========================================
# 5. TERAPKAN SEMUA METODE
# ==========================================

hasil_metode = {
    "Original": gray,
    "Mean Filter": mean_filter(gray),
    "Median Filter": median_filter(gray),
    "Gaussian Filter": gaussian_filter(gray),
    "Sharpening": sharpening(gray)
}


# ==========================================
# 6. TAMPILKAN CITRA SEBELUM DAN SESUDAH
# ==========================================

FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 5, figsize=(18, 7))

for i, (nama, citra) in enumerate(hasil_metode.items()):
    axes[0, i].imshow(citra, cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(nama)
    axes[0, i].axis("off")

    histogram = cv2.calcHist(
        [citra], [0], None, [256], [0, 256]
    )

    axes[1, i].plot(histogram, color="black")
    axes[1, i].set_title("Histogram")
    axes[1, i].set_xlabel("Intensitas")
    axes[1, i].set_ylabel("Jumlah Piksel")
    axes[1, i].set_xlim(0, 255)

plt.tight_layout()

plt.savefig(
    FOLDER_OUTPUT / "perbandingan_filter_histogram.png",
    dpi=150
)

plt.show()


# ==========================================
# 7. JALANKAN OCR DAN BANDINGKAN AKURASI
# ==========================================

print("\nHASIL PERBANDINGAN OCR")
print("-" * 100)
print(
    f"{'Metode':<18} | {'Hasil OCR':<25} | "
    f"{'Karakter Benar':<15} | {'Akurasi':<10}"
)
print("-" * 100)

if GROUND_TRUTH == "GANTI_DENGAN_NOMOR_IJAZAH":
    print("Isi GROUND_TRUTH dengan nomor ijazah sebenarnya terlebih dahulu.")
else:
    for nama, citra in hasil_metode.items():
        hasil_ocr = jalankan_ocr(citra)

        benar, total, akurasi = hitung_akurasi(
            hasil_ocr,
            GROUND_TRUTH
        )

        print(
            f"{nama:<18} | {hasil_ocr[:25]:<25} | "
            f"{benar}/{total:<13} | {akurasi:.2f}%"
        )

        nama_file = nama.lower().replace(" ", "_") + ".png"

        cv2.imwrite(
            str(FOLDER_OUTPUT / nama_file),
            citra
        )

print("\nProses selesai.")
print("Hasil gambar dan histogram ada di folder:", FOLDER_OUTPUT)
