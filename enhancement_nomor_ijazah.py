
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Baca gambar ijazah
gambar = cv2.imread("contoh_ijazah/foto 1.jpg")

if gambar is None:
    raise FileNotFoundError("Gambar ijazah tidak ditemukan.")

# 2. Crop area nomor ijazah
# Sesuaikan koordinat dengan posisi nomor pada gambar
x1, y1, x2, y2 = 300, 100, 600, 180
crop = gambar[y1:y2, x1:x2]

if crop.size == 0:
    raise ValueError("Area crop kosong. Periksa koordinat.")

# 3. Konversi ke grayscale
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# 4. Brightness adjustment
brightness = cv2.convertScaleAbs(gray, alpha=1.0, beta=30)

# 5. Contrast stretching
p1, p99 = np.percentile(gray, [1, 99])
if p99 > p1:
    contrast = np.clip(
        (gray.astype(np.float32) - p1) * 255 / (p99 - p1),
        0, 255
    ).astype(np.uint8)
else:
    contrast = gray.copy()

# 6. Histogram equalization
equalized = cv2.equalizeHist(gray)

# 7. Tampilkan citra sebelum dan sesudah
nama = [
    "Grayscale Asli",
    "Brightness Adjustment",
    "Contrast Stretching",
    "Histogram Equalization"
]

hasil = [gray, brightness, contrast, equalized]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

for i in range(4):
    axes[0, i].imshow(hasil[i], cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(nama[i])
    axes[0, i].axis("off")

    histogram = cv2.calcHist(
        [hasil[i]], [0], None, [256], [0, 256]
    )
    axes[1, i].plot(histogram, color="black")
    axes[1, i].set_title("Histogram")
    axes[1, i].set_xlabel("Intensitas Piksel")
    axes[1, i].set_ylabel("Jumlah Piksel")
    axes[1, i].set_xlim(0, 255)

plt.tight_layout()
plt.show()

# 8. Simpan semua hasil
cv2.imwrite("crop_nomor_ijazah.png", gray)
cv2.imwrite("brightness.png", brightness)
cv2.imwrite("contrast_stretching.png", contrast)
cv2.imwrite("histogram_equalization.png", equalized)

print("Proses enhancement selesai.")
