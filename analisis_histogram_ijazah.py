
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Daftar gambar ijazah yang akan dianalisis
daftar_gambar = [
    "contoh_ijazah/foto 1.jpg",
    "contoh_ijazah/foto 2.jpg",
]

folder_output = Path("hasil_analisis")
folder_output.mkdir(parents=True, exist_ok=True)


# 2. Analisis kondisi citra grayscale
def analisis_citra(gray):
    rata_rata = np.mean(gray)
    standar_deviasi = np.std(gray)
    p5, p95 = np.percentile(gray, [5, 95])
    rentang = p95 - p5

    # Membandingkan kecerahan beberapa area citra
    h, w = gray.shape
    ukuran_y = max(1, h // 4)
    ukuran_x = max(1, w // 4)
    rata_area = []

    for y in range(0, h, ukuran_y):
        for x in range(0, w, ukuran_x):
            area = gray[y:min(y + ukuran_y, h),
                        x:min(x + ukuran_x, w)]
            rata_area.append(np.mean(area))

    selisih_pencahayaan = max(rata_area) - min(rata_area)

    # Indikator sederhana noise atau detail tepi
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    indikator_noise = np.mean(cv2.absdiff(gray, blur))

    masalah = []
    rekomendasi = []

    if rata_rata < 85:
        masalah.append("Citra terlalu gelap")
        rekomendasi.append(
            "Gunakan gamma correction dengan gamma < 1 atau CLAHE."
        )
    elif rata_rata > 190:
        masalah.append("Citra terlalu terang")
        rekomendasi.append(
            "Gunakan gamma correction dengan gamma > 1."
        )

    if rentang < 70 or standar_deviasi < 35:
        masalah.append("Kontras rendah")
        rekomendasi.append(
            "Gunakan CLAHE atau histogram equalization."
        )

    if selisih_pencahayaan > 45:
        masalah.append("Kemungkinan pencahayaan tidak merata")
        rekomendasi.append(
            "Gunakan illumination correction atau adaptive thresholding."
        )

    if indikator_noise > 8:
        masalah.append("Kemungkinan noise atau banyak detail tepi")
        rekomendasi.append(
            "Periksa citra; median filter dapat membantu noise impulsif."
        )

    if not masalah:
        masalah.append("Tidak ditemukan masalah utama berdasarkan aturan sederhana")
        rekomendasi.append(
            "Pertahankan citra asli jika teks sudah jelas terbaca."
        )

    return rata_rata, standar_deviasi, rentang, masalah, rekomendasi


# 3. Proses setiap gambar
for nama_file in daftar_gambar:
    gambar = cv2.imread(nama_file)

    if gambar is None:
        print(f"\nGambar tidak ditemukan: {nama_file}")
        continue

    # OpenCV membaca BGR; ubah ke RGB untuk tampilan
    rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)

    # Konversi RGB menjadi grayscale
    gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    rata, std, rentang, masalah, rekomendasi = analisis_citra(gray)

    print("\n" + "=" * 50)
    print("Nama gambar:", nama_file)
    print(f"Rata-rata intensitas: {rata:.2f}")
    print(f"Standar deviasi: {std:.2f}")
    print(f"Rentang intensitas persentil 5%-95%: {rentang:.2f}")

    print("\nAnalisis masalah:")
    for item in masalah:
        print("-", item)

    print("\nRekomendasi enhancement:")
    for item in rekomendasi:
        print("-", item)

    # 4. Tampilkan citra dan histogram
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Citra Asli (RGB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Citra Grayscale")
    axes[0, 1].axis("off")

    # Histogram kanal RGB
    for kanal, warna in enumerate(["b", "g", "r"]):
        histogram = cv2.calcHist(
            [gambar], [kanal], None, [256], [0, 256]
        )
        warna_plot = {"b": "blue", "g": "green", "r": "red"}[warna]
        axes[1, 0].plot(
            histogram, color=warna_plot, label=warna.upper()
        )

    axes[1, 0].set_title("Histogram Citra Asli (RGB)")
    axes[1, 0].set_xlabel("Intensitas Piksel")
    axes[1, 0].set_ylabel("Jumlah Piksel")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].legend()

    # Histogram grayscale
    histogram_gray = cv2.calcHist(
        [gray], [0], None, [256], [0, 256]
    )
    axes[1, 1].plot(histogram_gray, color="black")
    axes[1, 1].set_title("Histogram Grayscale")
    axes[1, 1].set_xlabel("Intensitas Piksel")
    axes[1, 1].set_ylabel("Jumlah Piksel")
    axes[1, 1].set_xlim(0, 255)

    fig.suptitle(Path(nama_file).name)
    fig.tight_layout()

    nama_output = Path(nama_file).stem
    plt.savefig(
        folder_output / f"{nama_output}_histogram.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()

    # Simpan hasil grayscale
    cv2.imwrite(
        str(folder_output / f"{nama_output}_grayscale.png"),
        gray
    )

print("\nSelesai. Periksa folder hasil_analisis.")
