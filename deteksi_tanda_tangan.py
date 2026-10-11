import cv2
import numpy as np
from pathlib import Path
import csv

INPUT_FOLDER = Path("images")
OUTPUT_FOLDER = Path("output")

PRESENT_FOLDER = INPUT_FOLDER / "present"
ABSENT_FOLDER = INPUT_FOLDER / "absent"

ROI_PRESENT = (434, 1724, 952, 334)

ROI_ABSENT = (158, 630, 348, 122)

FOREGROUND_THRESHOLD = 0.10

OUTPUT_FOLDER.mkdir(exist_ok=True)


def get_images(folder):
    extensions = ["*.jpg", "*.jpeg", "*.png"]

    files = []

    for extension in extensions:
        files.extend(folder.glob(extension))

    return sorted(files)


def proses_gambar(image_path, actual_class):

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Gagal membaca {image_path.name}")
        return None


    height, width = img.shape[:2]

    if width == 3506 and height == 2481:
        x, y, w, h = ROI_PRESENT

    elif width == 1280 and height == 906:
        x, y, w, h = ROI_ABSENT

    else:
        print(f"Ukuran gambar tidak dikenali: {width} x {height}")
        return None

    crop = img[y:y+h, x:x+w]


    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)



    _, global_threshold = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )


    otsu_value, otsu_threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        otsu_threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )


    foreground_pixels = cv2.countNonZero(closing)

    total_pixels = closing.size

    foreground_ratio = (
        foreground_pixels / total_pixels
    )


    if foreground_ratio < FOREGROUND_THRESHOLD:
        
        prediction = "SIGNATURE PRESENT"
    else:
        prediction = "SIGNATURE ABSENT"

    # Kelas sebenarnya
    if actual_class == "PRESENT":
        actual_label = "SIGNATURE PRESENT"
    else:
        actual_label = "SIGNATURE ABSENT"

    # Cek hasil
    if prediction == actual_label:
        status = "CORRECT"
    else:
        status = "WRONG"


    print()
    print("======================================")
    print(f"File              : {image_path.name}")
    print(f"Actual            : {actual_label}")
    print(f"Otsu threshold    : {otsu_value:.0f}")
    print(f"Foreground pixels : {foreground_pixels}")
    print(f"Foreground ratio  : {foreground_ratio:.4f}")
    print(f"Prediction        : {prediction}")
    print(f"Status            : {status}")
    print("======================================")


    class_folder = OUTPUT_FOLDER / actual_class.lower()

    class_folder.mkdir(exist_ok=True)

    nama = image_path.stem

    cv2.imwrite(
        str(class_folder / f"{nama}_crop.jpg"),
        crop
    )

    cv2.imwrite(
        str(class_folder / f"{nama}_grayscale.jpg"),
        gray
    )

    cv2.imwrite(
        str(class_folder / f"{nama}_global.jpg"),
        global_threshold
    )

    cv2.imwrite(
        str(class_folder / f"{nama}_otsu.jpg"),
        otsu_threshold
    )

    cv2.imwrite(
        str(class_folder / f"{nama}_opening.jpg"),
        opening
    )

    cv2.imwrite(
        str(class_folder / f"{nama}_closing.jpg"),
        closing
    )

    return {
        "file": image_path.name,
        "actual": actual_label,
        "foreground_pixels": foreground_pixels,
        "foreground_ratio": foreground_ratio,
        "prediction": prediction,
        "status": status
    }


present_images = get_images(PRESENT_FOLDER)
absent_images = get_images(ABSENT_FOLDER)

print()
print("======================================")
print("DETEKSI TANDA TANGAN")
print("======================================")

print(f"Data PRESENT : {len(present_images)}")
print(f"Data ABSENT  : {len(absent_images)}")

print()
print("ROI otomatis digunakan.")
print("Tidak perlu memilih ROI secara manual.")

results = []


for image_path in present_images:

    result = proses_gambar(
        image_path,
        "PRESENT"
    )

    if result is not None:
        results.append(result)


for image_path in absent_images:

    result = proses_gambar(
        image_path,
        "ABSENT"
    )

    if result is not None:
        results.append(result)


total_data = len(results)

correct_data = sum(
    1
    for result in results
    if result["status"] == "CORRECT"
)

wrong_data = total_data - correct_data


if total_data > 0:

    accuracy = (
        correct_data / total_data
    ) * 100

else:

    accuracy = 0


# ======================================
# SIMPAN CSV
# ======================================

csv_path = OUTPUT_FOLDER / "hasil_deteksi.csv"


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "File",
        "Actual",
        "Foreground Pixels",
        "Foreground Ratio",
        "Prediction",
        "Status"
    ])

    for result in results:

        writer.writerow([
            result["file"],
            result["actual"],
            result["foreground_pixels"],
            f"{result['foreground_ratio']:.4f}",
            result["prediction"],
            result["status"]
        ])


# ======================================
# HASIL AKHIR
# ======================================

print()
print()
print("======================================")
print("HASIL AKHIR")
print("======================================")

print(f"Total data : {total_data}")
print(f"Data benar : {correct_data}")
print(f"Data salah : {wrong_data}")
print(f"Akurasi    : {accuracy:.2f}%")

print()
print("Hasil disimpan di:")
print(csv_path)

print()
print("======================================")
print("SELESAI")
print("======================================")