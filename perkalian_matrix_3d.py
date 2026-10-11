
import numpy as np


def input_array_3d(nama):
    print(f"\nInput Array 3D {nama}")

    batch = int(input("Masukkan jumlah matriks (batch): "))
    baris = int(input("Masukkan jumlah baris setiap matriks: "))
    kolom = int(input("Masukkan jumlah kolom setiap matriks: "))

    array = np.zeros((batch, baris, kolom), dtype=float)

    for b in range(batch):
        print(f"\nMatriks ke-{b + 1}")

        for i in range(baris):
            while True:
                try:
                    data = input(
                        f"Baris {i + 1} (masukkan {kolom} angka dipisahkan spasi): "
                    ).split()

                    if len(data) != kolom:
                        print(f"Harus memasukkan tepat {kolom} angka.")
                        continue

                    array[b, i] = [float(x) for x in data]
                    break

                except ValueError:
                    print("Input tidak valid. Masukkan angka yang benar.")

    return array


def perkalian_matriks_3d(array_a, array_b):
    if array_a.shape[0] != array_b.shape[0]:
        raise ValueError(
            "Jumlah matriks (batch) kedua array harus sama."
        )

    if array_a.shape[2] != array_b.shape[1]:
        raise ValueError(
            "Jumlah kolom Array A harus sama dengan "
            "jumlah baris Array B."
        )

    return np.matmul(array_a, array_b)


def main():
    print("======================================")
    print(" PERKALIAN MATRIKS ARRAY 3 DIMENSI")
    print("======================================")

    while True:
        array_a = input_array_3d("A")
        array_b = input_array_3d("B")

        try:
            hasil = perkalian_matriks_3d(array_a, array_b)

            print("\nArray A:")
            print(array_a)

            print("\nArray B:")
            print(array_b)

            print("\nHasil Perkalian Matriks A x B:")
            print(hasil)

            print("\nBentuk hasil (shape):", hasil.shape)
            break

        except ValueError as error:
            print("\nTerjadi kesalahan:", error)
            print("Silakan masukkan kembali kedua array.")


if __name__ == "__main__":
    main()
