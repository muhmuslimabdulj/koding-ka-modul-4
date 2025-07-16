# =============================
# CONTOH LIST DI PYTHON
# =============================

# Membuat sebuah list bernama "buah" yang berisi tiga elemen string
# List adalah struktur data yang bisa menyimpan banyak nilai dan bisa diubah (mutable)
buah = ["apel", "pisang", "jeruk"]

# Menampilkan isi list ke layar
print("Buah:", buah)

# Menambahkan elemen baru ke dalam list menggunakan method .append()
# Dalam hal ini, "mangga" akan ditambahkan ke akhir list
buah.append("mangga")
print("Setelah ditambahkan:", buah)

# Menghapus elemen dari list menggunakan method .remove()
# Di sini, "pisang" akan dihapus dari list jika ditemukan
buah.remove("pisang")
print("Setelah dihapus:", buah)

# =============================
# CONTOH TUPLE DI PYTHON
# =============================

# Membuat sebuah tuple bernama "angka" yang berisi empat elemen angka
# Tuple adalah struktur data mirip seperti list, tetapi bersifat tetap (immutable)
angka = (1, 2, 3, 4)

# Menampilkan isi tuple ke layar
print("Angka:", angka)

# Catatan penting:
# Kita TIDAK BISA mengubah isi tuple setelah dibuat.
# Jika kita mencoba mengubah nilai dalam tuple, Python akan menampilkan error.
# Contoh:
# angka[0] = 10  # Ini akan error karena tuple tidak bisa dimodifikasi
