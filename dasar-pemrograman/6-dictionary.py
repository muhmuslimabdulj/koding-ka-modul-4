# Contoh Dictionary
# Dictionary adalah struktur data di Python yang menyimpan data dalam pasangan key dan value.
# Setiap key harus unik. Di bawah ini adalah contoh dictionary bernama "mahasiswa".
mahasiswa = {
    "nama": "Budi",                # key = "nama", value = "Budi"
    "umur": 21,                    # key = "umur", value = 21
    "jurusan": "Teknik Informatika"  # key = "jurusan", value = "Teknik Informatika"
}

# Menampilkan seluruh isi dictionary
print("Mahasiswa:", mahasiswa)

# Mengakses nilai menggunakan key
# Untuk mengambil nilai tertentu dari dictionary, kita gunakan tanda kurung siku [key]
print("Nama:", mahasiswa["nama"])

# Memperbarui nilai dalam dictionary
# Nilai dari key "umur" akan diperbarui dari 21 menjadi 22
mahasiswa["umur"] = 22

# Menampilkan dictionary setelah pembaruan
print("Setelah update umur:", mahasiswa)
