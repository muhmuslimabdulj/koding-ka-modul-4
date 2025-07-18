# ================================
# Program Klasifikasi Suhu Sederhana
# Sesuai dengan Lembar Kerja 4.1
# ================================

# --------------------------------
# LANGKAH 1: Membuat Fungsi
# --------------------------------

# Fungsi ini menerima nilai suhu sebagai input (dalam derajat Celsius)
# dan mengembalikan kategori suhu: "Panas", "Sejuk", atau "Dingin"
def klasifikasi_suhu(suhu):
    # Jika suhu lebih dari 30, maka tergolong "Panas"
    if suhu > 30:
        return "Panas"
    # Jika suhu antara 20 sampai 30 (inklusif), maka "Sejuk"
    elif suhu >= 20:
        return "Sejuk"
    # Jika suhu di bawah 20, maka "Dingin"
    else:
        return "Dingin"

# --------------------------------
# LANGKAH 2: Input dari Pengguna
# --------------------------------

# Minta pengguna memasukkan suhu
# Gunakan int() untuk mengubah input string menjadi angka bulat
# (Diasumsikan pengguna selalu memasukkan angka yang valid)
suhu = int(input("Masukkan suhu dalam derajat Celsius: "))

# --------------------------------
# LANGKAH 3: Proses dan Output
# --------------------------------

# Panggil fungsi klasifikasi_suhu() dengan input dari pengguna
kategori = klasifikasi_suhu(suhu)

# Cetak hasil kategori ke layar
print(f"Kategori suhu: {kategori}")
