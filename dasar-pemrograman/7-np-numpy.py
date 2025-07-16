import pandas as pd  # Mengimpor library pandas dengan alias pd

# -------------------- Series --------------------
# Contoh Series: daftar nilai dengan label nama siswa
# pd.Series digunakan untuk membuat data satu dimensi yang memiliki label (index)
nilai = pd.Series([85, 90, 78], index=['Andi', 'Budi', 'Citra'])

# Output:
# Andi     85
# Budi     90
# Citra    78
print(nilai)

# ------------------- DataFrame -------------------
# Contoh DataFrame: tabel yang berisi dua kolom, yaitu "Nama" dan "Nilai"
# pd.DataFrame digunakan untuk membuat data dua dimensi seperti tabel Excel
data = pd.DataFrame({
    'Nama': ['Andi', 'Budi', 'Citra'],
    'Nilai': [85, 90, 78]
})

# Output:
#     Nama  Nilai
# 0   Andi     85
# 1   Budi     90
# 2  Citra     78
print(data)
