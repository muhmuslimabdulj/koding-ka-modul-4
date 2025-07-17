# Import library pandas untuk mengolah data
import pandas as pd

# ==============================================
# --- Membaca file CSV ---
# ==============================================

# Fungsi pd.read_csv() digunakan untuk membaca data dari file CSV (Comma-Separated Values).
# 'data.csv' adalah nama file yang ingin dibaca. Pastikan file tersebut ada di folder kerja Anda.
# Anda bisa mengganti 'data.csv' dengan nama file CSV lain sesuai kebutuhan.
df_csv = pd.read_csv('datasets/data.csv')  # Membaca file CSV dan menyimpannya ke dalam DataFrame bernama df_csv

# Menampilkan 5 baris pertama dari data CSV
print("Data dari file CSV:")
print(df_csv.head())  # df.head() akan menampilkan 5 baris pertama dari data (default-nya)

# ==============================================
# --- Membaca file JSON ---
# ==============================================

# Fungsi pd.read_json() digunakan untuk membaca data dari file JSON (JavaScript Object Notation).
# 'data.json' adalah nama file JSON yang ingin dibaca.
# Anda bisa mengganti 'data.json' dengan nama file JSON lain.
df_json = pd.read_json('datasets/data.json')  # Membaca file JSON dan menyimpannya ke dalam DataFrame bernama df_json

# Menampilkan 5 baris pertama dari data JSON
print("\nData dari file JSON:")
print(df_json.head())  # Menampilkan cuplikan baris pertama dari file JSON yang sudah dibaca

# ==============================================
# Penjelasan singkat fungsi:
# ==============================================

# - pd.read_csv()   : membaca file CSV menjadi DataFrame
# - pd.read_json()  : membaca file JSON menjadi DataFrame
# - df.head()       : menampilkan 5 baris pertama dari DataFrame (berguna untuk melihat isi data secara cepat)
