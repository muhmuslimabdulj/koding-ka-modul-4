# ===============================================================
# ANALISIS DATA TGM (Tingkat Kegemaran Membaca) 2020–2023
# ===============================================================
# Tujuan:
# - Mengeksplorasi data survei TGM
# - Membersihkan data (missing values, duplikat, format angka)
# - Menghapus outlier
# - Normalisasi fitur numerik
# - Menyimpan hasil akhir ke CSV

# =======================
# 1. Import Library
# =======================

import pandas as pd                    # Untuk memproses data tabular (DataFrame)
import numpy as np                     # Untuk operasi numerik dan array
import matplotlib.pyplot as plt        # Untuk visualisasi grafik dasar
import seaborn as sns                  # Untuk grafik statistik seperti histogram dan heatmap
from sklearn.preprocessing import MinMaxScaler  # Untuk normalisasi data ke skala 0–1

# Jika kamu pakai Jupyter Notebook, aktifkan baris di bawah:
# %matplotlib inline

# =======================
# 2. Membaca Dataset
# =======================

# Membaca file CSV dengan delimiter ";"
df = pd.read_csv("datasets/TGM 2020-2023_eng.csv", sep=";")  # Pastikan file CSV berada di folder yang sama

# =======================
# 3. Menampilkan Informasi Awal
# =======================

print("=== Detected Columns ===")
print(df.columns.tolist(), "\n")  # Menampilkan nama semua kolom

print("=== First 5 Rows ===")
print(df.head(), "\n")  # Menampilkan 5 baris pertama dari dataset

# =======================
# 4. Eksplorasi Awal Data
# =======================

print("=== DataFrame Info ===")
print(df.info(), "\n")  # Struktur data, tipe data setiap kolom, dan jumlah data non-null

print("=== Descriptive Statistics ===")
print(df.describe(include='all'), "\n")  # Statistik deskriptif dari semua kolom

print("=== Missing Values Per Column ===")
print(df.isnull().sum(), "\n")  # Menampilkan jumlah data kosong per kolom

# =======================
# 5. Cek dan Hapus Duplikat
# =======================

duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)

# Jika ditemukan duplikat, hapus
if duplicates > 0:
    df = df.drop_duplicates()
    print("After removing duplicates, shape:", df.shape, "\n")
else:
    print("No duplicate rows found.\n")

# =======================
# 6. Tangani Missing Values & Format Angka
# =======================

# Kolom-kolom numerik yang masih berupa string dengan koma (misal: "2,5")
reading_col = "Tingkat Kegemaran Membaca (Reading Interest)"
numeric_cols_with_commas = [
    "Reading Frequency per week",
    "Number of Readings per Quarter",
    "Daily Reading Duration (in minutes)",
    "Internet Access Frequency per Week",
    "Daily Internet Duration (in minutes)",
    reading_col
]

# Konversi ke float dan isi missing values dengan median
for col in numeric_cols_with_commas:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.')  # Ganti koma menjadi titik desimal
        df[col] = pd.to_numeric(df[col], errors='coerce')     # Ubah ke angka
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)                  # Isi nilai kosong dengan nilai tengah (median)
        print(f"Missing values in '{col}' filled with median: {median_val}")

# Tangani kolom 'Year' jika ada
if 'Year' in df.columns:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    year_median = df['Year'].median()
    df['Year'] = df['Year'].fillna(year_median)
    print(f"Missing values in 'Year' filled with median: {year_median}")

# Untuk kolom kategori seperti 'Provinsi' dan 'Category', isi nilai kosong dengan mode (yang paling sering muncul)
categorical_cols = ["Provinsi", "Category"]
for cat_col in categorical_cols:
    if cat_col in df.columns:
        mode_val = df[cat_col].mode()[0]
        df[cat_col] = df[cat_col].fillna(mode_val)
        print(f"Missing values in '{cat_col}' filled with mode: {mode_val}")

print()

# =======================
# 7. Visualisasi Awal (Sebelum Hapus Outlier)
# =======================

plt.figure(figsize=(8, 6))
sns.histplot(df[reading_col], bins=20, kde=True)
plt.title(f"Distribusi '{reading_col}' Sebelum Normalisasi")
plt.xlabel(reading_col)
plt.ylabel("Frekuensi")
plt.show()

# =======================
# 8. Hapus Outlier Menggunakan IQR
# =======================

def remove_outliers_iqr(data, cols):
    data_clean = data.copy()
    for c in cols:
        Q1 = data_clean[c].quantile(0.25)
        Q3 = data_clean[c].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = data_clean[(data_clean[c] < lower_bound) | (data_clean[c] > upper_bound)].index
        print(f"Kolom '{c}': {len(outlier_indices)} outlier ditemukan.")
        data_clean = data_clean.drop(index=outlier_indices)
    return data_clean

# Ambil semua kolom numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Kolom numerik:", numeric_cols, "\n")

# Hapus outlier
df_no_outliers = remove_outliers_iqr(df, numeric_cols)

print("\nJumlah data setelah hapus outlier:", df_no_outliers.shape)

# Visualisasi ulang setelah outlier dihapus
plt.figure(figsize=(8, 6))
sns.histplot(df_no_outliers[reading_col], bins=20, kde=True, color='orange')
plt.title(f"Distribusi '{reading_col}' Setelah Outlier Dihapus")
plt.xlabel(reading_col)
plt.ylabel("Frekuensi")
plt.show()

# =======================
# 9. Normalisasi Data (Min-Max Scaling)
# =======================

# Salin dataset hasil outlier removal
df_no_outliers_normalized = df_no_outliers.copy()

# Terapkan normalisasi ke semua kolom numerik
scaler = MinMaxScaler()
numeric_cols_no_outliers = df_no_outliers_normalized.select_dtypes(include=[np.number]).columns.tolist()
df_no_outliers_normalized[numeric_cols_no_outliers] = scaler.fit_transform(
    df_no_outliers_normalized[numeric_cols_no_outliers]
)

# Visualisasi distribusi setelah normalisasi
plt.figure(figsize=(8, 6))
sns.histplot(df_no_outliers_normalized[reading_col], bins=20, kde=True, color='purple')
plt.title(f"Distribusi '{reading_col}' Setelah Normalisasi")
plt.xlabel(f"{reading_col} (Ternormalisasi)")
plt.ylabel("Frekuensi")
plt.show()

# =======================
# 10. Simpan Dataset Hasil
# =======================

df_no_outliers_normalized.to_csv("datasets/TGM_2020-2023_normalized.csv", index=False)
df_no_outliers.to_csv("datasets/TGM_2020-2023_cleaned.csv", index=False)

print("\n✅ Data normalisasi disimpan sebagai: 'TGM_2020-2023_normalized.csv'")
print("✅ Data tanpa outlier disimpan sebagai: 'TGM_2020-2023_cleaned.csv'")
