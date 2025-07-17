# ======================================
# 1. Import Library yang Diperlukan
# ======================================

import pandas as pd              # Digunakan untuk manipulasi data dalam bentuk tabel (DataFrame)
import numpy as np               # Digunakan untuk operasi numerik (misal: menghitung korelasi)
import matplotlib.pyplot as plt  # Untuk membuat visualisasi grafik dasar
import seaborn as sns            # Library visualisasi statistik berbasis Matplotlib (lebih modern)

# Atur style visualisasi agar lebih konsisten dan menarik
sns.set(style='whitegrid')

# ======================================
# 2. Memuat Dataset Titanic dari Seaborn
# ======================================

# Dataset Titanic sudah disediakan secara langsung oleh library Seaborn
# Dataset ini berisi informasi penumpang kapal Titanic (umur, jenis kelamin, kelas, status selamat, dll.)
titanic = sns.load_dataset('titanic')

# Menampilkan 5 baris pertama untuk mengenal struktur dan isi dataset
print("5 Baris Pertama Dataset Titanic:")
print(titanic.head())

# ======================================
# 3. Informasi Umum dan Statistik Dataset
# ======================================

print("\nInformasi Dataset:")
# Menampilkan tipe data tiap kolom, jumlah non-null, dan total baris
print(titanic.info())

print("\nStatistik Deskriptif:")
# Menampilkan ringkasan statistik fitur numerik: count, mean, std, min, max, dll.
print(titanic.describe())

# ======================================
# 4. Analisis Nilai Kosong (Missing Values)
# ======================================

# Menghitung jumlah nilai kosong (missing) di setiap kolom
print("\nJumlah Missing Values per Kolom:")
print(titanic.isnull().sum())

# Visualisasi nilai kosong dengan heatmap
# Tujuannya untuk melihat bagian data mana yang tidak lengkap
plt.figure(figsize=(10, 6))
sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
plt.title("Heatmap Missing Values pada Dataset Titanic")
plt.show()

# ======================================
# 5. Analisis Fitur Kategorikal
# ======================================

# Distribusi penumpang berdasarkan jenis kelamin
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', data=titanic, palette='pastel')
plt.title("Distribusi Penumpang Berdasarkan Jenis Kelamin")
plt.xlabel("Jenis Kelamin")
plt.ylabel("Jumlah Penumpang")
plt.show()

# Distribusi status kelangsungan hidup (0 = tidak selamat, 1 = selamat)
plt.figure(figsize=(8, 6))
sns.countplot(x='survived', data=titanic, palette='Set2')
plt.title("Distribusi Status Kelangsungan Hidup Penumpang")
plt.xlabel("Status (0 = Tidak Selamat, 1 = Selamat)")
plt.ylabel("Jumlah Penumpang")
plt.show()

# ======================================
# 6. Analisis Fitur Numerik
# ======================================

# Histogram: distribusi umur penumpang
plt.figure(figsize=(8, 6))
sns.histplot(titanic['age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title("Distribusi Umur Penumpang")
plt.xlabel("Umur")
plt.ylabel("Frekuensi")
plt.show()

# Boxplot: melihat persebaran umur di setiap kelas penumpang
plt.figure(figsize=(8, 6))
sns.boxplot(x='pclass', y='age', data=titanic, palette='Set3')
plt.title("Distribusi Umur Berdasarkan Kelas Penumpang")
plt.xlabel("Kelas Penumpang")
plt.ylabel("Umur")
plt.show()

# ======================================
# 7. Analisis Korelasi antar Variabel Numerik
# ======================================

# Memilih hanya kolom numerik untuk analisis korelasi
numeric_data = titanic.select_dtypes(include=[np.number])

# Menghitung korelasi antar fitur numerik
correlation_matrix = numeric_data.corr()

# Menampilkan matriks korelasi dalam bentuk teks
print("\nMatriks Korelasi:")
print(correlation_matrix)

# Visualisasi matriks korelasi dengan heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap Matriks Korelasi pada Dataset Titanic")
plt.show()

# ======================================
# 8. Visualisasi Hubungan Antar Variabel
# ======================================

# Scatter plot: hubungan antara umur dan tarif, diberi warna berdasarkan status selamat
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='fare', data=titanic, hue='survived', palette='Set1', alpha=0.7)
plt.title("Scatter Plot: Umur vs. Tarif dengan Status Kelangsungan Hidup")
plt.xlabel("Umur")
plt.ylabel("Tarif")
plt.legend(title="Survived", loc="upper right")
plt.show()

# ======================================
# 9. Visualisasi Menyeluruh dengan Pairplot
# ======================================

# Pairplot: gabungan scatter plot dan histogram antar fitur
# Menunjukkan pola hubungan antar variabel numerik seperti age, fare, dan pclass
sns.pairplot(titanic, hue='survived', palette='Set2', vars=['age', 'fare', 'pclass'])
plt.suptitle("Pairplot Variabel pada Dataset Titanic", y=1.02)
plt.show()

# ======================================
# 10. Kesimpulan Sementara dari EDA Titanic
# ======================================

# - Kolom seperti 'age', 'deck', dan 'embark_town' memiliki banyak missing values.
# - Penumpang laki-laki lebih banyak, tetapi tingkat keselamatan perempuan lebih tinggi.
# - Terdapat variasi besar dalam tarif dan umur penumpang; outlier terdeteksi pada beberapa kolom.
# - Terdapat korelasi negatif antara 'fare' dan 'pclass' (semakin tinggi kelas, tarif lebih murah).
# - Scatter dan pairplot memperlihatkan perbedaan pola antara penumpang yang selamat dan tidak selamat.
