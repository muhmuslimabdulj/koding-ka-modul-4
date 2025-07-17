# ===========================
# 1. Import Library
# ===========================
import warnings
warnings.filterwarnings('ignore')  # Menyembunyikan warning dari library agar output bersih

import numpy as np                      # Untuk operasi numerik dan array
import pandas as pd                     # Untuk manipulasi dan analisis data berbasis tabel
import matplotlib.pyplot as plt         # Untuk visualisasi grafik sederhana
import seaborn as sns                   # Untuk visualisasi grafik yang interaktif dan menarik
from sklearn.datasets import load_wine # Dataset Wine bawaan dari Scikit-learn
from sklearn.cluster import KMeans      # Algoritma unsupervised learning: KMeans
from sklearn.preprocessing import StandardScaler  # Untuk normalisasi data
from collections import Counter         # Untuk menghitung label yang paling sering muncul (mayoritas)

# ===========================
# 2. Memuat Dataset Wine
# ===========================
wine = load_wine()  # Memuat dataset Wine ke dalam variabel

# Kita hanya ambil dua fitur: alcohol dan malic_acid, agar bisa divisualisasi 2D
features = ["alcohol", "malic_acid"]
X = pd.DataFrame(wine.data, columns=wine.feature_names)[features]

# Tampilkan 5 baris pertama untuk melihat struktur datanya
print("=== 5 Baris Pertama Data Wine (2 fitur) ===")
print(X.head(), "\n")

# ===========================
# 3. EDA Sebelum Preprocessing
# ===========================
print("=== Statistik Deskriptif Sebelum Preprocessing ===")
print(X.describe())  # Statistik dasar: mean, std, min, max, dll.

# Visualisasi distribusi masing-masing fitur
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(X["alcohol"], bins=15, kde=True)
plt.title("Distribusi Alcohol")

plt.subplot(1, 2, 2)
sns.histplot(X["malic_acid"], bins=15, kde=True)
plt.title("Distribusi Malic Acid")
plt.tight_layout()
plt.show()

# Visualisasi outlier dengan boxplot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=X["alcohol"])
plt.title("Boxplot Alcohol")

plt.subplot(1, 2, 2)
sns.boxplot(y=X["malic_acid"])
plt.title("Boxplot Malic Acid")
plt.tight_layout()
plt.show()

# ===========================
# 4. Preprocessing Data
# ===========================

# Fungsi untuk menghapus outlier menggunakan metode IQR (Interquartile Range)
def remove_outliers(df, features):
    df_clean = df.copy()
    for f in features:
        Q1 = df_clean[f].quantile(0.25)  # Kuartil pertama (25%)
        Q3 = df_clean[f].quantile(0.75)  # Kuartil ketiga (75%)
        IQR = Q3 - Q1                    # Rentang antar kuartil
        lower = Q1 - 1.5 * IQR           # Batas bawah
        upper = Q3 + 1.5 * IQR           # Batas atas
        # Hapus data yang berada di luar batas bawah dan atas
        df_clean = df_clean[(df_clean[f] >= lower) & (df_clean[f] <= upper)]
    return df_clean

# Terapkan penghapusan outlier
X_clean = remove_outliers(X, features)
print(f"Jumlah data setelah hapus outlier: {len(X_clean)} dari {len(X)}")

# Normalisasi fitur agar skala nilainya setara (penting untuk clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)  # Output: array NumPy yang sudah ternormalisasi

# ===========================
# 5. EDA Setelah Preprocessing
# ===========================
print("=== Statistik Setelah Penghapusan Outlier ===")
print(X_clean.describe())  # Cek distribusi data setelah dibersihkan

# Visualisasi scatter plot setelah outlier dihapus
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_clean["alcohol"], y=X_clean["malic_acid"])
plt.title("Scatter Plot Setelah Outlier Dihapus")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.show()

# ===========================
# 6. Klasterisasi dengan KMeans
# ===========================
# Membuat objek KMeans dan menetapkan jumlah klaster = 4
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)  # Melakukan klasterisasi dan menyimpan hasilnya

# Tambahkan kolom klaster ke DataFrame agar bisa dianalisis lebih lanjut
X_clean = X_clean.copy()
X_clean["cluster"] = clusters

# ===========================
# 7. Analisis Klaster
# ===========================
# Gunakan label asli (ground truth) dari dataset Wine untuk mengevaluasi klaster (tidak digunakan dalam pelatihan)
y_true = wine.target
label_names = ["Cultivar 0", "Cultivar 1", "Cultivar 2"]  # Nama asli kelas

cluster_names = {}  # Dictionary untuk menyimpan nama tiap klaster
for c in range(4):  # Karena kita menggunakan 4 klaster
    idx = X_clean.index[X_clean["cluster"] == c]  # Ambil index data yang termasuk klaster c
    if len(idx) > 0:
        labels = y_true[idx]  # Ambil label asli berdasarkan indeks
        most_common, count = Counter(labels).most_common(1)[0]  # Cari label paling sering
        cluster_names[c] = f"Mirip {label_names[most_common]}"
    else:
        cluster_names[c] = "Kosong"  # Jika klaster kosong (jarang terjadi)

# ===========================
# 8. Visualisasi Klaster
# ===========================
# Ambil pusat klaster dari model KMeans dalam skala normalisasi
centers_scaled = kmeans.cluster_centers_

# Kembalikan pusat klaster ke skala asli agar mudah dipahami
centers = scaler.inverse_transform(centers_scaled)

# Visualisasi scatter plot klaster + pusat klaster
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_clean["alcohol"], y=X_clean["malic_acid"],
    hue=X_clean["cluster"], palette="viridis", s=100
)
# Tambahkan titik pusat klaster dengan simbol X
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Pusat Klaster')
plt.title("Hasil Klasterisasi Wine (Alcohol vs Malic Acid)")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.legend(title="Klaster")
plt.show()

# Tampilkan informasi pusat klaster
print("Pusat Klaster (skala asli):")
print(centers)

# Tampilkan label nama klaster
print("\nPenamaan Klaster berdasarkan label mayoritas:")
for c in range(4):
    print(f"Klaster {c}: {cluster_names[c]}")

# ===========================
# 9. Prediksi Interaktif
# ===========================
# Pengguna bisa memasukkan data (dua fitur) dan program akan memprediksi klasternya
while True:
    try:
        print("\nMasukkan fitur baru untuk prediksi:")
        alcohol = float(input("Alcohol: "))
        malic_acid = float(input("Malic Acid: "))

        # Gabungkan input menjadi array dan normalisasi
        user_input = np.array([[alcohol, malic_acid]])
        user_scaled = scaler.transform(user_input)

        # Prediksi menggunakan model KMeans
        pred = kmeans.predict(user_scaled)[0]
        name = cluster_names.get(pred, "Tidak Dikenal")
        print(f"Data termasuk dalam klaster {pred} -> {name}")
    except ValueError:
        print("Input tidak valid. Masukkan angka.")

    ulang = input("Ingin mencoba lagi? (y/n): ")
    if ulang.lower() != 'y':
        print("Program selesai.")
        break
