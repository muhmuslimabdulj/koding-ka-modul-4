# ================================================================
# PROGRAM K-MEANS CLUSTERING: Dataset Wine (2 Fitur Saja)
# Fitur: 'alcohol' dan 'malic_acid'
# ================================================================

# ===========================
# 1. Import Library
# ===========================

import warnings
warnings.filterwarnings('ignore')  # Menonaktifkan peringatan dari library

import numpy as np                   # Untuk manipulasi numerik
import pandas as pd                  # Untuk mengelola data tabular
import matplotlib.pyplot as plt      # Untuk membuat grafik
import seaborn as sns                # Untuk visualisasi statistik yang lebih menarik

from sklearn.datasets import load_wine           # Dataset Wine dari scikit-learn
from sklearn.cluster import KMeans               # Algoritma K-Means Clustering
from sklearn.preprocessing import StandardScaler # Untuk menstandardisasi fitur (mean=0, std=1)
from sklearn.metrics import silhouette_score     # Untuk evaluasi hasil clustering
from collections import Counter                  # Untuk analisis distribusi label

# (Hanya digunakan di Jupyter Notebook)
# %matplotlib inline

# ===========================
# 2. Memuat Dataset Wine
# ===========================

wine = load_wine()  # Memuat dataset wine bawaan sklearn

# Pilih dua fitur: alcohol & malic_acid untuk visualisasi 2D
features = ["alcohol", "malic_acid"]
X = pd.DataFrame(wine.data, columns=wine.feature_names)[features]

# ===========================
# 3. Preprocessing Data
# ===========================
print("=== Pre-Processing Data ===")

# a. Menghapus Outlier Menggunakan IQR
def remove_outliers(df, features):
    df_clean = df.copy()
    for feature in features:
        Q1 = df_clean[feature].quantile(0.25)  # Kuartil bawah
        Q3 = df_clean[feature].quantile(0.75)  # Kuartil atas
        IQR = Q3 - Q1                          # Rentang interkuartil
        lower_bound = Q1 - 1.5 * IQR           # Batas bawah
        upper_bound = Q3 + 1.5 * IQR           # Batas atas
        # Hapus data di luar batas (outlier)
        df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
    return df_clean

# Terapkan fungsi untuk membersihkan data dari outlier
X_clean = remove_outliers(X, features)

# b. Standardisasi Fitur agar skala setara (mean = 0, std = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)  # Hasil dalam bentuk array numpy

# ===========================
# 4. Mencari Jumlah Cluster Optimal
# ===========================
print("=== K-Means Clustering ===")

# --------------------------
# a. Elbow Method
# --------------------------
# Metode ini melihat titik di mana penurunan inertia mulai melambat (elbow)
inertia = []  # Inertia = total jarak kuadrat antar data dan pusat cluster

for k in range(1, 11):  # Uji dari k=1 sampai 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # Simpan inertia untuk tiap k

# Visualisasi Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia (Total Jarak Kuadrat)')
plt.grid(True)
plt.show()

# --------------------------
# b. Silhouette Analysis
# --------------------------
# Silhouette score mengukur seberapa baik data cocok dalam cluster-nya
silhouette_avg = []

for k in range(2, 11):  # Silhouette tidak valid untuk k=1
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_avg.append(score)

# Visualisasi Silhouette Score
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_avg, marker='o', color='orange')
plt.title('Silhouette Score untuk Tiap Nilai k')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# ===========================
# 5. Implementasi K-Means
# ===========================

# Asumsikan dari grafik, k=3 adalah jumlah cluster optimal
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_  # Label cluster hasil prediksi

# Tambahkan label cluster ke DataFrame asli
X_clean['cluster'] = labels

# ===========================
# 6. Visualisasi Hasil Clustering
# ===========================

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='alcohol',
    y='malic_acid',
    hue='cluster',
    data=X_clean,
    palette="viridis",
    s=100
)
plt.title('Visualisasi Hasil Clustering K-Means (3 Cluster)')
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
