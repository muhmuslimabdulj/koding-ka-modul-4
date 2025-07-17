# ======================================================
# PROGRAM KLASIFIKASI DENGAN LOGISTIC REGRESSION
# Dataset: Iris (3 Kelas Bunga)
# ======================================================

# =============================
# 1. Import library yang dibutuhkan
# =============================

import pandas as pd                         # Untuk mengelola data dalam bentuk DataFrame (tabel)
import numpy as np                          # Untuk operasi numerik (opsional di sini)
import matplotlib.pyplot as plt             # Untuk membuat grafik sederhana
import seaborn as sns                       # Untuk visualisasi statistik yang lebih baik

# Import dari scikit-learn untuk machine learning
from sklearn.datasets import load_iris                      # Dataset iris bawaan sklearn
from sklearn.model_selection import train_test_split        # Untuk membagi data ke training dan testing
from sklearn.linear_model import LogisticRegression         # Algoritma klasifikasi (logistic regression)
from sklearn.metrics import accuracy_score, confusion_matrix # Untuk evaluasi model

# =============================
# 2. Memuat Dataset Iris
# =============================

# Memuat dataset iris
iris = load_iris()

# Mengubah dataset menjadi DataFrame agar mudah dianalisis
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Menambahkan kolom target (label numerik 0,1,2)
df['target'] = iris.target

# Menambahkan kolom label nama bunga (setosa, versicolor, virginica)
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# Menampilkan 5 baris pertama untuk mengenal data
print("5 Baris Pertama Dataset Iris:")
print(df.head())

# =============================
# 3. Eksplorasi Data
# =============================

# Menampilkan struktur kolom dan tipe data
print("\nInformasi Dataset:")
print(df.info())  # Melihat jumlah data, kolom, dan missing values

# Statistik deskriptif: nilai minimum, maksimum, rata-rata, dll
print("\nStatistik Deskriptif:")
print(df.describe())

# Cek apakah ada nilai kosong
print("\nJumlah Missing Values per Kolom:")
print(df.isnull().sum())

# =============================
# 4. Visualisasi Data
# =============================

# Scatter plot: hubungan antara panjang dan lebar sepal
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', palette='Set1')
plt.title("Scatter Plot Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

# Histogram: distribusi panjang petal
plt.figure(figsize=(8, 6))
sns.histplot(df['petal length (cm)'], bins=20, kde=True)
plt.title("Distribusi Panjang Petal")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frekuensi")
plt.show()

# =============================
# 5. Membagi Data Latih dan Uji
# =============================

# Memisahkan fitur (X) dan label target (y)
X = df[iris.feature_names]  # Fitur: sepal & petal panjang/lebar
y = df['target']            # Target: 0, 1, 2

# Membagi dataset menjadi data training dan data testing
# test_size=0.3 artinya 30% untuk testing, 70% untuk training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Tampilkan jumlah data train dan test
print("\nJumlah Data Training:", X_train.shape[0])
print("Jumlah Data Testing :", X_test.shape[0])

# =============================
# 6. Melatih Model
# =============================

# Inisialisasi model Logistic Regression
# max_iter=200 artinya maksimal iterasi training adalah 200
model = LogisticRegression(max_iter=200)

# Melatih model dengan data training
model.fit(X_train, y_train)

# =============================
# 7. Memprediksi dan Mengevaluasi Model
# =============================

# Memprediksi label untuk data testing
y_pred = model.predict(X_test)

# Menghitung akurasi prediksi model
accuracy = accuracy_score(y_test, y_pred)
print("\nAkurasi Model: {:.2f}%".format(accuracy * 100))

# Membuat confusion matrix untuk melihat performa klasifikasi
cm = confusion_matrix(y_test, y_pred)

# Visualisasi confusion matrix dalam bentuk heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,   # Label prediksi (horizontal)
            yticklabels=iris.target_names)   # Label aktual (vertikal)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.show()
