# ============================
# 1. IMPORT LIBRARY
# ============================

# Library ini digunakan untuk memuat dataset iris bawaan dari sklearn
from sklearn.datasets import load_iris

# Library untuk membagi dataset menjadi data latih (training) dan data uji (testing)
from sklearn.model_selection import train_test_split

# Library untuk menstandarisasi fitur agar berada dalam skala yang sama (penting untuk SVM)
from sklearn.preprocessing import StandardScaler

# Library untuk membuat dan melatih model Support Vector Machine
from sklearn.svm import SVC

# Library untuk mengevaluasi model: classification_report dan confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

# Library visualisasi data
import matplotlib.pyplot as plt
import seaborn as sns

# Library manipulasi data berbasis tabel (mirip Excel)
import pandas as pd

# ============================
# 2. MEMUAT DATASET IRIS
# ============================

# Memanggil dataset Iris yang berisi informasi tentang 3 spesies bunga iris
iris = load_iris()

# `iris.data` adalah array 2D yang berisi fitur (panjang sepal, lebar sepal, dll.)
X_full = iris.data

# `iris.target` adalah array 1D yang berisi label klasifikasi: 0 (setosa), 1 (versicolor), 2 (virginica)
y_full = iris.target

# Konversi data ke dalam bentuk DataFrame agar lebih mudah dianalisis dan divisualisasikan
df = pd.DataFrame(X_full, columns=iris.feature_names)
df['target'] = y_full  # Menambahkan kolom label (target) ke dalam DataFrame

# ============================
# 3. EDA (Exploratory Data Analysis)
# ============================

# Melihat 5 baris pertama untuk memahami struktur data
print("=== 5 Baris Pertama Dataset Iris ===")
print(df.head(), "\n")

# Menampilkan informasi dasar: jumlah kolom, tipe data, dan apakah ada nilai kosong
print("=== Info Dataset Iris ===")
print(df.info(), "\n")

# Menampilkan statistik deskriptif: mean, std, min, max, dll.
print("=== Statistik Deskriptif Dataset Iris ===")
print(df.describe(), "\n")

# Menampilkan jumlah missing values di tiap kolom
print("=== Missing Values per Kolom ===")
print(df.isnull().sum(), "\n")

# ============================
# 4. VISUALISASI HUBUNGAN FITUR
# ============================

# Membuat scatterplot antar pasangan fitur, diwarnai berdasarkan label target (0,1,2)
# Tujuannya agar kita bisa melihat apakah ada pemisahan visual antar kelas
sns.pairplot(df, hue='target', vars=iris.feature_names)
plt.suptitle("Pairplot Dataset Iris", y=1.02)
plt.show()

# Membuat heatmap korelasi antar fitur untuk mengetahui seberapa kuat hubungan antar fitur numerik
plt.figure(figsize=(8,6))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap Iris Features")
plt.show()

# ============================
# 5. PEMISAHAN FITUR DAN LABEL + NORMALISASI
# ============================

# Kita ambil ulang X (fitur) dan y (target)
X = iris.data
y = iris.target

# Membagi dataset menjadi data latih (70%) dan data uji (30%)
# random_state memastikan hasil pembagian selalu sama setiap dijalankan
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Normalisasi fitur agar tiap fitur memiliki mean 0 dan standar deviasi 1
# Ini penting karena algoritma SVM sangat sensitif terhadap skala data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Hitung skala dari data training
X_test = scaler.transform(X_test)        # Gunakan skala yang sama untuk data testing

# ============================
# 6. MEMBANGUN DAN MELATIH MODEL SVM
# ============================

# Kita menggunakan model SVM dengan kernel linear
# Kernel linear artinya kita berasumsi data bisa dipisahkan dengan garis lurus (atau bidang)
model = SVC(kernel='linear', C=1.0, random_state=42)

# Melatih model menggunakan data training yang telah dinormalisasi
model.fit(X_train, y_train)

# ============================
# 7. PREDIKSI DAN EVALUASI
# ============================

# Gunakan model untuk memprediksi label dari data testing
y_pred = model.predict(X_test)

# Tampilkan metrik evaluasi: precision, recall, f1-score, dan akurasi
# Ini membantu kita menilai seberapa baik model mengenali tiap kelas
print("=== Laporan Klasifikasi ===")
print(classification_report(y_test, y_pred))

# Tampilkan confusion matrix dalam bentuk heatmap
# Matrix ini menunjukkan berapa banyak data kelas A diprediksi sebagai kelas B
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,    # Label prediksi
            yticklabels=iris.target_names)    # Label aktual
plt.xlabel('Prediksi')
plt.ylabel('Kelas Sebenarnya')
plt.title('Confusion Matrix')
plt.show()
