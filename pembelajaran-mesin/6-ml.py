# =========================================================
# Program Linear Regression Menggunakan Dataset Diabetes
# Cocok untuk Pemula yang Belajar Data Science/Machine Learning
# =========================================================

# --- Import Library ---
import numpy as np                    # Untuk operasi numerik (misalnya array)
import pandas as pd                  # Untuk manipulasi data dalam bentuk tabel (DataFrame)
import matplotlib.pyplot as plt      # Untuk visualisasi data (grafik, plot, dll)
import seaborn as sns                # Library visualisasi berbasis Matplotlib (lebih menarik)

# Import library dari Scikit-Learn (sklearn)
from sklearn.datasets import load_diabetes              # Dataset diabetes
from sklearn.model_selection import train_test_split    # Untuk membagi data menjadi train/test
from sklearn.linear_model import LinearRegression       # Algoritma Regresi Linear
from sklearn.preprocessing import MinMaxScaler          # Untuk normalisasi fitur
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Untuk evaluasi model

# --- Tahap 1: Load & Persiapan Data ---
print("=== Tahap Data Preparation ===")

# Memuat dataset diabetes dari Scikit-learn
data = load_diabetes()

# Mengubah data ke dalam bentuk DataFrame agar mudah dianalisis
df = pd.DataFrame(data.data, columns=data.feature_names)

# Menambahkan kolom target (nilai yang ingin diprediksi)
df['target'] = data.target  # target = perkembangan penyakit diabetes

# Menampilkan informasi awal tentang dataset
print("Jumlah sampel:", df.shape[0])                      # Berapa banyak baris (data)
print("Jumlah fitur (termasuk target):", df.shape[1])     # Berapa banyak kolom
print("Jumlah data duplikat:", df.duplicated().sum())     # Apakah ada data yang sama persis
print("Missing values per kolom:")
print(df.isnull().sum(), "\n")                             # Cek apakah ada data kosong

# --- Tahap 2: Preprocessing (Normalisasi) ---
# Normalisasi adalah proses mengubah skala data ke rentang tertentu (misal: 0-1)
scaler = MinMaxScaler()
df['bmi_normalized'] = scaler.fit_transform(df[['bmi']])  # Normalisasi fitur 'bmi'

# Menampilkan perbandingan nilai asli dan nilai hasil normalisasi
print("Contoh normalisasi fitur 'bmi':")
print(df[['bmi', 'bmi_normalized']].head(), "\n")

# --- Tahap 3: Exploratory Data Analysis (EDA) ---
print("=== Tahap Exploratory Data Analysis (EDA) ===")

# Histogram distribusi nilai BMI (belum dinormalisasi)
plt.figure(figsize=(8, 6))
sns.histplot(df['bmi'], bins=20, kde=True, color='skyblue')
plt.title("Distribusi BMI (Sebelum Normalisasi)")
plt.xlabel("BMI")
plt.ylabel("Frekuensi")
plt.show()

# Histogram BMI setelah normalisasi
plt.figure(figsize=(8, 6))
sns.histplot(df['bmi_normalized'], bins=20, kde=True, color='green')
plt.title("Distribusi BMI (Setelah Normalisasi)")
plt.xlabel("BMI (Normalized)")
plt.ylabel("Frekuensi")
plt.show()

# Scatter plot antara BMI dan target untuk melihat pola hubungan
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bmi', y='target', data=df, color='purple', alpha=0.7)
plt.title("Hubungan antara BMI dan Disease Progression")
plt.xlabel("BMI")
plt.ylabel("Disease Progression (Target)")
plt.show()

# Heatmap korelasi antar fitur: untuk melihat fitur mana yang paling berhubungan dengan target
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Korelasi antar fitur")
plt.show()

# Menampilkan ringkasan data
print("=== 5 Baris Pertama Data ===")
print(df.head(), "\n")

print("=== Info Data ===")
print(df.info(), "\n")

print("=== Deskripsi Statistik ===")
print(df.describe(), "\n")

# --- Tahap 4: Pemilihan Fitur dan Target ---
# Kita hanya akan menggunakan 1 fitur (bmi) untuk prediksi
X = df[['bmi']]       # X = fitur
y = df['target']      # y = label / target (yang ingin diprediksi)

# --- Tahap 5: Membagi Data Menjadi Data Latih & Uji ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 70% data akan digunakan untuk melatih model, 30% untuk menguji model

print("Jumlah data training:", len(X_train))
print("Jumlah data testing :", len(X_test))

# --- Tahap 6: Melatih Model Linear Regression ---
model = LinearRegression()
model.fit(X_train, y_train)  # Melatih model dengan data latih

# --- Tahap 7: Evaluasi Model ---
# Melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# Menampilkan parameter model: koefisien dan intercept
print("\nKoefisien (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
print("Persamaan regresi: y = {:.2f} * BMI + {:.2f}".format(model.coef_[0], model.intercept_))

# Menampilkan skor R^2: seberapa baik model menjelaskan variasi data
r2_score = model.score(X_test, y_test)
print("Skor R^2 pada data testing:", r2_score)

# Evaluasi tambahan: menghitung error antara hasil prediksi dan nilai asli
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# --- Tahap 8: Visualisasi Garis Regresi ---
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Data Training')
plt.scatter(X_test, y_test, color='red', label='Data Testing')

# Membuat garis regresi untuk divisualisasikan
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='green', label='Garis Regresi')

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression (Single Feature: BMI) - Diabetes Dataset")
plt.legend()
plt.show()

# --- Tahap 9: Prediksi Interaktif oleh Pengguna ---
# Pengguna bisa memasukkan nilai BMI, dan model akan memprediksi progresi penyakit

while True:
    user_input = input("\nMasukkan nilai BMI (atau ketik 'exit' untuk keluar): ")
    if user_input.lower() == 'exit':
        print("Program selesai. Terima kasih!")
        break

    try:
        user_bmi = float(user_input)
        X_user = np.array([[user_bmi]])
        y_user_pred = model.predict(X_user)
        print(f"Prediksi disease progression untuk BMI = {user_bmi:.2f}: {y_user_pred[0]:.2f}")
    except ValueError:
        print("Input tidak valid. Harap masukkan angka.")
