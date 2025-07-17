# Import library yang dibutuhkan
import pandas as pd  # Untuk mengolah data dalam bentuk tabel (DataFrame)
from sklearn.model_selection import train_test_split  # Untuk membagi data menjadi data latih dan data uji

# ===============================
# 1. Muat dataset contoh (Iris)
# ===============================
from sklearn.datasets import load_iris  # Fungsi untuk memuat dataset Iris
iris = load_iris()  # Memuat data Iris ke dalam variabel iris

# Membuat DataFrame dari data iris
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)  # Membuat tabel dengan nama kolom dari fitur Iris

# Menambahkan kolom 'target' yang berisi label (jenis bunga iris)
df['target'] = iris.target  # Kolom ini adalah hasil yang ingin diprediksi

# ===============================
# 2. Pisahkan fitur dan target
# ===============================
# Fitur (X) adalah data input (panjang dan lebar sepal & petal)
X = df.drop('target', axis=1)  # Menghapus kolom target dari DataFrame untuk mendapatkan fitur

# Target (y) adalah label yang akan diprediksi
y = df['target']  # Menyimpan kolom target sebagai label

# ===============================
# 3. Bagi data latih dan data uji
# ===============================
# Membagi data menjadi 80% data latih dan 20% data uji
# - test_size=0.2 artinya 20% data dipakai untuk pengujian
# - random_state=42 agar hasil pembagian selalu sama tiap dijalankan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 4. Tampilkan ukuran data latih dan data uji
# ===============================
print("Ukuran data latih:", X_train.shape)  # Menampilkan jumlah data latih (baris, kolom)
print("Ukuran data uji:", X_test.shape)    # Menampilkan jumlah data uji (baris, kolom)
