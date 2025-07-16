import pandas as pd  # Mengimpor library pandas dan menyingkatnya menjadi pd

# Membuat pandas Series
data = [10, 20, 30, 40]  # Data nilai
index = ["a", "b", "c", "d"]  # Label indeks

seri = pd.Series(data, index=index)  # Membuat Series dengan data dan indeks khusus

# Menampilkan Series
print("Pandas Series:")
print(seri)
