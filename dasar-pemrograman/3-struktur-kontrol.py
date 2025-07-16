# LANGKAH 1: Tanyakan kepada pengguna apakah sedang hujan
# Fungsi input() digunakan untuk mengambil jawaban dari pengguna lewat keyboard
# .strip() digunakan untuk menghapus spasi di awal dan akhir, .lower() untuk mengubah huruf jadi kecil semua
raining_input = input("Apakah sedang hujan? (ya/tidak): ").strip().lower()

# Periksa apakah jawaban pengguna adalah "ya"
# Jika iya, maka variabel raining bernilai True, artinya sedang hujan
# Jika tidak, maka raining bernilai False
raining = (raining_input == "ya")

# LANGKAH 2: Jika TIDAK hujan (raining == False), langsung cetak "Keluar rumah"
if not raining:
    print("Keluar rumah.")
else:
    # LANGKAH 3: Jika sedang hujan, tanyakan apakah pengguna membawa payung
    umbrella_input = input("Apakah kamu membawa payung? (ya/tidak): ").strip().lower()

    # Jika jawabannya "ya", maka umbrella akan bernilai True (artinya membawa payung)
    umbrella = (umbrella_input == "ya")

    # LANGKAH 4: Jika pengguna membawa payung, maka bisa keluar rumah meskipun sedang hujan
    if umbrella:
        print("Keluar rumah.")
    else:
        # LANGKAH 5: Jika pengguna tidak membawa payung, tunggu sampai hujan berhenti
        # Gunakan perulangan while True agar terus berjalan sampai kondisi hujan berhenti (break)
        while True:
            # Tampilkan pesan bahwa program sedang menunggu
            print("Menunggu sebentar...")

            # Tanyakan lagi kepada pengguna apakah hujan masih turun
            still_raining_input = input("Apakah masih hujan? (ya/tidak): ").strip().lower()
            still_raining = (still_raining_input == "ya")  # True jika masih hujan, False jika tidak

            # Jika hujan sudah berhenti (still_raining == False), maka keluar dari loop
            if not still_raining:
                print("Keluar rumah.")
                break  # Menghentikan perulangan
