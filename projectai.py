from sklearn.tree import DecisionTreeClassifier

# Data latih: [suhu, aktivitas, sudah_minum, durasi_terakhir_minum]
# Label: 0=Sudah cukup, 1=Tunggu sebentar, 2=Minum sekarang!
data = [
    [22, 0, 1, 10],  # Suhu dingin, rebahan, sudah minum, baru saja => cukup
    [30, 2, 1, 60],  # Panas, olahraga, sudah minum tapi lama => minum
    [25, 1, 0, 90],  # Normal, kuliah, belum minum, lama => minum
    [28, 1, 1, 15],  # Agak panas, kuliah, baru minum => tunggu
    [21, 0, 0, 30],  # Dingin, rebahan, belum minum => tunggu
    [29, 2, 0, 120], # Panas, olahraga, belum minum lama => minum
    [24, 0, 1, 5],   # Normal, rebahan, baru minum => cukup
    [26, 1, 1, 35],  # Sedikit lama => tunggu
]

labels = [0, 2, 2, 1, 1, 2, 0, 1]  # Target: kebutuhan minum

# Model
model = DecisionTreeClassifier()
model.fit(data, labels)

# aktivitas
aktivitas_mapping = {
    "rebahan": 0,
    "kuliah": 1,
    "olahraga": 2
}

# Label output
output_mapping = {
    0: "Sudah cukup",
    1: "Minum sebentar lagi",
    2: "Minum sekarang!"
}

# Input dari user
print("=== Prediksi Kebutuhan Minum Air ===")
suhu = int(input("Berapa suhu ruangan sekarang? (°C): "))
aktivitas_str = input("Sedang melakukan apa? (rebahan/kuliah/olahraga): ").lower()
aktivitas = aktivitas_mapping.get(aktivitas_str, 1)  # default: kuliah
sudah_minum = int(input("Sudah minum hari ini? (0=Belum, 1=Sudah): "))
durasi = int(input("Berapa menit lalu kamu terakhir minum?: "))

# Prediksi
input_data = [[suhu, aktivitas, sudah_minum, durasi]]
prediksi = model.predict(input_data)[0]

print(f"\nSaran: {output_mapping[prediksi]}")