import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Waktu Pengisian Baterai", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Prediksi Waktu Pengisian Baterai</h1>", unsafe_allow_html=True)

# CSS untuk mempercantik tampilan
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stSlider, .stNumberInput {
            margin-top: 10px;
        }
        .prediction-box {
            background-color: #4CAF50;
            padding: 1.5rem;
            border-radius: 35px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Simulasi dataset untuk training model
np.random.seed(42)
data = {
    "charger_power": np.random.choice([5, 10, 15, 18, 20, 25, 30, 33, 40, 45], 100),
    "battery_capacity": np.random.choice([2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000], 100)
}
noise = np.random.normal(0, 0.1, size=100)
data["charging_time"] = (np.array(data["battery_capacity"]) / (np.array(data["charger_power"]) * 200)) + noise
df = pd.DataFrame(data)

# Training model regresi linier
X = df[["charger_power", "battery_capacity"]]
y = df["charging_time"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)

# Form input pengguna
st.markdown("###Masukkan Data Pengisian")

col1, col2 = st.columns(2)
with col1:
    charger = st.slider("Daya Charger (Watt)", 5, 45, 25, step=1)
with col2:
    battery = st.slider("Kapasitas Baterai (mAh)", 2000, 7000, 5000, step=100)

# Prediksi waktu pengisian
input_data = pd.DataFrame({
    "charger_power": [charger],
    "battery_capacity": [battery]
})
predicted_time = model.predict(input_data)[0]
predicted_time = max(predicted_time, 0)

# Konversi ke jam dan menit
hours = int(predicted_time)
minutes = int(round((predicted_time - hours) * 60))
time_formatted = f"{hours} jam {minutes} menit"

# Hasil prediksi
st.markdown("### Hasil Prediksi")
st.markdown(f"""
    <div class="prediction-box">
        <h2>Estimasi Waktu Pengisian</h2>
        <p style='font-size: 18px;'>Dengan charger <strong>{charger}W</strong> dan baterai <strong>{battery}mAh</strong>,</p>
        <h3 style='color: ##34542C;'>Waktu pengisian dari 0% - 100% : <strong>{time_formatted}</strong></h3>
    </div>
""", unsafe_allow_html=True)