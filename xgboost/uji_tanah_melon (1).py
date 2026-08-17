import pandas as pd
import numpy as np
import xgboost as xgb
import os

print("="*60)
print("PENGUJI AN MODEL XGBOOST TERHADAP DATASET TANAH MELON (12 TITIK)")
print("="*60)

# 1. Load Model XGBoost yang sudah ditraining
base_dir = r'c:\Users\zahbi\Downloads\khansa'
model_path = os.path.join(base_dir, 'xgb_soil_model.json')

if not os.path.exists(model_path):
    print(f"Error: Model tidak ditemukan di {model_path}")
    exit(1)

model = xgb.XGBClassifier()
model.load_model(model_path)
print(f"[1] Model XGBoost terlatih berhasil dimuat dari:\n    {model_path}\n")

# 2. Load File Excel Data Baru
excel_path = os.path.join(base_dir, 'Dataset XGBoost', 'Tanah Melon 12 Titik Sampel.xlsx')
df = pd.read_excel(excel_path)
print(f"[2] Membaca data baru dari:\n    {excel_path}")
print(f"    Total data mentah: {len(df)} baris")

# Forward fill kolom Jenis Sampel
df['Jenis Sampel'] = df['Jenis Sampel'].ffill().str.replace('\n', ' ').str.strip()

# Mapping kolom ke standar fitur training
rename_map = {
    df.columns[3]: 'Kelembaban',
    df.columns[4]: 'Suhu',
    df.columns[5]: 'Konduktivitas',
    df.columns[6]: 'pH',
    df.columns[7]: 'Nitrogen',
    df.columns[8]: 'Fosfor',
    df.columns[9]: 'Kalium'
}
df.rename(columns=rename_map, inplace=True)

# Hapus baris NaN sensor (baris kosong di akhir sheet)
df.dropna(subset=['Kelembaban', 'pH', 'Nitrogen', 'Fosfor', 'Kalium'], inplace=True)
print(f"    Total data sensor valid (setelah dropna): {len(df)} baris\n")

# 3. Prediksi Menggunakan Model XGBoost
FITUR_SENSOR = ['Kelembaban', 'Suhu', 'Konduktivitas', 'pH', 'Nitrogen', 'Fosfor', 'Kalium']
X_new = df[FITUR_SENSOR].copy()

y_pred_code = model.predict(X_new)
probs = model.predict_proba(X_new)
confidence = np.max(probs, axis=1) * 100

label_map = {0: 'Tidak Subur', 1: 'Kurang Subur', 2: 'Subur'}
df['Hasil_Prediksi_XGBoost'] = [label_map[code] for code in y_pred_code]
df['Tingkat_Keyakinan (%)'] = confidence.round(2)

print("[3] Hasil Prediksi Model XGBoost (Ringkasan Total Data Valid):")
print(df['Hasil_Prediksi_XGBoost'].value_counts().to_string())
print()

# 4. Ringkasan per Titik Sampel (12 Titik)
print("[4] Analisis Ringkasan Hasil Uji per 12 Titik Sampel Melon:")
summary_df = df.groupby('Jenis Sampel', sort=False).agg(
    Jumlah_Data=('Hasil_Prediksi_XGBoost', 'count'),
    pH_Rata2=('pH', 'mean'),
    N_Rata2=('Nitrogen', 'mean'),
    P_Rata2=('Fosfor', 'mean'),
    K_Rata2=('Kalium', 'mean'),
    Kelembaban_Rata2=('Kelembaban', 'mean'),
    EC_Rata2=('Konduktivitas', 'mean'),
    Status_Kesuburan_Dominan=('Hasil_Prediksi_XGBoost', lambda x: x.mode()[0]),
    Rata2_Keyakinan=('Tingkat_Keyakinan (%)', 'mean')
).round(2)

print(summary_df.to_string())

# 5. Export Hasil ke File Excel
output_excel = os.path.join(base_dir, 'Hasil_Prediksi_Tanah_Melon_12Titik.xlsx')

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    summary_df.reset_index().to_excel(writer, sheet_name='Ringkasan 12 Titik', index=False)
    df.to_excel(writer, sheet_name='Data Detail Prediksi', index=False)

print(f"\n[5] Laporan Hasil Pengujian Excel Berhasil Disimpan ke:\n    {output_excel}")
print("\n=== PENGUJI AN SELESAI ===")
