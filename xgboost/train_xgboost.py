import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI PATH
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "Dataset XGBoost", "Dataset XGBoost Kesuburan Tanah.csv")

# ============================================================
# 1. PEMUATAN DATASET
# ============================================================
print("[1] Membaca dataset dari ThingSpeak Export...")
print(f"    Path: {dataset_path}")
df_raw = pd.read_csv(dataset_path)
print(f"    Total baris mentah : {len(df_raw)}")

# Konversi timestamp ke WIB (UTC+7)
df_raw['waktu_wib'] = pd.to_datetime(df_raw['created_at'], utc=True).dt.tz_convert('Asia/Jakarta')

# ============================================================
# 2. DEFINISI SESI PENGAMBILAN DATA
#
#    Eksperimen dilakukan pada 2026-07-23 (Waktu WIB).
#    Durasi tiap sesi: 15 menit.
#    Frekuensi pengiriman: 1 data / 20 detik => ~3 data/menit => ~45 data/sesi.
#
#    Jenis sesi:
#      KALIBRASI  = hanya untuk kalibrasi alat, tidak masuk training
#      SAMPEL     = data tanah, masuk ke training
#      KEDUANYA   = media kalibrasi sekaligus masuk training (double-purpose)
#
#    Catatan khusus:
#      - Media Tanam Campuran (coco peat, sekam bakar, top soil, humus teh,
#        pupuk kompos) bersifat KEDUANYA: dipakai kalibrasi DAN masuk dataset.
#      - Tanah Kering: nilai 0 pada unsur hara (N=0) adalah data VALID yang
#        merepresentasikan ketiadaan unsur hara, bukan error sensor.
#        Hanya baris yang SEMUA 7 field sekaligus = 0 yang dianggap error alat.
# ============================================================
# Format: (jam_mulai, menit_mulai, jam_selesai, menit_selesai, nama_sesi, jenis)
SESI = [
    (11, 46, 12,  1, "Air (Kalibrasi)",                          "KALIBRASI"),
    (13,  3, 13, 18, "Tanah Merah Bawah Pohon",                  "SAMPEL"),
    (13, 22, 13, 37, "Tanah Merah + AB Mix 50ml",                "SAMPEL"),
    (13, 43, 13, 58, "Tanah Merah + AB Mix 100ml",               "SAMPEL"),
    (14,  1, 14, 16, "Tanah Merah + AB Mix 200ml",               "SAMPEL"),
    (14, 24, 14, 39, "Tanah Sampah Organik",                     "SAMPEL"),
    (14, 44, 14, 59, "Air AB Mix (Kalibrasi)",                   "KALIBRASI"),
    (16, 32, 16, 47, "Tanah Kering",                             "SAMPEL"),
    (16, 51, 17,  6, "Media Tanam Campuran",                     "KEDUANYA"),   # Kalibrasi + Sampel
    (17, 13, 17, 28, "Tanah Pot + Kompos",                       "SAMPEL"),
    (17, 31, 17, 46, "Tanah Pot + Kompos + AB Mix 50ml",         "SAMPEL"),
    (17, 52, 18,  7, "Tanah Pot + Kompos + AB Mix 100ml",        "SAMPEL"),
    (18, 13, 18, 28, "Tanah Pot + Kompos + AB Mix 200ml",        "SAMPEL"),
]

# ============================================================
# 3. FILTER DATA BERDASARKAN TIMESTAMP SESI
# ============================================================
print("\n[2] Memetakan data ke sesi pengambilan sampel...")

TANGGAL_EKSPERIMEN = '2026-07-23'
df_raw['tanggal_wib'] = df_raw['waktu_wib'].dt.strftime('%Y-%m-%d')
df23 = df_raw[df_raw['tanggal_wib'] == TANGGAL_EKSPERIMEN].copy()
df23['total_menit'] = df23['waktu_wib'].dt.hour * 60 + df23['waktu_wib'].dt.minute
print(f"    Data pada {TANGGAL_EKSPERIMEN}: {len(df23)} baris")

# Assign label sesi ke tiap baris
df23['Jenis_Sampel'] = None
df23['Tipe_Sesi']    = None

for (sh, sm, eh, em, nama, tipe) in SESI:
    start_min = sh * 60 + sm
    end_min   = eh * 60 + em
    mask = (df23['total_menit'] >= start_min) & (df23['total_menit'] <= end_min)
    df23.loc[mask, 'Jenis_Sampel'] = nama
    df23.loc[mask, 'Tipe_Sesi']    = tipe

# Buang baris yang tidak masuk window sesi manapun
df23 = df23[df23['Jenis_Sampel'].notna()].copy()
print(f"    Baris dalam window sesi : {len(df23)}")

# Pisahkan kalibrasi murni vs. data sampel (termasuk KEDUANYA)
df_kalibrasi = df23[df23['Tipe_Sesi'] == 'KALIBRASI']
df_sampel    = df23[df23['Tipe_Sesi'].isin(['SAMPEL', 'KEDUANYA'])].copy()
print(f"    Sesi KALIBRASI (dibuang)        : {len(df_kalibrasi)} baris [{df_kalibrasi['Jenis_Sampel'].unique().tolist()}]")
print(f"    Sesi SAMPEL + KEDUANYA (dipakai): {len(df_sampel)} baris")

# ============================================================
# 4. RENAME KOLOM SENSOR
# ============================================================
rename_map = {
    'field1': 'Kelembaban',    # Soil Moisture (%)
    'field2': 'Suhu',          # Soil Temperature (degrees C)
    'field3': 'Konduktivitas', # Electrical Conductivity / EC (uS/cm)
    'field4': 'pH',            # Soil pH
    'field5': 'Nitrogen',      # N content (mg/kg)
    'field6': 'Fosfor',        # P content (mg/kg)
    'field7': 'Kalium',        # K content (mg/kg)
}
df_sampel.rename(columns=rename_map, inplace=True)

FITUR_SENSOR = ['Kelembaban', 'Suhu', 'Konduktivitas', 'pH', 'Nitrogen', 'Fosfor', 'Kalium']

# ============================================================
# 5. PEMBERSIHAN DATA
#
#    Aturan pembersihan:
#      1. Hapus baris yang ada nilai NaN (data tidak terkirim)
#      2. Hapus baris yang SEMUA 7 field sekaligus bernilai 0
#         (ini error sensor/koneksi saat inisialisasi, bukan data nyata)
#      PENTING: Baris yang hanya SEBAGIAN field = 0 (misal N=0 pada
#               tanah kering) adalah data VALID dan tetap disimpan.
# ============================================================
print("\n[3] Membersihkan data...")

# Hapus NaN
sebelum = len(df_sampel)
df_sampel.dropna(subset=FITUR_SENSOR, inplace=True)
print(f"    Baris hapus (NaN)                    : {sebelum - len(df_sampel)}")

# Hapus baris yang SEMUA 7 sensor sekaligus = 0 (error alat total)
sebelum = len(df_sampel)
mask_semua_nol = (df_sampel[FITUR_SENSOR] == 0).all(axis=1)
n_semua_nol = mask_semua_nol.sum()
df_sampel = df_sampel[~mask_semua_nol].copy()
print(f"    Baris hapus (semua field = 0, error)  : {n_semua_nol}")
print(f"    Total data bersih                     : {len(df_sampel)} baris")
print(f"    Catatan: Nilai 0 parsial (misal N=0 pada tanah kering) TETAP disimpan")

# ============================================================
# 6. RINGKASAN DATA PER SESI (VERIFIKASI ~45 DATA/SESI)
# ============================================================
print("\n    Verifikasi jumlah data per sesi (target: ~45 data/sesi):")
print("    [15 menit x 3 data/menit (delay 20 detik) = 45 titik data]")
print()
summary = df_sampel.groupby('Jenis_Sampel', sort=False).agg(
    Jml=('Nitrogen', 'count'),
    N_med=('Nitrogen', 'median'),
    P_med=('Fosfor',   'median'),
    K_med=('Kalium',   'median'),
    pH_med=('pH',      'median'),
    Kelembaban_med=('Kelembaban', 'median'),
).round(2)
print(summary.to_string())

# ============================================================
# 7. PELABELAN KESUBURAN (RULE-BASED SCORING)
#
#    Referensi: Balai Penelitian Tanah, Kementan RI
#
#    Skor per parameter (0 = Rendah, 1 = Sedang, 2 = Tinggi/Ideal):
#      Nitrogen  : <2 mg/kg  -> 0  |  2-5 mg/kg  -> 1  |  >5 mg/kg  -> 2
#      Fosfor    : <10 mg/kg -> 0  |  10-20 mg/kg -> 1  |  >20 mg/kg -> 2
#      Kalium    : <10 mg/kg -> 0  |  10-20 mg/kg -> 1  |  >20 mg/kg -> 2
#      pH        : <5.5/>7.5 -> 0  |  5.5-6.0/7.0-7.5 -> 1  |  6.0-7.0 -> 2
#
#    Total skor (0-8) -> Kelas Output:
#      0 = Tidak Subur  (skor 0-2)   <- termasuk tanah kering (N=0)
#      1 = Kurang Subur (skor 3-5)
#      2 = Subur        (skor 6-8)
# ============================================================
print("\n[4] Membuat label kesuburan tanah...")

def skor_N(n):
    if n < 2:    return 0
    elif n <= 5: return 1
    else:        return 2

def skor_P(p):
    if p < 10:    return 0
    elif p <= 20: return 1
    else:         return 2

def skor_K(k):
    if k < 10:    return 0
    elif k <= 20: return 1
    else:         return 2

def skor_pH(ph):
    if ph < 5.5 or ph > 7.5:
        return 0
    elif (5.5 <= ph < 6.0) or (7.0 < ph <= 7.5):
        return 1
    else:
        return 2

def label_kesuburan(skor):
    if skor <= 2:   return 0  # Tidak Subur
    elif skor <= 5: return 1  # Kurang Subur
    else:           return 2  # Subur

df_sampel['skor'] = (
    df_sampel['Nitrogen'].apply(skor_N) +
    df_sampel['Fosfor'].apply(skor_P) +
    df_sampel['Kalium'].apply(skor_K) +
    df_sampel['pH'].apply(skor_pH)
)
df_sampel['Output'] = df_sampel['skor'].apply(label_kesuburan)
df_sampel.drop(columns=['skor'], inplace=True)

label_map = {0: 'Tidak Subur', 1: 'Kurang Subur', 2: 'Subur'}
dist = df_sampel['Output'].value_counts().sort_index()
print("    Distribusi kelas:")
for k, v in dist.items():
    print(f"      {k} ({label_map[k]:12s}) : {v} sampel")

print("\n    Label tiap jenis sampel:")
sesi_label = df_sampel.groupby('Jenis_Sampel')['Output'].agg(lambda x: label_map[round(x.median())])
for nama, lbl in sesi_label.items():
    tipe = next(t for (sh,sm,eh,em,n,t) in SESI if n == nama)
    tag = " [KALIBRASI+SAMPEL]" if tipe == 'KEDUANYA' else ""
    print(f"      {nama:<45} -> {lbl}{tag}")

# Simpan dataset berlabel
labeled_path = os.path.join(script_dir, "Dataset XGBoost", "Dataset_Berlabel.csv")
cols_simpan  = ['waktu_wib', 'Jenis_Sampel', 'Tipe_Sesi'] + FITUR_SENSOR + ['Output']
df_sampel[cols_simpan].to_csv(labeled_path, index=False)
print(f"\n    Dataset berlabel disimpan ke: {labeled_path}")

# ============================================================
# 8. PEMISAHAN FITUR DAN TARGET
# ============================================================
X = df_sampel[FITUR_SENSOR].copy()
y = df_sampel['Output'].copy()

print(f"\n[5] Fitur training : {FITUR_SENSOR}")
print(f"    Total sampel   : {len(X)}")

# ============================================================
# 9. PEMBAGIAN DATA LATIH DAN DATA UJI (80:20)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[6] Split 80:20 -> {len(X_train)} latih | {len(X_test)} uji")

# ============================================================
# 10. SMOTE - Menyeimbangkan Kelas Data Latih
# ============================================================
print("\n[7] Menerapkan SMOTE pada data latih...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

dist_smote = pd.Series(y_train_res).value_counts().sort_index()
print("    Distribusi setelah SMOTE:")
for k, v in dist_smote.items():
    print(f"      {k} ({label_map[k]:12s}) : {v} sampel")

# ============================================================
# 11. PELATIHAN MODEL XGBOOST CLASSIFIER
# ============================================================
print("\n[8] Melatih model XGBoost Classifier...")
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=150,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False,
)
model.fit(X_train_res, y_train_res)
print("    Pelatihan selesai!")

# ============================================================
# 12. EVALUASI MODEL
# ============================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n[9] Akurasi pada data uji  : {accuracy * 100:.2f}%")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"    Cross-Val 5-fold       : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

class_names = ['Tidak Subur (0)', 'Kurang Subur (1)', 'Subur (2)']
print("\n    Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================================
# 13. VISUALISASI CONFUSION MATRIX
# ============================================================
print("[10] Menyimpan grafik Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix - Klasifikasi Kesuburan Tanah (XGBoost)', fontsize=14)
plt.ylabel('Kenyataan (Actual)', fontsize=12)
plt.xlabel('Prediksi (Predicted)', fontsize=12)
plt.tight_layout()
cm_path = os.path.join(script_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"     Disimpan ke: {cm_path}")

# ============================================================
# 14. VISUALISASI FEATURE IMPORTANCE
# ============================================================
print("\n[11] Menyimpan grafik Feature Importance...")
importances = pd.Series(model.feature_importances_, index=FITUR_SENSOR).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
colors = sns.color_palette('viridis', len(importances))
bars = plt.barh(importances.index[::-1], importances.values[::-1], color=colors[::-1])
plt.title('Parameter Sensor Paling Berpengaruh\ndalam Penentuan Kesuburan Tanah (XGBoost)', fontsize=13)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Parameter Sensor', fontsize=12)
for bar, val in zip(bars, importances.values[::-1]):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', ha='left', fontsize=9)
plt.tight_layout()
fi_path = os.path.join(script_dir, 'feature_importance.png')
plt.savefig(fi_path, dpi=300)
plt.close()
print(f"     Disimpan ke: {fi_path}")

# ============================================================
# 15. VISUALISASI DISTRIBUSI KELAS PER SESI
# ============================================================
print("\n[12] Menyimpan grafik distribusi per sesi...")
fig, ax = plt.subplots(figsize=(14, 6))
sesi_dist = df_sampel.groupby('Jenis_Sampel')['Output'].value_counts().unstack(fill_value=0)
sesi_dist.columns = [label_map.get(c, str(c)) for c in sesi_dist.columns]
for col in ['Tidak Subur', 'Kurang Subur', 'Subur']:
    if col not in sesi_dist.columns:
        sesi_dist[col] = 0
sesi_dist = sesi_dist[['Tidak Subur', 'Kurang Subur', 'Subur']]
sesi_dist.plot(kind='bar', ax=ax, color=['#e74c3c', '#f39c12', '#27ae60'],
               edgecolor='black', linewidth=0.5)
ax.set_title('Distribusi Kelas Kesuburan per Jenis Sampel', fontsize=13)
ax.set_xlabel('Jenis Sampel', fontsize=11)
ax.set_ylabel('Jumlah Data', fontsize=11)
ax.legend(title='Kelas Kesuburan')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.tight_layout()
dist_path = os.path.join(script_dir, 'distribusi_sampel.png')
plt.savefig(dist_path, dpi=300)
plt.close()
print(f"     Disimpan ke: {dist_path}")

# ============================================================
# 16. PENYIMPANAN MODEL
# ============================================================
model_path = os.path.join(script_dir, 'xgb_soil_model.json')
model.save_model(model_path)
print(f"\n[13] Model tersimpan ke: {model_path}")
print("\n=== SELESAI ===")
