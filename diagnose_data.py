"""
diagnose_data.py
Memeriksa apakah data tiap huruf benar-benar berisi bentuk tangan yang konsisten.

Ide utama: untuk setiap sampel dihitung "ekstensi jari" =
    jarak ujung jari ke pergelangan / jarak pangkal jari ke pergelangan
(rata-rata 4 jari: telunjuk, tengah, manis, kelingking).
  - Tangan MENGEPAL  -> nilainya kecil (sekitar 1 atau kurang)
  - Tangan TERBUKA   -> nilainya besar (sekitar 2)
Kalau huruf A (kepalan) ternyata punya banyak baris bernilai besar,
berarti ada pose terbuka yang ikut terekam sebagai A.

Jalankan: python diagnose_data.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from talky_common import FEATURE_COLUMNS, FOLDER

CSV_PATH = os.path.join(FOLDER, "data", "dataset.csv")
UJUNG = [8, 12, 16, 20]   # ujung telunjuk, tengah, manis, kelingking
PANGKAL = [5, 9, 13, 17]  # pangkal keempat jari itu

if not os.path.exists(CSV_PATH):
    print("File data/dataset.csv belum ada.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
pts = df[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(-1, 21, 3)

# Pergelangan sudah (0,0,0) setelah normalisasi, jadi jarak = norma titik
rasio = np.linalg.norm(pts[:, UJUNG], axis=2) / np.linalg.norm(pts[:, PANGKAL], axis=2)
df["ekstensi"] = rasio.mean(axis=1)

labels = sorted(df["label"].unique())
print(f"Baris: {len(df)} | Huruf: {', '.join(labels)}\n")

# ---------- 1. Sebaran ekstensi jari per huruf ----------
print("=" * 60)
print("1. EKSTENSI JARI PER HURUF  (kecil = mengepal, besar = terbuka)")
print("=" * 60)
bins = np.arange(0.4, 3.01, 0.2)
for lab in labels:
    e = df.loc[df["label"] == lab, "ekstensi"].to_numpy()
    print(f"\nHuruf {lab}  (n={len(e)})  "
          f"p5={np.percentile(e, 5):.2f}  median={np.median(e):.2f}  "
          f"p95={np.percentile(e, 95):.2f}")
    hist, _ = np.histogram(e, bins=bins)
    for i, h in enumerate(hist):
        if h == 0:
            continue
        print(f"   {bins[i]:.1f}-{bins[i + 1]:.1f} | {'#' * max(1, int(40 * h / len(e)))} {h}")

# ---------- 2. Pose yang tercampur dalam satu huruf ----------
print("\n" + "=" * 60)
print("2. BARIS YANG MENYIMPANG DARI POLA HURUFNYA")
print("=" * 60)
total_menyimpang = 0
for lab in labels:
    sub = df[df["label"] == lab]
    X = sub[FEATURE_COLUMNS].to_numpy(dtype=float)
    pusat = np.median(X, axis=0)
    jarak = np.linalg.norm(X - pusat, axis=1)
    batas = np.median(jarak) * 3
    idx = np.where(jarak > batas)[0]
    total_menyimpang += len(idx)
    baris_csv = [int(sub.index[i]) + 2 for i in idx[:8]]  # +2 = header + mulai dari 1
    ket = f"contoh baris CSV: {baris_csv}" if len(idx) else ""
    print(f"Huruf {lab}: {len(idx)} dari {len(sub)} baris menyimpang "
          f"({len(idx) / len(sub):.0%})  {ket}")

# ---------- 3. Apakah antar huruf bisa dibedakan? ----------
print("\n" + "=" * 60)
print("3. KETERPISAHAN ANTAR HURUF  (uji silang 5-lipatan)")
print("=" * 60)
if len(labels) < 2:
    print("Perlu minimal 2 huruf untuk uji ini.")
else:
    n_min = df["label"].value_counts().min()
    if n_min < 5:
        print("Ada huruf dengan sampel < 5, uji dilewati.")
    else:
        Xa = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        ya = df["label"].to_numpy()
        knn = KNeighborsClassifier(n_neighbors=5)
        pred = cross_val_predict(knn, Xa, ya,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0))
        print(f"Akurasi keseluruhan: {(pred == ya).mean():.1%}")
        print("(angka ini optimistis: data dari sesi yang sama saling mirip)\n")
        print("Tabel silang (baris = huruf asli, kolom = tebakan):")
        print(pd.crosstab(pd.Series(ya, name="asli"), pd.Series(pred, name="tebakan")).to_string())

# ---------- Panduan ----------
print("\n" + "=" * 60)
print("PANDUAN MEMBACA")
print("=" * 60)
print("""- Bagian 1: satu huruf sebaiknya punya SATU gundukan angka. Kepalan (A)
  harus berkumpul di nilai kecil. Kalau ada gundukan kedua di nilai besar,
  berarti ada pose terbuka yang ikut terekam sebagai huruf itu.
- Bagian 2: persentase menyimpang yang tinggi (di atas ~15%) menandakan pose
  tercampur atau tangan banyak tidak stabil saat merekam.
- Bagian 3: akurasi tinggi dan tabel silang bersih berarti antar huruf
  memang terpisah di data. Ini TIDAK membuktikan pose-nya benar sesuai SIBI,
  hanya bahwa tiap huruf konsisten dan berbeda dari huruf lain.
- Batas angka di atas adalah perkiraan, bukan aturan pasti. Yang terpenting
  adalah pola sebarannya dan apakah cocok dengan bentuk tangan yang Anda rekam.""")
