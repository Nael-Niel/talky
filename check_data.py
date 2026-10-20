"""
check_data.py
Memeriksa apakah data/dataset.csv hasil rekaman sudah benar.
Jalankan: python check_data.py
"""
import os

import numpy as np
import pandas as pd

from talky_common import FEATURE_COLUMNS, FOLDER

CSV_PATH = os.path.join(FOLDER, "data", "dataset.csv")
TARGET_MIN = 200  # batas minimal sampel per huruf yang dianggap cukup


def status(ok):
    return "OK   " if ok else "CEK! "


if not os.path.exists(CSV_PATH):
    print("File data/dataset.csv belum ada. Jalankan 01_collect_data.py dulu.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
print(f"File     : {CSV_PATH}")
print(f"Baris    : {len(df)}")
print(f"Kolom    : {df.shape[1]} (seharusnya 66)\n")

# 1. Struktur kolom
kolom_ok = list(df.columns) == ["person_id", "label", "handedness"] + FEATURE_COLUMNS
print(f"[{status(kolom_ok)}] Nama dan urutan kolom sesuai")

# 2. Nilai kosong / tidak valid
fitur = df[FEATURE_COLUMNS].to_numpy(dtype=float)
ada_nan = bool(np.isnan(fitur).any() or np.isinf(fitur).any())
print(f"[{status(not ada_nan)}] Tidak ada nilai kosong (NaN/Inf)")

# 3. Normalisasi: pergelangan (titik 0) harus (0,0,0)
pergelangan = fitur[:, :3]
nol_ok = bool(np.allclose(pergelangan, 0, atol=1e-5))
print(f"[{status(nol_ok)}] Titik pergelangan = (0,0,0) pada semua baris")

# 4. Normalisasi: jarak terjauh dari pergelangan harus 1
pts = fitur.reshape(-1, 21, 3)
maks = np.linalg.norm(pts, axis=2).max(axis=1)
skala_ok = bool(np.allclose(maks, 1.0, atol=1e-3))
print(f"[{status(skala_ok)}] Skala ter-normalisasi (jarak maks = 1), "
      f"rentang {maks.min():.3f} - {maks.max():.3f}")

# 5. Baris yang identik persis (tangan tidak bergerak sama sekali)
duplikat = int(df.duplicated(subset=FEATURE_COLUMNS).sum())
print(f"[{status(duplikat < 0.05 * len(df))}] Baris identik persis: {duplikat} "
      f"({duplikat / len(df):.1%}) - makin kecil makin baik")

# 6. Jumlah per huruf
print("\nJumlah sampel per huruf:")
jumlah = df["label"].value_counts().sort_index()
for huruf, n in jumlah.items():
    bar = "#" * int(n / 10)
    tanda = "" if n >= TARGET_MIN else "  <- kurang"
    print(f"  {huruf}: {n:4d} {bar}{tanda}")

# 7. Per orang
print("\nJumlah sampel per orang:")
print(df["person_id"].value_counts().to_string())

# 8. Huruf yang belum direkam
semua = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
belum = [h for h in semua if h not in set(df["label"])]
print(f"\nHuruf belum direkam: {', '.join(belum) if belum else '(tidak ada)'}")
print("Catatan: J dan Z ditunda dulu karena berupa gerakan.")
