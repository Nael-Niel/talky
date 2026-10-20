"""
hapus_huruf.py
Menghapus semua baris sebuah huruf (opsional: hanya milik satu person_id)
dari data/dataset.csv. Backup dibuat otomatis sebelum menghapus.

Contoh:
  python hapus_huruf.py A              -> hapus semua data huruf A
  python hapus_huruf.py A nathan       -> hapus data huruf A milik 'nathan' saja
"""
import os
import shutil
import sys
import time

import pandas as pd

from talky_common import FOLDER

CSV_PATH = os.path.join(FOLDER, "data", "dataset.csv")

if len(sys.argv) < 2:
    print(__doc__)
    raise SystemExit

huruf = sys.argv[1].strip().upper()
person = sys.argv[2].strip().lower() if len(sys.argv) > 2 else None

if not os.path.exists(CSV_PATH):
    print("File data/dataset.csv tidak ditemukan.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
mask = df["label"] == huruf
if person:
    mask &= df["person_id"] == person

n_hapus = int(mask.sum())
if n_hapus == 0:
    print("Tidak ada baris yang cocok. Tidak ada yang diubah.")
    raise SystemExit

target = f"huruf {huruf}" + (f" milik '{person}'" if person else " (semua orang)")
print(f"Akan menghapus {n_hapus} baris: {target}")
print(f"Sisa data setelah dihapus: {len(df) - n_hapus} baris")
if input("Ketik YA untuk melanjutkan: ").strip() != "YA":
    print("Dibatalkan. Tidak ada yang diubah.")
    raise SystemExit

backup = os.path.join(
    FOLDER, "data", f"dataset_backup_{time.strftime('%Y%m%d_%H%M%S')}.csv")
shutil.copy2(CSV_PATH, backup)
df[~mask].to_csv(CSV_PATH, index=False)
print(f"Selesai. Backup disimpan di: {backup}")
print("Sekarang Anda bisa merekam ulang huruf itu dengan 01_collect_data.py")
