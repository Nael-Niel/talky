"""
check_data.py
Memeriksa apakah file CSV dataset hasil rekaman atau ekstraksi sudah benar.

Jalankan (dari folder proyek):
  python check_data.py
  python check_data.py data\\dataset_luar_sibi1.csv
  python check_data.py data\\dataset.csv --min 200
"""
import argparse
import os

import numpy as np
import pandas as pd

from talky_common import FEATURE_COLUMNS, FOLDER

DEFAULT_CSV = os.path.join(FOLDER, "data", "dataset.csv")
TARGET_PER_ORANG = 100  # selaras dengan TARGET di 01_collect_data.py


def status(ok):
    return "OK   " if ok else "CEK! "


def main():
    parser = argparse.ArgumentParser(description="Validasi teknis CSV dataset landmark")
    parser.add_argument("file", nargs="?", default=DEFAULT_CSV,
                        help="Path file CSV (bawaan: data/dataset.csv)")
    parser.add_argument("--min", type=int, default=None,
                        help="Batas minimal sampel per huruf (bawaan: 100 x jumlah orang)")
    args = parser.parse_args()

    csv_path = args.file
    if not os.path.exists(csv_path):
        print(f"File tidak ditemukan: {csv_path}")
        if os.path.abspath(csv_path) == os.path.abspath(DEFAULT_CSV):
            print("Jalankan 01_collect_data.py dulu untuk merekam data.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path)
    print(f"File     : {csv_path}")
    print(f"Baris    : {len(df)}")
    print(f"Kolom    : {df.shape[1]} (seharusnya 66)\n")

    # 1. Struktur kolom
    kolom_ok = list(df.columns) == ["person_id", "label", "handedness"] + FEATURE_COLUMNS
    print(f"[{status(kolom_ok)}] Nama dan urutan kolom sesuai")
    if not kolom_ok or len(df) == 0:
        if len(df) == 0:
            print("[CEK! ] File CSV kosong.")
        raise SystemExit(1)

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

    # 6. Distribusi handedness
    handed = df["handedness"].value_counts().to_dict()
    print(f"[INFO ] Handedness: {handed}")

    # 7. Jumlah per huruf
    n_orang = max(1, df["person_id"].nunique())
    target_min = args.min if args.min is not None else TARGET_PER_ORANG * n_orang
    print(f"\nJumlah sampel per huruf (batas minimal = {target_min}):")
    jumlah = df["label"].value_counts().sort_index()
    for huruf, n in jumlah.items():
        bar = "#" * int(n / 10)
        tanda = "" if n >= target_min else "  <- kurang"
        print(f"  {huruf}: {n:4d} {bar}{tanda}")

    # 8. Per orang
    print("\nJumlah sampel per orang:")
    tabel_orang = pd.crosstab(df["label"], df["person_id"])
    for orang, total in df["person_id"].value_counts().items():
        kol = tabel_orang[orang]
        kosong = [h for h, v in kol.items() if v == 0]
        catatan = f" | belum ada: {', '.join(kosong)}" if kosong else ""
        print(f"  {orang}: {total} baris (min {int(kol.min())}, maks {int(kol.max())} per huruf){catatan}")

    # 9. Huruf yang belum ada
    semua = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    belum = [h for h in semua if h not in set(df["label"])]
    print(f"\nHuruf belum ada: {', '.join(belum) if belum else '(tidak ada)'}")
    print("Catatan: J dan Z ditangani gerak_jz.py karena berupa gerakan.")


if __name__ == "__main__":
    main()
