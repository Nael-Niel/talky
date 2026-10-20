"""
quick_test.py
Tes cepat (tanpa training MLP): apakah bentuk tangan Anda saat ini dikenali
sebagai huruf yang sudah direkam di data/dataset.csv?

Cara kerja: membandingkan tangan live dengan data rekaman memakai KNN
(mencari sampel rekaman yang paling mirip).

Cara pakai: python quick_test.py     (tekan ESC atau Q untuk keluar)
"""
import os
import time
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from talky_common import (FEATURE_COLUMNS, FOLDER, create_landmarker, detect,
                          draw_hand, normalize)

CSV_PATH = os.path.join(FOLDER, "data", "dataset.csv")
MIN_CONF = 0.6      # di bawah ini ditampilkan "?" (tidak yakin)
WINDOW = 7          # jumlah frame untuk penghalusan
MIN_VOTES = 4       # minimal suara sama agar hasil dianggap stabil
FAKTOR_BATAS = 1.5  # makin besar = makin longgar menerima bentuk tangan

if not os.path.exists(CSV_PATH):
    print("File data/dataset.csv belum ada. Rekam dulu dengan 01_collect_data.py.")
    raise SystemExit

df = pd.read_csv(CSV_PATH)
X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
y = df["label"].to_numpy()
huruf_ada = sorted(set(y))
print(f"Data dimuat: {len(df)} sampel, huruf: {', '.join(huruf_ada)}")

knn = KNeighborsClassifier(n_neighbors=min(5, len(df)), weights="distance")
knn.fit(X, y)

# Batas 'tidak dikenal': tangan yang jauh dari pusat data sebuah huruf ditolak,
# supaya bentuk tangan lain tidak dipaksa menjadi A atau B.
pusat = {c: X[y == c].mean(axis=0) for c in huruf_ada}
batas = {}
for c in huruf_ada:
    jarak_c = np.linalg.norm(X[y == c] - pusat[c], axis=1)
    batas[c] = float(np.percentile(jarak_c, 95)) * FAKTOR_BATAS
    print(f"  Huruf {c}: batas jarak = {batas[c]:.2f}")


def put(frame, text, pos, color=(255, 255, 255), scale=0.6, tebal=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), tebal + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, tebal)


landmarker = create_landmarker(num_hands=1)
cap = cv2.VideoCapture(0)  # ganti ke 1 jika kamera tidak terbaca
t0 = time.time()
riwayat = deque(maxlen=WINDOW)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kamera tidak terbaca. Coba ganti VideoCapture(0) menjadi VideoCapture(1).")
            break

        frame = cv2.flip(frame, 1)  # sama seperti saat merekam
        pts, _ = detect(landmarker, frame, int((time.time() - t0) * 1000))

        put(frame, f"Huruf yang ada di data: {' '.join(huruf_ada)}", (10, 24), scale=0.5)

        if pts is None:
            riwayat.clear()
            put(frame, "Tidak ada tangan", (10, 60), (0, 0, 255), 0.8)
        else:
            draw_hand(frame, pts)
            fitur = normalize(pts)
            if fitur is not None:
                proba = knn.predict_proba([fitur])[0]
                idx = int(proba.argmax())
                huruf, conf = knn.classes_[idx], float(proba[idx])

                jarak = float(np.linalg.norm(fitur - pusat[huruf]))
                dikenal = jarak <= batas[huruf]

                riwayat.append(huruf if (conf >= MIN_CONF and dikenal) else "?")
                stabil, suara = Counter(riwayat).most_common(1)[0]
                tampil = stabil if suara >= MIN_VOTES else "..."

                if tampil == "?":
                    put(frame, "Tidak dikenal", (10, 70), (0, 165, 255), 1.2, 3)
                elif tampil == "...":
                    put(frame, "...", (10, 70), (0, 255, 255), 1.4, 3)
                else:
                    put(frame, f"Terbaca: {tampil}", (10, 70), (0, 255, 0), 1.4, 3)
                put(frame, f"Frame ini: {huruf} ({conf:.0%})  jarak {jarak:.2f}/{batas[huruf]:.2f}",
                    (10, 105), scale=0.55)

        cv2.imshow("Talky - Tes Cepat", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()