"""
01_collect_data.py
Merekam landmark tangan (sudah dinormalisasi) per huruf ke data/dataset.csv

Cara pakai:
  - Tekan huruf A-Z di keyboard  -> memilih huruf yang akan direkam
  - Tekan SPASI                  -> mulai / jeda perekaman otomatis
  - Tekan ESC                    -> simpan dan keluar
(Q tidak dipakai untuk keluar karena Q adalah huruf SIBI.)
"""
import csv
import os
import time

import cv2

from talky_common import (FEATURE_COLUMNS, FOLDER, create_landmarker, detect,
                          draw_hand, normalize)

DATA_DIR = os.path.join(FOLDER, "data")
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
HEADER = ["person_id", "label", "handedness"] + FEATURE_COLUMNS

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
TARGET = 400          # target sampel per huruf
INTERVAL = 0.10       # jeda antar sampel (detik) saat perekaman otomatis
FLUSH_EVERY = 20      # tulis ke file setiap sekian sampel

os.makedirs(DATA_DIR, exist_ok=True)

person_id = input("Masukkan person_id (contoh: nathan / sheva): ").strip().lower()
if not person_id:
    person_id = "anon"
print(f"person_id = {person_id}")

# --- Hitung data yang sudah ada (agar bisa dilanjutkan) ---
counts = {c: 0 for c in LETTERS}
file_baru = not os.path.exists(CSV_PATH)
if not file_baru:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # lewati header
        for row in reader:
            if len(row) > 1 and row[1] in counts:
                counts[row[1]] += 1
    print(f"Melanjutkan dataset yang ada: {sum(counts.values())} sampel.")

buffer = []


def flush():
    """Tulis buffer ke CSV (append)."""
    global buffer
    if not buffer:
        return
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(HEADER)
        writer.writerows(buffer)
    buffer = []


def put(frame, text, pos, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


landmarker = create_landmarker(num_hands=1)
cap = cv2.VideoCapture(0)  # ganti ke 1 jika kamera tidak terbaca
t0 = time.time()
last_save = 0.0
active = None
recording = False

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kamera tidak terbaca. Coba ganti VideoCapture(0) menjadi VideoCapture(1).")
            break

        frame = cv2.flip(frame, 1)  # tampilan cermin (harus sama di skrip lain)
        now = time.time()
        pts, handed = detect(landmarker, frame, int((now - t0) * 1000))

        if pts is not None:
            draw_hand(frame, pts)

            if recording and active and now - last_save >= INTERVAL:
                fitur = normalize(pts)
                if fitur is not None:
                    buffer.append([person_id, active, handed] +
                                  [round(float(v), 6) for v in fitur])
                    counts[active] += 1
                    last_save = now
                    if len(buffer) >= FLUSH_EVERY:
                        flush()
                    if counts[active] >= TARGET:
                        recording = False
                        print(f"Huruf {active} mencapai target {TARGET}.")

        # --- Tampilan di layar ---
        if active is None:
            put(frame, "Tekan huruf A-Z untuk memilih", (10, 28), (0, 255, 255))
        else:
            status = "REKAM" if recording else "JEDA"
            warna = (0, 0, 255) if recording else (0, 255, 255)
            put(frame, f"Huruf {active} | {status} | {counts[active]}/{TARGET}",
                (10, 28), warna, 0.8)
        if pts is None:
            put(frame, "Tidak ada tangan", (10, 56), (0, 0, 255))

        h = frame.shape[0]
        put(frame, "Huruf=pilih  SPASI=rekam/jeda  ESC=keluar", (10, h - 80), scale=0.5)
        for i in range(3):  # tiga baris ringkasan jumlah sampel
            potong = LETTERS[i * 9:(i + 1) * 9]
            teks = " ".join(f"{c}:{counts[c]}" for c in potong)
            put(frame, teks, (10, h - 55 + i * 20), (200, 255, 200), 0.5)

        cv2.imshow("Talky - Rekam Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPASI
            if active:
                recording = not recording
        elif 65 <= key <= 90 or 97 <= key <= 122:
            active = chr(key).upper()
            recording = False
finally:
    flush()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\nRingkasan jumlah sampel per huruf:")
    print({c: n for c, n in counts.items() if n > 0})
    print(f"Total: {sum(counts.values())} sampel -> {CSV_PATH}")
