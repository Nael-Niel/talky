"""
01_collect_kata.py
Merekam sekuens landmark tangan dinamis (30 frame) untuk kata sapaan dasar SIBI.
Disimpan sebagai array NumPy (.npy) berukuran (30, 63) di folder data_kata/<LABEL>/

Daftar kata sapaan dasar (Fase 2):
  1: HALO          (melambaikan tangan di samping kepala)
  2: TERIMA_KASIH  (ujung jari menyentuh dagu lalu bergerak maju ke depan)
  3: SAMA_SAMA     (kedua telapak tangan menghadap atas / gerakan maju)
  4: MAAF          (tangan mengepal memutar di depan dada)

Cara pakai:
  - Tekan angka 1-4 di keyboard untuk memilih kata sapaan.
  - Tekan SPASI untuk mulai hitung mundur (countdown 3..2..1) lalu peragakan gerakan selama 30 frame (~1 detik).
  - Tekan ESC untuk keluar.
"""
import os
import sys
import time
import cv2
import numpy as np

from talky_common import (FOLDER, create_landmarker, detect,
                          draw_hand, normalize)

DATA_KATA_DIR = os.path.join(FOLDER, "data_kata")
SEQUENCE_LENGTH = 30  # 30 frame (~1 detik pada webcam 30 FPS)
TARGET_PER_KATA = 30  # target sampel per kata per orang

KATA_LIST = [
    "HALO",
    "TERIMA_KASIH",
    "SAMA_SAMA",
    "MAAF",
]

# Pastikan folder tiap kata tersedia
for kata in KATA_LIST:
    os.makedirs(os.path.join(DATA_KATA_DIR, kata), exist_ok=True)

print("=" * 60)
print("TALKY - Perekaman Sekuens Kata Sapaan Dinamis (1D-CNN)")
print("=" * 60)
person_id = input("Masukkan person_id (contoh: nathan / sheva / hafiid): ").strip().lower()
if not person_id:
    person_id = "anon"
print(f"person_id aktif: {person_id}\n")


def hitung_sampel(kata, p_id):
    """Hitung berapa file rekaman milik person_id untuk kata tertentu."""
    folder = os.path.join(DATA_KATA_DIR, kata)
    if not os.path.exists(folder):
        return 0
    files = [f for f in os.listdir(folder) if f.startswith(f"{p_id}_") and f.endswith(".npy")]
    return len(files)


def simpan_sekuens(kata, p_id, frames_array):
    """Menyimpan array shape (30, 63) ke disk."""
    folder = os.path.join(DATA_KATA_DIR, kata)
    idx = hitung_sampel(kata, p_id) + 1
    nama_file = f"{p_id}_{kata}_{idx:03d}.npy"
    path = os.path.join(folder, nama_file)
    np.save(path, np.asarray(frames_array, dtype=np.float32))
    return nama_file


def put(frame, text, pos, color=(255, 255, 255), scale=0.6, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    landmarker = create_landmarker(num_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses.")
        return

    kata_idx = 0  # Default kata: HALO
    state = "IDLE"  # IDLE, COUNTDOWN, RECORDING
    countdown_start = 0
    COUNTDOWN_SEC = 2.0
    recorded_frames = []
    status_msg = "Tekan SPASI untuk merekam kata terpilih"
    status_color = (255, 255, 255)
    status_timer = 0

    print("Kontrol:")
    print("  [1-4] : Pilih kata sapaan")
    print("  [SPASI]: Mulai hitung mundur & rekam gerakan")
    print("  [ESC]  : Keluar")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip cermin agar orientasi natural
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        timestamp_ms = int(now * 1000)

        kata_aktif = KATA_LIST[kata_idx]
        n_sampel = hitung_sampel(kata_aktif, person_id)

        # Deteksi tangan
        pts, handed = detect(landmarker, frame, timestamp_ms)

        if pts is not None:
            draw_hand(frame, pts)
            norm = normalize(pts)
        else:
            norm = None

        # State Machine Perekaman
        if state == "COUNTDOWN":
            elapsed = now - countdown_start
            sisa = COUNTDOWN_SEC - elapsed
            if sisa > 0:
                put(frame, f"SIAP: {sisa:.1f}s", (w // 2 - 120, h // 2), (0, 255, 255), scale=1.5, thick=4)
                put(frame, f"Gerakan: {kata_aktif}", (w // 2 - 140, h // 2 + 50), (255, 255, 255), scale=0.8)
            else:
                state = "RECORDING"
                recorded_frames = []

        elif state == "RECORDING":
            if norm is not None:
                recorded_frames.append(norm)

            progress = len(recorded_frames)
            # Progress bar
            bar_w = int((progress / SEQUENCE_LENGTH) * (w - 80))
            cv2.rectangle(frame, (40, h - 50), (w - 40, h - 30), (50, 50, 50), -1)
            cv2.rectangle(frame, (40, h - 50), (40 + bar_w, h - 30), (0, 255, 0), -1)
            put(frame, f"MEREKAM GERAKAN: {progress}/{SEQUENCE_LENGTH} frame", (40, h - 60), (0, 255, 0), scale=0.8, thick=2)

            if len(recorded_frames) >= SEQUENCE_LENGTH:
                # Selesai 30 frame
                nama = simpan_sekuens(kata_aktif, person_id, recorded_frames)
                state = "IDLE"
                status_msg = f"BERHASIL DISIMPAN: {nama} ({progress} frame)"
                status_color = (0, 255, 0)
                status_timer = now + 3.0

        # UI Header & Keterangan
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        put(frame, f"KATA AKTIF: [{kata_idx + 1}] {kata_aktif} ({n_sampel}/{TARGET_PER_KATA} sampel)", (15, 30), (0, 255, 255), scale=0.8, thick=2)
        put(frame, f"Perekam: {person_id} | Status: {state}", (15, 65), (200, 200, 200), scale=0.6)

        # Baris bantuan pemilihan kata
        pilihan_str = " | ".join([f"[{i+1}] {k}" for i, k in enumerate(KATA_LIST)])
        put(frame, pilihan_str, (15, 105), (180, 180, 180), scale=0.5)

        # Status notifikasi
        if now < status_timer:
            put(frame, status_msg, (15, h - 15), status_color, scale=0.65, thick=2)
        elif state == "IDLE":
            put(frame, "Tekan [SPASI] untuk merekam | [1-4] ganti kata | [ESC] keluar", (15, h - 15), (255, 255, 255), scale=0.55)

        cv2.imshow("TALKY - Rekam Kata Sapaan (1D-CNN)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key in [ord("1"), ord("2"), ord("3"), ord("4")]:
            if state == "IDLE":
                kata_idx = key - ord("1")
        elif key == 32:  # SPASI
            if state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = now

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")


if __name__ == "__main__":
    main()
