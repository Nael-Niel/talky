"""
test_dua_tangan.py
Tes deteksi DUA tangan sekaligus (file baru, tidak mengubah skrip lain).

- Memakai create_landmarker(num_hands=2) dari talky_common.py.
- normalize() TIDAK diubah: dipakai apa adanya, satu kali per tangan.
- Tidak menyentuh data/dataset.csv.

Jalankan dari folder talky-ml dengan venv aktif:
    python -m tests.test_dua_tangan
Tekan ESC atau Q untuk keluar.
"""
import time

import cv2
import numpy as np
import mediapipe as mp

from talky_common import create_landmarker, draw_hand, normalize


def detect_all(landmarker, frame_bgr, timestamp_ms):
    """
    Seperti detect() di talky_common.py, tetapi mengembalikan SEMUA tangan.
    Return: list of (pts, handedness). pts = array (21, 3) mentah dari MediaPipe.
    List kosong jika tidak ada tangan.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    tangan = []
    for i, lm in enumerate(result.hand_landmarks):
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        handed = result.handedness[i][0].category_name if result.handedness else "?"
        tangan.append((pts, handed))
    return tangan


def put(frame, text, pos, color=(255, 255, 255), scale=0.6, tebal=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), tebal + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, tebal)


landmarker = create_landmarker(num_hands=2)
cap = cv2.VideoCapture(0)  # ganti ke 1 jika kamera tidak terbaca
t0 = time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kamera tidak terbaca. Coba ganti VideoCapture(0) menjadi VideoCapture(1).")
            break

        frame = cv2.flip(frame, 1)  # sama seperti pipeline proyek
        h, w = frame.shape[:2]
        tangan = detect_all(landmarker, frame, int((time.time() - t0) * 1000))

        # Urutkan dari kiri ke kanan di layar (berdasarkan posisi pergelangan),
        # supaya urutan tangan stabil dan tidak bergantung pada label handedness.
        tangan.sort(key=lambda t: t[0][0][0])

        fitur_semua = []
        for pts, handed in tangan:
            draw_hand(frame, pts)
            x, y = int(pts[0][0] * w), int(pts[0][1] * h)
            put(frame, handed, (x - 20, y + 25), (0, 255, 255), 0.6)
            f = normalize(pts)          # 63 nilai per tangan, normalize() tidak diubah
            if f is not None:
                fitur_semua.append(f)

        put(frame, f"Tangan terdeteksi: {len(tangan)}", (10, 28), (0, 255, 0), 0.8)
        if len(fitur_semua) == 2:
            vektor = np.concatenate(fitur_semua)
            put(frame, f"Fitur dua tangan: {vektor.shape[0]} nilai (2 x 63)",
                (10, 58), (255, 255, 0), 0.6)
        elif len(fitur_semua) == 1:
            put(frame, "Fitur satu tangan: 63 nilai", (10, 58), (255, 255, 0), 0.6)

        cv2.imshow("Talky - Tes Dua Tangan", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
