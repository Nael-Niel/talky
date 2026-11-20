"""
test_gerak_jz.py
Pengujian mandiri modul deteksi huruf bergerak J dan Z (gerak_jz.py).
Skrip ini TIDAK membutuhkan model training atau file data/dataset.csv.

Mekanisme:
  - Menggunakan Google MediaPipe Tasks API untuk membaca 21 titik tangan.
  - Melacak ujung kelingking (titik 20) untuk huruf J (gerak lengkung kail).
  - Melacak ujung telunjuk (titik 8) untuk huruf Z (gerak zig-zag).
  - Menggambar jejak lintasan ujung jari dengan garis kuning secara real-time.

Kontrol Keyboard:
  - ESC / Q : Keluar dari program
  - M       : Balik arah sumbu X (flip_x toggle, jika arah kamera berlawanan)
  - C       : Bersihkan jejak lintasan saat ini
"""

import os
import sys
import time

import cv2
import numpy as np

from talky_common import create_landmarker, detect, draw_hand
from gerak_jz import JZDetector


def put_text(frame, text, pos, color=(255, 255, 255), scale=0.6, tebal=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), tebal + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, tebal)


def main():
    print("=" * 60)
    print("TALKY - PENGUJIAN MANDIRI DETEKSI HURUF J DAN Z")
    print("=" * 60)
    print("Petunjuk gerakan:")
    print("  - Huruf J: Angkat hanya jari kelingking, gerakkan melengkung ke bawah lalu naik.")
    print("  - Huruf Z: Angkat hanya jari telunjuk, gerakkan zig-zag (kanan -> serong kiri -> kanan).")
    print("Kontrol: Tekan ESC atau Q untuk keluar | Tekan M untuk toggle mirror X\n")

    landmarker = create_landmarker(num_hands=1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses.")
        return

    jz = JZDetector(flip_x=False)
    t0 = time.time()
    last_detected = None
    last_detect_time = 0.0
    DISPLAY_DURATION = 2.0  # Durasi huruf hasil deteksi tampil di layar (detik)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Frame di-flip cermin agar konsisten dengan standar Talky
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            timestamp_ms = int((now - t0) * 1000)

            pts, handed = detect(landmarker, frame, timestamp_ms)

            if pts is not None:
                draw_hand(frame, pts)
                hasil = jz.update(pts, w, h, now=now)
                if hasil:
                    last_detected = hasil
                    last_detect_time = now
                    print(f"[{time.strftime('%H:%M:%S')}] Terdeteksi Huruf Dinamis: {hasil}")

            # Gambar lintasan garis kuning jejak jari
            jz.draw(frame)

            # Tampilan Status & UI
            put_text(frame, f"Deteksi Gerak J & Z (Mirror X: {jz.flip_x})", (10, 30), (0, 255, 255), 0.7)

            if pts is not None:
                pose_aktif = jz.pose
                pose_str = f"Pose Jari: {pose_aktif}" if pose_aktif else "Pose: Bukan J/Z"
                put_text(frame, f"Tangan: {handed} | {pose_str}", (10, 60), (200, 200, 200), 0.5)
            else:
                put_text(frame, "Arahkan satu tangan ke kamera", (10, 60), (0, 0, 255), 0.5)

            # Menampilkan huruf yang berhasil dikenali
            if last_detected and (now - last_detect_time < DISPLAY_DURATION):
                warna = (0, 255, 0)
                teks_hasil = f"TERDETEKSI: HURUF {last_detected}!"
                cv2.putText(frame, teks_hasil, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                cv2.putText(frame, teks_hasil, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, warna, 2)

            put_text(frame, "ESC/Q = Keluar  |  M = Balik Arah X  |  C = Hapus Jejak", (10, h - 20), (220, 220, 220), 0.5)

            cv2.imshow("Talky - Tes Gerak J & Z", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('m'), ord('M')):
                jz.flip_x = not jz.flip_x
                print(f"[INFO] Mirror X diubah menjadi: {jz.flip_x}")
            elif key in (ord('c'), ord('C')):
                for tr in jz.trails.values():
                    tr.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\nPengujian selesai.")


if __name__ == "__main__":
    main()
