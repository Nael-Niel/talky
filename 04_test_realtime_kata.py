"""
04_test_realtime_kata.py
Uji coba inferensi real-time webcam untuk kata sapaan dasar SIBI menggunakan model 1D-CNN (TFLite).

Mekanisme:
  - Menggunakan circular buffer (FIFO) sepanjang 30 frame.
  - Setiap kali tangan terdeteksi, koordinat ternormalisasi (63 float) dimasukkan ke buffer.
  - Saat buffer penuh (30 frame), model 1D-CNN memprediksi kata sapaan secara kontinyu.
  - Ambang keyakinan (threshold) default >= 70%.

Kontrol:
  - ESC : Keluar
"""
import os
import sys
import time
from collections import deque
import cv2
import numpy as np
import tensorflow as tf

from talky_common import (FOLDER, create_landmarker, detect,
                          draw_hand, normalize)

MODEL_DIR = os.path.join(FOLDER, "model")
TFLITE_PATH = os.path.join(MODEL_DIR, "talky_kata_1dcnn.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels_kata.txt")

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70


def put(frame, text, pos, color=(255, 255, 255), scale=0.6, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    print("=" * 60)
    print("TALKY - Uji Coba Real-Time Kata Sapaan SIBI (1D-CNN)")
    print("=" * 60)

    if not os.path.exists(TFLITE_PATH) or not os.path.exists(LABELS_PATH):
        print("Error: Model TFLite atau labels_kata.txt belum ditemukan.")
        print("Pastikan sudah menjalankan 01_collect_kata.py, 02_train_1dcnn_kata.py, dan 03_export_kata_tflite.py.")
        sys.exit(1)

    # 1. Muat labels
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"Label kata ({len(labels)} kelas): {labels}")

    # 2. Muat TFLite
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Inisialisasi MediaPipe & Kamera
    landmarker = create_landmarker(num_hands=1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak dapat dibuka.")
        sys.exit(1)

    buffer = deque(maxlen=SEQUENCE_LENGTH)
    kata_terprediksi = "-"
    confidence_terprediksi = 0.0

    print("Kamera aktif. Arahkan tangan ke kamera dan peragakan kata sapaan...")
    print("Tekan [ESC] untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        timestamp_ms = int(now * 1000)

        pts, handed = detect(landmarker, frame, timestamp_ms)

        if pts is not None:
            draw_hand(frame, pts)
            norm = normalize(pts)
            if norm is not None:
                buffer.append(norm)
        else:
            # Jika tangan hilang dari layar, reset buffer bertahap
            if len(buffer) > 0 and (now % 0.5 < 0.1):
                buffer.popleft()

        # Jika buffer sudah terkumpul 30 frame, lakukan inferensi
        if len(buffer) == SEQUENCE_LENGTH:
            input_tensor = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])[0]

            idx_max = int(np.argmax(preds))
            conf = float(preds[idx_max])

            if conf >= CONFIDENCE_THRESHOLD:
                kata_terprediksi = labels[idx_max]
                confidence_terprediksi = conf
            else:
                kata_terprediksi = "(Ragu / Kurang Jelas)"
                confidence_terprediksi = conf

        # UI Visualisasi
        cv2.rectangle(frame, (0, 0), (w, 85), (30, 30, 30), -1)
        put(frame, "TALKY - MODE KATA SAPAAN DINAMIS (1D-CNN)", (15, 28), (0, 255, 255), scale=0.75, thick=2)

        buffer_info = f"Buffer: {len(buffer)}/{SEQUENCE_LENGTH} frame"
        cv2.putText(frame, buffer_info, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Bilah Hasil Tebakan
        cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)
        if kata_terprediksi != "-" and kata_terprediksi != "(Ragu / Kurang Jelas)":
            teks_hasil = f"TERJEMAHAN: {kata_terprediksi} ({confidence_terprediksi * 100:.1f}%)"
            put(frame, teks_hasil, (20, h - 25), (0, 255, 0), scale=0.85, thick=2)
        else:
            put(frame, f"Menunggu gerakan... {kata_terprediksi}", (20, h - 25), (180, 180, 180), scale=0.65)

        cv2.imshow("TALKY - Test Real-Time Kata Sapaan (1D-CNN)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")


if __name__ == "__main__":
    main()
