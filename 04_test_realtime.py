"""
04_test_realtime.py
Pengujian Real-Time Inferensi Talky melalui Webcam Laptop.
Menggabungkan Model Neural Network MLP (Huruf Statis A-Y) dan Modul Gerak J/Z (Huruf Dinamis).

Arsitektur Pipeline:
  1. Kamera Webcam -> Frame di-flip cermin (cv2.flip).
  2. MediaPipe Tasks Vision -> Ekstraksi 21 koordinat landmark tangan 3D.
  3. Modul Gerak J/Z (gerak_jz.py) -> Deteksi trajektori gerakan kail (J) atau zig-zag (Z).
  4. Jika tidak ada gerakan J/Z -> Koordinat dinormalisasi via normalize() (63 float).
  5. Model TFLite (model/talky_mlp.tflite) -> Prediksi probabilitas softmax 24 kelas.
  6. Stabilisasi Anti-Flicker:
     - Confidence Threshold (>= 60% untuk validitas prediksi).
     - Majority Voting Buffer (5 frame terakhir agar teks di layar stabil).
  7. Penyusun Kata / Kalimat (Text Composer):
     - Huruf yang stabil otomatis ditambahkan ke kalimat atau bisa dikontrol via tombol.

Kontrol Keyboard:
  - ESC / Q     : Keluar dari program
  - SPASI       : Masukkan karakter Spasi ke kalimat
  - BACKSPACE   : Hapus 1 huruf terakhir dari kalimat
  - C           : Hapus seluruh kalimat (Clear)
  - M           : Toggle balik sumbu X untuk modul gerak J/Z
"""

import os
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

from talky_common import (FEATURE_COLUMNS, FOLDER, create_landmarker, detect,
                          draw_hand, normalize)
from gerak_jz import JZDetector

# Path model dan label
MODEL_DIR = os.path.join(FOLDER, "model")
TFLITE_PATH = os.path.join(MODEL_DIR, "talky_mlp.tflite")
KERAS_PATH = os.path.join(MODEL_DIR, "talky_mlp.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

# Parameter stabilisasi
CONFIDENCE_THRESHOLD = 0.60  # Minimal keyakinan model agar huruf dianggap valid
SMOOTHING_FRAMES = 5         # Ukuran buffer majority voting
HOLD_TO_TYPE_SEC = 0.8       # Durasi huruf stabil ditahan sebelum masuk ke kalimat


def put_text(frame, text, pos, color=(255, 255, 255), scale=0.6, tebal=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), tebal + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, tebal)


def load_classifier():
    """Memuat model TFLite (atau fallback ke Keras) dan berkas labels.txt."""
    if not os.path.exists(LABELS_PATH):
        print(f"[ERROR] Berkas labels.txt tidak ditemukan di: {LABELS_PATH}")
        sys.exit(1)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    # Utamakan model TFLite
    if os.path.exists(TFLITE_PATH):
        print(f"[INFO] Memuat model TFLite dari: {TFLITE_PATH}")
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        def predict_fn(normalized_63):
            inp = np.array([normalized_63], dtype=np.float32)
            interpreter.set_tensor(input_details[0]["index"], inp)
            interpreter.invoke()
            probs = interpreter.get_tensor(output_details[0]["index"])[0]
            idx = int(np.argmax(probs))
            return labels[idx], float(probs[idx])

        return predict_fn, labels, "TFLite"

    # Fallback ke Keras jika TFLite belum ada
    elif os.path.exists(KERAS_PATH):
        print(f"[INFO] TFLite belum ada, memuat model Keras dari: {KERAS_PATH}")
        model = tf.keras.models.load_model(KERAS_PATH)

        def predict_fn(normalized_63):
            inp = np.array([normalized_63], dtype=np.float32)
            probs = model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            return labels[idx], float(probs[idx])

        return predict_fn, labels, "Keras"

    else:
        print("[ERROR] Model tidak ditemukan di folder model/.")
        print("Silakan jalankan '02_train_mlp.ipynb' atau '03_export_tflite.py' terlebih dahulu.")
        sys.exit(1)


def main():
    print("=" * 60)
    print("TALKY - PENGUJIAN INFERENSI REAL-TIME SIBI (WEBCAM)")
    print("=" * 60)

    predict_fn, labels, engine_name = load_classifier()
    print(f"Engine Aktif   : {engine_name}")
    print(f"Jumlah Kelas   : {len(labels)} huruf statis ({', '.join(labels)})")
    print("Modul Tambahan : Deteksi gerak J & Z via JZDetector")
    print("Ambang Keyakinan : >= {:.0%}".format(CONFIDENCE_THRESHOLD))
    print("=" * 60)
    print("Kontrol Keyboard:")
    print("  - ESC / Q   : Keluar")
    print("  - SPASI     : Tambah spasi ke kalimat")
    print("  - BACKSPACE : Hapus huruf terakhir")
    print("  - C         : Hapus seluruh teks")
    print("  - M         : Toggle balik arah sumbu X (gerak J/Z)\n")

    landmarker = create_landmarker(num_hands=1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Webcam tidak terdeteksi.")
        return

    jz = JZDetector(flip_x=False)
    vote_buffer = deque(maxlen=SMOOTHING_FRAMES)
    kalimat = ""

    last_stable_letter = None
    stable_since = 0.0
    last_added_letter = None
    t0 = time.time()
    fps_time = time.time()
    fps_frames = 0
    fps_val = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Frame di-flip cermin
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            timestamp_ms = int((now - t0) * 1000)

            # Hitung FPS
            fps_frames += 1
            if now - fps_time >= 1.0:
                fps_val = fps_frames / (now - fps_time)
                fps_frames = 0
                fps_time = now

            pts, handed = detect(landmarker, frame, timestamp_ms)

            current_letter = None
            current_conf = 0.0
            source_tag = "-"

            # 1. Cek huruf dinamis J atau Z terlebih dahulu
            hasil_jz = jz.update(pts, w, h, now=now)
            if hasil_jz:
                current_letter = hasil_jz
                current_conf = 1.0
                source_tag = "Gerak J/Z"
                vote_buffer.clear()
                # Langsung masukkan ke kalimat saat gerak selesai
                kalimat += current_letter
                last_added_letter = current_letter
                print(f"[KALIMAT] + Huruf dinamis: '{current_letter}' -> {kalimat}")

            elif jz.current(now=now):
                # Sedang dalam masa cooldown setelah J/Z terdeteksi
                current_letter = jz.current(now=now)
                current_conf = 1.0
                source_tag = "Gerak J/Z"

            # 2. Jika tidak ada J/Z, lakukan klasifikasi huruf statis via model MLP
            elif pts is not None:
                draw_hand(frame, pts)
                fitur = normalize(pts)
                if fitur is not None:
                    huruf_pred, conf = predict_fn(fitur)
                    if conf >= CONFIDENCE_THRESHOLD:
                        vote_buffer.append(huruf_pred)
                        # Majority voting
                        counts = Counter(vote_buffer)
                        most_common_letter, count = counts.most_common(1)[0]
                        if count >= (SMOOTHING_FRAMES // 2 + 1):
                            current_letter = most_common_letter
                            current_conf = conf
                            source_tag = "MLP Statis"
                    else:
                        vote_buffer.append("?")
                else:
                    vote_buffer.clear()
            else:
                vote_buffer.clear()

            # Gambar lintasan garis kuning jejak jari jika ada
            jz.draw(frame)

            # Logika otomatis menambahkan huruf jika pose ditahan stabil
            if current_letter and current_letter not in ("?", None) and source_tag == "MLP Statis":
                if current_letter == last_stable_letter:
                    if (now - stable_since >= HOLD_TO_TYPE_SEC) and (last_added_letter != current_letter):
                        kalimat += current_letter
                        last_added_letter = current_letter
                        print(f"[KALIMAT] + '{current_letter}' -> {kalimat}")
                else:
                    last_stable_letter = current_letter
                    stable_since = now
                    last_added_letter = None
            else:
                if now - stable_since > 0.5:
                    last_stable_letter = None
                    last_added_letter = None

            # ================= TAMPILAN ANTARMUKA (HUD) =================
            # Header info
            put_text(frame, f"TALKY REAL-TIME ({engine_name}) | FPS: {fps_val:.1f}", (10, 30), (0, 255, 255), 0.65)
            handed_text = f"Tangan: {handed}" if pts is not None else "Arahkan tangan ke kamera"
            put_text(frame, handed_text, (10, 60), (200, 200, 200), 0.5)

            # Kotak display huruf hasil tebakan
            if current_letter and current_letter != "?":
                warna_box = (0, 255, 0)
                teks_pred = f"{current_letter}"
                teks_detail = f"{current_conf*100:.1f}% ({source_tag})"
            else:
                warna_box = (0, 165, 255)
                teks_pred = "?"
                teks_detail = "Tidak yakin / Di bawah threshold" if pts is not None else "Menunggu tangan..."

            # Gambar kartu prediksi di kanan atas
            cv2.rectangle(frame, (w - 180, 15), (w - 15, 125), (40, 40, 40), -1)
            cv2.rectangle(frame, (w - 180, 15), (w - 15, 125), warna_box, 2)
            cv2.putText(frame, teks_pred, (w - 130, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            cv2.putText(frame, teks_detail, (w - 175, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

            # Baris Kalimat Tersusun (Bottom Bar)
            cv2.rectangle(frame, (0, h - 75), (w, h), (25, 25, 25), -1)
            cv2.line(frame, (0, h - 75), (w, h - 75), (100, 100, 100), 1)
            teks_tampil = kalimat if kalimat else "Ketik dengan memperagakan huruf..."
            warna_kalimat = (255, 255, 255) if kalimat else (150, 150, 150)
            put_text(frame, f"Teks: {teks_tampil}", (15, h - 45), warna_kalimat, 0.75, tebal=2)
            put_text(frame, "SPASI: Tambah Spasi | BACKSPACE: Hapus | C: Clear | ESC: Keluar", (15, h - 15), (180, 180, 180), 0.45, tebal=1)

            cv2.imshow("Talky - Uji Coba Real-Time", frame)

            # Penanganan Tombol Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # ESC atau Q
                break
            elif key == 32:  # SPASI
                kalimat += " "
                print(f"[KALIMAT] + Spasi -> '{kalimat}'")
            elif key in (8, 127):  # Backspace
                kalimat = kalimat[:-1]
                print(f"[KALIMAT] - Hapus -> '{kalimat}'")
            elif key in (ord('c'), ord('C')):  # Clear
                kalimat = ""
                print("[KALIMAT] Kalimat dibersihkan.")
            elif key in (ord('m'), ord('M')):  # Mirror toggle J/Z
                jz.flip_x = not jz.flip_x
                print(f"[INFO] Mirror J/Z diubah ke: {jz.flip_x}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\nSelesai. Kalimat akhir yang terbentuk:")
        print(f"'{kalimat}'")


if __name__ == "__main__":
    main()
