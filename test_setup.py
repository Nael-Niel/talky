import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

print("MediaPipe versi:", mp.__version__)

FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FOLDER, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# Unduh model sekali saja
if not os.path.exists(MODEL_PATH):
    print("Mengunduh model hand_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Selesai.")

with open(MODEL_PATH, "rb") as f:
    model_bytes = f.read()

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Pasangan titik yang dihubungkan untuk menggambar kerangka tangan
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

cap = cv2.VideoCapture(0)
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera tidak terbaca. Coba ganti VideoCapture(0) menjadi VideoCapture(1).")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int((time.time() - t0) * 1000)
    hasil = landmarker.detect_for_video(mp_image, timestamp_ms)

    if hasil.hand_landmarks:
        h, w = frame.shape[:2]
        titik = [(int(p.x * w), int(p.y * h)) for p in hasil.hand_landmarks[0]]
        for a, b in CONNECTIONS:
            cv2.line(frame, titik[a], titik[b], (255, 255, 255), 2)
        for x, y in titik:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(frame, "Tangan terdeteksi (21 titik)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tidak ada tangan", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Tes Talky (tekan Q untuk keluar)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()