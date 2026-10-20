"""
talky_common.py
Fungsi bersama untuk semua skrip Talky (rekam data, tes real-time, dll).
PENTING: normalize() harus SAMA di training dan di aplikasi mobile.
"""
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(FOLDER, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# Nama kolom fitur: x0,y0,z0,x1,y1,z1,...,x20,y20,z20 (63 kolom)
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]

# Pasangan titik yang dihubungkan untuk menggambar kerangka tangan
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def create_landmarker(num_hands=1):
    """Buat HandLandmarker (mode VIDEO). Model diunduh otomatis jika belum ada."""
    if not os.path.exists(MODEL_PATH):
        print("Mengunduh model hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Selesai.")
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect(landmarker, frame_bgr, timestamp_ms):
    """
    Deteksi tangan pada satu frame (BGR).
    Return: (pts, handedness) atau (None, None) jika tidak ada tangan.
    pts = array (21, 3) berisi x, y, z mentah dari MediaPipe.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None, None
    pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]],
                   dtype=np.float32)
    handed = result.handedness[0][0].category_name if result.handedness else "?"
    return pts, handed


def normalize(pts):
    """
    pts: array (21, 3) mentah dari MediaPipe.
    1) Translasi: pergelangan tangan (titik 0) menjadi titik asal (0,0,0).
    2) Skala: bagi dengan jarak terjauh, sehingga ukuran tangan tidak berpengaruh.
    Return: array 63 nilai, atau None jika data tidak valid.
    """
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6:
        return None
    return (pts / scale).flatten()


def draw_hand(frame, pts):
    """Gambar kerangka tangan di frame. pts: array (21, 3) mentah (0-1)."""
    h, w = frame.shape[:2]
    xy = [(int(p[0] * w), int(p[1] * h)) for p in pts]
    for a, b in CONNECTIONS:
        cv2.line(frame, xy[a], xy[b], (255, 255, 255), 2)
    for x, y in xy:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
