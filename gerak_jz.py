"""
gerak_jz.py
Modul deteksi huruf bergerak J dan Z berbasis aturan (tanpa training).
Tidak membuka kamera dan tidak menyentuh normalize() / dataset.csv.

Input landmark = array (21, 3) mentah dari detect() di talky_common.py
(nilai x, y sudah 0-1, BELUM dinormalisasi).

Contoh pemakaian di dalam loop kamera:

    from gerak_jz import JZDetector
    jz = JZDetector()                         # buat sekali, sebelum loop

    h, w = frame.shape[:2]
    pts, _ = detect(landmarker, frame, ts)    # pts = None jika tidak ada tangan
    jz.update(pts, w, h)
    huruf_final = jz.current() or huruf_statis
    jz.draw(frame)
"""
import time
from collections import deque

import cv2
import numpy as np

# ---------------- Pengaturan (sama seperti test_gerak_jz.py) ----------------
WINDOW_SEC = 1.5
MAX_GAP_SEC = 0.35
MIN_POINTS = 8
MIN_PATH_J = 1.2
MIN_PATH_Z = 2.0
SEG_MIN = 0.3
COOLDOWN_SEC = 1.5
# ----------------------------------------------------------------------------

TIP = {"J": 20, "Z": 8}   # J = ujung kelingking, Z = ujung telunjuk


def _pt(pts, i, w, h):
    return np.array([pts[i][0] * w, pts[i][1] * h])


def _is_up(pts, tip, pip, w, h):
    wrist = _pt(pts, 0, w, h)
    d_tip = np.linalg.norm(_pt(pts, tip, w, h) - wrist)
    d_pip = np.linalg.norm(_pt(pts, pip, w, h) - wrist)
    return d_tip > d_pip * 1.1


def get_pose(pts, w, h):
    """Kembalikan 'J', 'Z', atau None berdasarkan jari yang terangkat."""
    idx = _is_up(pts, 8, 6, w, h)
    mid = _is_up(pts, 12, 10, w, h)
    ring = _is_up(pts, 16, 14, w, h)
    pinky = _is_up(pts, 20, 18, w, h)
    if pinky and not (idx or mid or ring):
        return "J"
    if idx and not (mid or ring or pinky):
        return "Z"
    return None


def _resample(points, n=4):
    arr = np.array(points, dtype=float)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    dist = np.concatenate([[0], np.cumsum(seg)])
    total = dist[-1]
    if total == 0:
        return None, 0.0
    targets = np.linspace(0, total, n)
    xs = np.interp(targets, dist, arr[:, 0])
    ys = np.interp(targets, dist, arr[:, 1])
    return np.stack([xs, ys], axis=1), total


def _classify(letter, trail, hand_size, flip_x):
    if len(trail) < MIN_POINTS or hand_size <= 0:
        return False
    sx = -1 if flip_x else 1
    pts = [[p[1] * sx, p[2]] for p in trail]
    p4, total = _resample(pts, 4)
    if p4 is None:
        return False
    d = np.diff(p4, axis=0) / hand_size
    path = total / hand_size

    if letter == "J":
        down = (p4[3, 1] - p4[0, 1]) / hand_size
        return (path >= MIN_PATH_J and down > 0.5
                and d[0, 1] > SEG_MIN
                and d[2, 0] < -SEG_MIN)

    if letter == "Z":
        return (path >= MIN_PATH_Z
                and d[0, 0] > SEG_MIN
                and d[1, 0] < -SEG_MIN
                and d[1, 1] > SEG_MIN * 0.5
                and d[2, 0] > SEG_MIN)
    return False


class JZDetector:
    def __init__(self, flip_x=False):
        # flip_x: isi True jika di tes sebelumnya arah harus dibalik dengan tombol M
        self.flip_x = flip_x
        self.trails = {"J": deque(), "Z": deque()}
        self.pose = None
        self.last_letter = None
        self.last_time = 0.0

    def update(self, pts, w, h, now=None):
        """Panggil tiap frame. pts = array (21,3) dari detect(), atau None.
        Mengembalikan 'J'/'Z' pada frame saat huruf baru dikenali, selain itu None."""
        now = time.time() if now is None else now
        self.pose = None
        hand_size = 0.0

        if pts is not None:
            hand_size = np.linalg.norm(_pt(pts, 0, w, h) - _pt(pts, 9, w, h))
            self.pose = get_pose(pts, w, h)
            if self.pose and now - self.last_time > COOLDOWN_SEC:
                x, y = _pt(pts, TIP[self.pose], w, h)
                tr = self.trails[self.pose]
                if tr and now - tr[-1][0] > MAX_GAP_SEC:
                    tr.clear()
                tr.append((now, x, y))

        for tr in self.trails.values():
            while tr and now - tr[0][0] > WINDOW_SEC:
                tr.popleft()

        if now - self.last_time > COOLDOWN_SEC:
            for letter, tr in self.trails.items():
                if _classify(letter, tr, hand_size, self.flip_x):
                    self.last_letter, self.last_time = letter, now
                    for t in self.trails.values():
                        t.clear()
                    return letter
        return None

    def current(self, now=None):
        """Huruf J/Z yang baru dikenali (bertahan COOLDOWN_SEC detik), atau None."""
        now = time.time() if now is None else now
        if self.last_letter and now - self.last_time < COOLDOWN_SEC:
            return self.last_letter
        return None

    def draw(self, frame):
        """Gambar jejak ujung jari (garis kuning) di frame."""
        for tr in self.trails.values():
            if len(tr) >= 2:
                arr = np.array([[p[1], p[2]] for p in tr], dtype=np.int32)
                cv2.polylines(frame, [arr], False, (0, 255, 255), 3)