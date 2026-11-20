"""
ekstrak_gambar.py
Mengubah gambar dataset luar (satu folder per huruf) menjadi baris landmark
ternormalisasi, lalu menyimpannya ke CSV terpisah (format sama dengan dataset.csv).

- TIDAK membuka kamera. Hanya membaca gambar.
- normalize() dipakai apa adanya (diimpor dari talky_common.py, tidak diubah).
- TIDAK menyentuh data/dataset.csv.
- Huruf J dan Z dilewati (ditangani gerak_jz.py, tidak ikut MLP).

Perilaku bawaan yang dirapikan agar cocok dengan MediaPipe HandLandmarker:
  1) Gambar grayscale (hitam-putih) dilewati karena akurasi & tingkat deteksinya rendah
     (pakai --ikut-gray jika tetap ingin mencobanya).
  2) Ukuran gambar disesuaikan otomatis ke rentang 400-640 px tanpa mengubah rasio,
     supaya gambar kecil terbaca dan gambar raksasa (mis. 3000x3000) tetap cepat.
  3) Deteksi 2 tahap otomatis: tanpa bingkai dulu -> jika gagal, tambah bingkai (--pad 0.3).
  4) Saat --flip dipakai, hanya sisi 'Left' yang disimpan (selaras dengan dataset.csv)
     untuk membuang landmark yang tertukar kiri/kanan.

Cara pakai (dari folder proyek, venv aktif):

  1) Uji dulu tanpa menulis CSV (maks. 20 gambar per huruf, simpan contoh ke contoh_hasil/):
       python ekstrak_gambar.py --coba --folder data_luar/sibi_1
       python ekstrak_gambar.py --coba --flip --folder data_luar/sibi_1

  2) Setelah cocok, tulis CSV terpisah:
       python ekstrak_gambar.py --flip --folder data_luar/sibi_1 --out data/dataset_luar_sibi1.csv --person sibi1

Opsi utama:
  --folder     folder gambar            (bawaan: data_luar/sibi_kaggle)
  --out        file CSV keluaran        (bawaan: data/dataset_luar_kaggle.csv)
  --person     isi kolom person_id      (bawaan: kaggle)
  --maks N     maksimum baris valid per huruf (diambil acak, seed tetap)
  --flip       cerminkan gambar sebelum deteksi (samakan dengan pipeline kamera)
  --coba       uji maks. 20 gambar per huruf tanpa menulis CSV
  --timpa      boleh menimpa CSV keluaran yang sudah ada
  --sisi S     filter handedness: Left, Right, atau semua
               (bawaan: Left jika --flip aktif, selainnya semua)
  --pad F      pakai bingkai manual 1 tahap (mis. 0.3, atau 0 tanpa bingkai);
               jika tidak diisi, otomatis pakai 2 tahap (tanpa pad -> pad 0.3)
  --skala K    skala manual (mis. 1.0 ukuran asli, 2.0 perbesar 2x);
               jika tidak diisi, otomatis disesuaikan ke rentang 400-640 px
  --conf C     ambang keyakinan deteksi tangan MediaPipe (bawaan: 0.5)
  --ikut-gray  ikut memproses gambar grayscale (bawaan: dilewati)
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from talky_common import FEATURE_COLUMNS, FOLDER, MODEL_PATH, draw_hand, normalize

EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DILEWATI = {"J", "Z"}
HEADER = ["person_id", "label", "handedness"] + FEATURE_COLUMNS
SEED = 42
JUMLAH_COBA = 20

# Batas penyesuaian resolusi adaptif (menjaga rasio aspek tetap sama)
BATAS_KECIL = 350
TARGET_MIN_SISI = 400
BATAS_BESAR = 800
TARGET_MAX_SISI = 640
PAD_TAHAP_2 = 0.3


def baca_argumen():
    p = argparse.ArgumentParser(description="Gambar dataset luar -> CSV landmark")
    p.add_argument("--folder", default=os.path.join(FOLDER, "data_luar", "sibi_kaggle"))
    p.add_argument("--out", default=os.path.join(FOLDER, "data", "dataset_luar_kaggle.csv"))
    p.add_argument("--person", default="kaggle")
    p.add_argument("--maks", type=int, default=None)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--coba", action="store_true")
    p.add_argument("--timpa", action="store_true")
    p.add_argument("--pad", type=float, default=None)
    p.add_argument("--skala", type=float, default=None)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--sisi", choices=["Left", "Right", "semua"], default=None)
    p.add_argument("--ikut-gray", action="store_true")
    return p.parse_args()


def kumpulkan_gambar(folder):
    """Kembalikan {huruf: [path, ...]} dan daftar nama folder induk yang bukan huruf."""
    per_huruf = defaultdict(list)
    bukan_huruf = Counter()
    for akar, _, berkas in os.walk(folder):
        for nama in sorted(berkas):
            if os.path.splitext(nama)[1].lower() not in EKSTENSI:
                continue
            induk = os.path.basename(akar).strip()
            huruf = induk.upper()
            if len(huruf) == 1 and "A" <= huruf <= "Z":
                per_huruf[huruf].append(os.path.join(akar, nama))
            else:
                bukan_huruf[induk] += 1
    return per_huruf, bukan_huruf


def buat_detektor(conf=0.5):
    if not os.path.exists(MODEL_PATH):
        print("File hand_landmarker.task tidak ditemukan di:", MODEL_PATH)
        print("Jalankan salah satu skrip yang memakai kamera sekali (mis. 01_collect_data.py) "
              "agar model diunduh, atau salin file itu ke folder proyek.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    opsi = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.IMAGE,   # gambar lepas, bukan video
        num_hands=1,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
    )
    return vision.HandLandmarker.create_from_options(opsi)


def baca_gambar(path):
    """cv2.imread bisa gagal pada path tertentu di Windows; imdecode lebih aman."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def adalah_grayscale(img):
    """Benar jika kanal B, G, R identik di semua piksel (gambar hitam-putih)."""
    return bool(np.array_equal(img[:, :, 0], img[:, :, 1])
                and np.array_equal(img[:, :, 1], img[:, :, 2]))


def sesuaikan_ukuran(img, skala=None):
    """Ubah resolusi gambar secara proporsional tanpa mengubah rasio aspek.
    Hanya mengubah citra masukan sebelum deteksi; normalize() tidak disentuh."""
    if skala is not None:
        if skala == 1.0:
            return img
        interp = cv2.INTER_AREA if skala < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(img, None, fx=skala, fy=skala, interpolation=interp)

    h, w = img.shape[:2]
    sisi = max(h, w)
    if sisi > BATAS_BESAR:
        faktor = TARGET_MAX_SISI / sisi
        return cv2.resize(img, (int(round(w * faktor)), int(round(h * faktor))),
                          interpolation=cv2.INTER_AREA)
    if sisi < BATAS_KECIL:
        faktor = TARGET_MIN_SISI / sisi
        return cv2.resize(img, (int(round(w * faktor)), int(round(h * faktor))),
                          interpolation=cv2.INTER_CUBIC)
    return img


def tambah_bingkai(img, pad):
    """Beri bingkai warna rata-rata agar tangan yang mepet tepi tetap terdeteksi."""
    if pad <= 0:
        return img
    p = int(round(pad * max(img.shape[:2])))
    warna = [int(v) for v in img.reshape(-1, 3).mean(axis=0)]
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=warna)


def deteksi_gambar(detektor, img_awal, pad_manual=None):
    """Deteksi 1 tangan (2 tahap jika pad_manual None: tanpa pad -> pad 0.3).
    Return: (pts, sisi, fitur, img_dipakai, tahap) atau (None, None, None, None, 0).
    """
    if pad_manual is not None:
        kandidat_daftar = [(1, tambah_bingkai(img_awal, pad_manual))]
    else:
        kandidat_daftar = [(1, img_awal), (2, tambah_bingkai(img_awal, PAD_TAHAP_2))]

    for tahap, kandidat in kandidat_daftar:
        rgb = cv2.cvtColor(kandidat, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hasil = detektor.detect(mp_image)
        if not hasil.hand_landmarks:
            continue
        pts = np.array([[p.x, p.y, p.z] for p in hasil.hand_landmarks[0]],
                       dtype=np.float32)
        fitur = normalize(pts)
        if fitur is None:
            continue
        sisi = hasil.handedness[0][0].category_name if hasil.handedness else "?"
        return pts, sisi, fitur, kandidat, tahap

    return None, None, None, None, 0


def simpan_gambar(path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(path)


def bersihkan_folder_contoh(folder_contoh):
    """Hapus .jpg lama di contoh_hasil/ agar isi folder selalu dari uji terbaru."""
    os.makedirs(folder_contoh, exist_ok=True)
    for nama in os.listdir(folder_contoh):
        if nama.lower().endswith(".jpg"):
            os.remove(os.path.join(folder_contoh, nama))


def main():
    args = baca_argumen()

    if not os.path.isdir(args.folder):
        print("Folder gambar tidak ditemukan:", args.folder)
        print("Pastikan dataset sudah diekstrak ke sana, atau pakai --folder.")
        sys.exit(1)

    dataset_utama = os.path.abspath(os.path.join(FOLDER, "data", "dataset.csv"))
    if not args.coba and os.path.abspath(args.out) == dataset_utama:
        print("Ditolak: --out tidak boleh menimpa data/dataset.csv.")
        print("Gunakan nama terpisah, misalnya: data/dataset_luar_sibi1.csv")
        sys.exit(1)

    if not args.coba and os.path.exists(args.out) and not args.timpa:
        print("CSV keluaran sudah ada:", args.out)
        print("Hapus/pindahkan dulu, atau jalankan dengan --timpa jika memang ingin menimpa.")
        sys.exit(1)

    per_huruf, bukan_huruf = kumpulkan_gambar(args.folder)
    if not per_huruf:
        print("Tidak ada gambar yang labelnya bisa dibaca dari nama folder.")
        if bukan_huruf:
            print("Nama folder induk yang ditemukan (contoh):", list(bukan_huruf)[:10])
        print("Skrip ini mengharapkan struktur .../<HURUF>/gambar.jpg. "
              "Kirim struktur foldernya supaya skrip bisa disesuaikan.")
        sys.exit(1)

    filter_sisi = args.sisi if args.sisi is not None else ("Left" if args.flip else "semua")
    rng = random.Random(SEED)
    terpilih = {}
    for huruf in sorted(per_huruf):
        if huruf in DILEWATI:
            continue
        daftar = list(per_huruf[huruf])
        if args.coba or args.maks is not None:
            rng.shuffle(daftar)
        terpilih[huruf] = daftar

    mode_skala = "adaptif 400-640px" if args.skala is None else f"{args.skala}x"
    mode_pad = f"otomatis 2-tahap (0 -> {PAD_TAHAP_2})" if args.pad is None else f"{args.pad} (1-tahap)"

    print(f"Folder     : {args.folder}")
    print(f"Mode       : {'COBA (tidak menulis CSV)' if args.coba else 'TULIS CSV'} | "
          f"flip={args.flip} | sisi={filter_sisi} | conf={args.conf}")
    print(f"Pra-olah   : skala={mode_skala} | pad={mode_pad} | "
          f"grayscale={'diproses' if args.ikut_gray else 'dilewati'}")
    print(f"Huruf      : {', '.join(terpilih)} | J/Z dilewati jika ada")
    if bukan_huruf:
        print(f"Catatan    : {sum(bukan_huruf.values())} gambar di folder bukan-huruf diabaikan "
              f"(contoh folder: {list(bukan_huruf)[:5]})")
    print()

    detektor = buat_detektor(args.conf)
    folder_contoh = os.path.join(FOLDER, "data_luar", "contoh_hasil")
    if args.coba:
        bersihkan_folder_contoh(folder_contoh)

    print(f"{'Huruf':<6}{'Diproses':>9}{'Terbaca':>9}{'Ditolak':>9}{'Gagal':>7}"
          f"{'Gray':>7}{'Rusak':>7}  Handedness")

    mulai = time.time()
    baris = []
    tot_proses = tot_ok = tot_ditolak = tot_gagal = tot_gray = tot_rusak = 0
    handed_total = Counter()
    tahap_total = Counter()

    try:
        for huruf, daftar in terpilih.items():
            diproses = ok_n = ditolak = gagal = gray_n = rusak = 0
            handed = Counter()
            contoh_tersimpan = False

            for path in daftar:
                if args.coba and diproses >= JUMLAH_COBA:
                    break
                if not args.coba and args.maks is not None and ok_n >= args.maks:
                    break

                img = baca_gambar(path)
                if img is None:
                    rusak += 1
                    continue

                if not args.ikut_gray and adalah_grayscale(img):
                    gray_n += 1
                    continue

                diproses += 1
                if args.flip:
                    img = cv2.flip(img, 1)
                img = sesuaikan_ukuran(img, args.skala)

                pts, sisi, fitur, img_pakai, tahap = deteksi_gambar(
                    detektor, img, pad_manual=args.pad
                )
                if pts is None:
                    gagal += 1
                    continue

                handed[sisi] += 1
                if filter_sisi != "semua" and sisi != filter_sisi:
                    ditolak += 1
                    continue

                ok_n += 1
                tahap_total[tahap] += 1
                if not args.coba:
                    baris.append([args.person, huruf, sisi] +
                                 [round(float(v), 6) for v in fitur])
                elif not contoh_tersimpan:
                    tampil = img_pakai.copy()
                    draw_hand(tampil, pts)
                    label_foto = f"{huruf} ({sisi}) T{tahap}"
                    cv2.putText(tampil, label_foto, (8, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
                    cv2.putText(tampil, label_foto, (8, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    simpan_gambar(os.path.join(folder_contoh, f"{huruf}.jpg"), tampil)
                    contoh_tersimpan = True

            tot_proses += diproses
            tot_ok += ok_n
            tot_ditolak += ditolak
            tot_gagal += gagal
            tot_gray += gray_n
            tot_rusak += rusak
            handed_total.update(handed)

            print(f"{huruf:<6}{diproses:>9}{ok_n:>9}{ditolak:>9}{gagal:>7}"
                  f"{gray_n:>7}{rusak:>7}  {dict(handed)}", flush=True)
    finally:
        detektor.close()

    durasi = time.time() - mulai
    persen = (tot_ok / tot_proses) if tot_proses else 0.0
    print(f"\nSelesai dalam {durasi:.1f} detik.")
    print(f"Total      : {tot_ok} baris valid dari {tot_proses} gambar diproses ({persen:.1%}).")
    if tot_ditolak or tot_gray or tot_rusak:
        print(f"Disaring   : {tot_ditolak} sisi lain ditolak, {tot_gray} grayscale dilewati, "
              f"{tot_rusak} rusak.")
    if tahap_total:
        print(f"Asal tahap : {dict(sorted(tahap_total.items()))}")
    print("Handedness terdeteksi:", dict(handed_total))
    print("Pembanding : pada rekaman sendiri (data/dataset.csv), hampir semua sampel berlabel 'Left'.")

    if not args.flip and handed_total.get("Right", 0) > handed_total.get("Left", 0):
        print("Catatan    : mayoritas terdeteksi 'Right'; coba jalankan ulang dengan --flip.")

    if args.coba:
        print(f"\nContoh gambar berkerangka disimpan di: {folder_contoh}")
        print("Periksa contoh_hasil/: apakah kerangka menempel rapi dan pose sesuai Kamus SIBI.")
        return

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(baris)
    print(f"\nTersimpan: {len(baris)} baris -> {args.out}")


if __name__ == "__main__":
    main()
