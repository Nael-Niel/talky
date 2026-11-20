# 📘 TALKY: Real-Time SIBI Sign Language Recognition Engine
> **Spesifikasi Arsitektur Sistem, Rekayasa Pipeline ML, dan Desain Mobile Deployment**
> *Author: Tim Pengembang TALKY (Nathaniel Andyno Gunawan & Sheva Padhista Putri Sanny)*
> *Versi Dokumen: 2.1 (Pembaruan Modul Temporal 1D-CNN, Normalisasi Kanonikal & Deployment Flutter)*
> *Status: Production-Ready Architectural Blueprint*

---

## 1. Pendahuluan & Batasan Masalah

Proyek **TALKY** adalah sistem penerjemah Sistem Isyarat Bahasa Indonesia (SIBI) secara *real-time* berbasis perangkat bergerak (*mobile edge-computing*). Sistem dirancang agar dapat berjalan secara mandiri (*offline-first*) langsung di CPU smartphone tanpa memerlukan server eksternal, meminimalkan latensi, dan menjaga privasi pengguna.

### 1.1 Batasan Masalah Penelitian

1. **Modul Ejaan Alfabet (Fingerspelling - Fase 1 / Target Utama TA):**
   * **24 Huruf Statis:** Huruf `A–I` dan `K–Y` diproses secara instan per *single-frame* citra menggunakan arsitektur **Multilayer Perceptron (MLP)**.
   * **2 Huruf Dinamis:** Huruf `J` dan `Z` diproses melalui analisis trajektori spasial (*rule-based movement tracking*) terhadap koordinat ujung jari telunjuk dan kelingking.
2. **Modul Kata Sapaan Dasar (Dynamic Gestures - Fase 2 / Pengembangan Lanjutan):**
   * Mengklasifikasikan frasa/kata sapaan dinamis berbasis waktu (seperti: *Halo, Terima Kasih, Sama-Sama, Apa Kabar, Maaf*).
   * Memproses sekuens deret koordinat menggunakan arsitektur **1D-CNN (Temporal Convolution)** di atas koordinat MediaPipe, **bukan 2D-CNN berbasis citra piksel/video mentah**.
3. **Hardware Target:** Perangkat Android kelas menengah ke bawah (*mid-to-low end*) dengan batasan komputasi memori dan daya baterai, dibangun menggunakan kerangka kerja **Flutter (Dart)** dan `tflite_flutter`.

---

## 2. Diagram Alur Sistem (System Architecture)

Pipeline pemrosesan data dirancang modular dan deterministik untuk menghindari *bottleneck* performa di runtime mobile:

```text
+-----------------------------------+
|        Kamera HP (30 FPS)         |
+-----------------------------------+
                  │ (Flip Cermin Kamera Depan)
                  ▼
+-----------------------------------+
|      MediaPipe Hands Native       |
| (Ekstraksi 21 Titik 3D: x, y, z)  |
+-----------------------------------+
                  │
                  ▼
+-----------------------------------+
|    Normalisasi Kanonikal Lokal    |
| (Translasi Wrist + Skala Max-Dist)|
+-----------------------------------+
                  │
        +---------┴-----------------------+
        │ [Mode Ejaan Huruf]              │ [Mode Kata Sapaan - Fase 2]
        ▼                                 ▼
+-----------------------+         +-----------------------+
|  Array 1 Frame (1x63) |         |  Ring Buffer (30x63)  |
+-----------------------+         +-----------------------+
        │                                 │
        ├──► Huruf Statis (A-I, K-Y)      ▼
        │    └─► [MLP Inference Engine]   +-----------------------+
        │                                 | 1D-CNN Temporal Eng.  |
        └──► Huruf Dinamis (J, Z)         +-----------------------+
             └─► [Trajectory Tracking]            │
                        │                         │
                        ▼                         ▼
+─────────────────────────────────────────────────────────+
|               Stabilisasi Output (Anti-Flicker)         |
|      - Confidence Threshold (>= 60% / >= 85%)           |
|      - Majority Voting Buffer (5 frame)                 |
+─────────────────────────────────────────────────────────+
                            │
                            ▼
+---------------------------------------------------------+
|            UI Rendering (Teks di Layar HP)              |
+---------------------------------------------------------+
```

---

## 3. Matematika & Algoritma Normalisasi Data

Guna menjamin model invarian terhadap jarak tangan ke kamera (*scale invariance*) dan posisi tangan di layar (*translation invariance*), setiap frame koordinat wajib melalui tahapan normalisasi kanonikal berikut sebelum diumpankan ke model:

### 3.1 Translasi (Origin Invariance)
Memindahkan titik acuan (origin $(0,0,0)$) ke titik pergelangan tangan (Wrist Landmark $P_0$):
$$P'_i = P_i - P_0, \quad \forall i \in \{0, 1, 2, \dots, 20\}$$
Di mana $P_i = [x_i, y_i, z_i]^T$.

### 3.2 Penskalaan (Scale Invariance)
Menghitung jarak Euclidean maksimum dari titik origin ke titik landmark terjauh:
$$d_{\max} = \max_{i=0}^{20} \left( \sqrt{(x'_i)^2 + (y'_i)^2 + (z'_i)^2} \right)$$

Lalu menormalkan seluruh koordinat:
$$P_{\text{final}, i} = \frac{P'_i}{d_{\max}}, \quad \forall i \in \{0, 1, 2, \dots, 20\}$$

### 3.3 Perataan Vektor (Flattening)
Hasil akhir di-flatten menjadi vektor 1D berukuran $63$ elemen float:
$$\mathbf{V} = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$$

---

## 4. Rincian Arsitektur Model Machine Learning

### 4.1 Modul Huruf Statis: Multi-Layer Perceptron (MLP)
* **Tujuan**: Klasifikasi instan bentuk tangan 24 huruf statis SIBI.
* **Input Dimension**: Matriks $(1, 63)$ bertipe float32.
* **Output**: 24 probabilitas kelas (Softmax).
* **Struktur Layer**:
  ```python
  model = Sequential([
      Input(shape=(63,)),
      Dense(128, activation='relu'),
      Dropout(0.3),
      Dense(64, activation='relu'),
      Dropout(0.3),
      Dense(24, activation='softmax')
  ])
  ```
* **Karakteristik & Status**:
  * Latensi inferensi $< 3\text{ ms}$ di smartphone Android.
  * Ukuran model `.tflite`: **206,82 KB** (sudah diekspor ke `model/talky_mlp.tflite`).

### 4.2 Modul Huruf J & Z: Heuristic Movement Analyzer (`gerak_jz.py`)
* **Tujuan**: Menghindari galat klasifikasi pada huruf J dan Z yang membutuhkan komponen gerak.
* **Mekanisme**:
  * Memonitor pergerakan landmark ujung jari dalam buffer 15–30 frame.
  * **Huruf J**: Memvalidasi jari kelingking tegak, diikuti perubahan arah trajektori membentuk kurva lengkung ke bawah lalu naik ke atas ($\cap$-shaped loop).
  * **Huruf Z**: Memvalidasi arah hadap telunjuk, diikuti trajektori zig-zag: horizontal kanan $\rightarrow$ diagonal kiri-bawah $\rightarrow$ horizontal kanan.

### 4.3 Modul Kata Sapaan: 1D-CNN (Temporal Convolutional Network - Fase 2)
* **Tujuan**: Memahami pola gestur dinamis kata sapaan dalam rentang waktu.
* **Input Dimension**: Matriks sekuens $(30, 63)$ (30 frame berturut-turut setara $\approx 1$ detik rekaman).
* **Mengapa 1D-CNN, bukan 2D-CNN?**
  1. Mengabaikan piksel visual (kebal gangguan latar belakang, warna kulit, dan pencahayaan).
  2. Kernel konvolusi 1D meluncur sepanjang sumbu waktu (*time-step*), menangkap pola akselerasi, orientasi, dan kecepatan gerak sendi tangan.
  3. Sangat hemat daya baterai smartphone.
* **Topologi Model**:
  ```python
  model = Sequential([
      Input(shape=(30, 63)),
      Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
      BatchNormalization(),
      MaxPooling1D(pool_size=2),
      Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
      BatchNormalization(),
      MaxPooling1D(pool_size=2),
      GlobalAveragePooling1D(),
      Dense(64, activation='relu'),
      Dropout(0.3),
      Dense(NUM_CLASSES_KATA, activation='softmax')
  ])
  ```
* **Ukuran Model .tflite**: $\approx 450\text{ KB}$.

---

## 5. Strategi Deployment & Rekayasa Performa Mobile (Flutter)

1. **State Management Ring Buffer (FIFO)**:
   Pada Flutter (Dart), streaming frame untuk modul kata dialokasikan menggunakan array circular statis berukuran tetap $30 \times 63$. Mengalokasikan array baru di setiap frame dihindari guna mencegah beban *Garbage Collector (GC Thrashing)*.
2. **Pemisahan Mode di UI (*Explicit State Toggle*)**:
   Aplikasi menyediakan tombol beralih mode:
   * **Mode Ejaan (Spelling)**: Menjalankan pipeline MLP + Heuristic J/Z.
   * **Mode Sapaan (Phrase)**: Menjalankan pipeline buffer 30 frame + 1D-CNN.
   * *Pemisahan ini memastikan kedua inferensi tidak berjalan bersamaan sehingga CPU load tetap di bawah 15%.*
3. **Inference Stabilization (Anti-Flicker)**:
   Prediksi hanya dinyatakan valid dan dimunculkan ke antarmuka pengguna apabila probabilitas Softmax memenuhi ambang batas:
   $$\text{Confidence Threshold} \ge 0.60 \text{ (Huruf)} \quad \text{atau} \quad \ge 0.85 \text{ (Kata Sapaan)}$$
   Disertai penstabil *majority voting* dari 5 frame terakhir.

---

## 6. Struktur Direktori Repositori TALKY

```text
talky-ml/
├── data/
│   ├── dataset.csv             # 9.600 baris dataset utama 4 orang (A-Y)
│   └── dataset_luar_sibi1.csv  # 2.372 baris ekstraksi sibi_1 (Kaggle)
├── model/
│   ├── talky_mlp.keras         # Model Keras terlatih
│   ├── talky_mlp.tflite        # Model TFLite untuk Flutter (206,8 KB)
│   └── labels.txt              # Daftar 24 label kelas huruf
├── hand_landmarker.task        # Model MediaPipe Vision
├── talky_common.py             # create_landmarker, detect, normalize
├── 01_collect_data.py          # Skrip perekaman dataset
├── 02_train_mlp.ipynb          # Notebook training & evaluasi MLP
├── 03_export_tflite.py         # Skrip ekspor & verifikasi TFLite
├── gerak_jz.py                 # Rule-based detector huruf J & Z
├── PANDUAN_FLUTTER_SHEVA.md    # Panduan implementasi Dart untuk Flutter
├── projek.md                   # Dokumen acuan tunggal terpadu
└── PROJECT_ARCHITECTURE.md     # Dokumen blueprint arsitektur ini
```

---

## 7. Rujukan Ilmiah (Academic References)

1. Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv preprint arXiv:1906.08172.
2. Sugiantari, dkk. (2025). *Klasifikasi Bahasa Isyarat SIBI Menggunakan Algoritma Multilayer Perceptron dan Ekstraksi Fitur Landmark MediaPipe*.
3. Kementerian Pendidikan dan Kebudayaan RI. *Kamus Sistem Isyarat Bahasa Indonesia (SIBI)*.
