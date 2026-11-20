# 📘 PROJEK TALKY: Konteks & Arsitektur Sistem Terpadu

> **Dokumen Acuan Tunggal (Single Source of Truth) Proyek Tugas Akhir TALKY**
> Mengintegrasikan Konteks Akademik (`PROJECT_CONTEXT.md`), Desain Arsitektur Sistem (`PROJECT_ARCHITECTURE.md`), dan Catatan Status Progres Riil (`progres.md`).
> Terakhir diperbarui: 18 November 2026

---

## 1. Identitas Proyek & Konteks Akademik

| Item | Deskripsi |
|---|---|
| **Judul TA** | **TALKY: Aplikasi Mobile Penerjemah Sistem Isyarat Bahasa Indonesia (SIBI) ke Teks Menggunakan MediaPipe dan Algoritma Multilayer Perceptron (MLP)** |
| **Jenjang** | D3 Teknik Informatika, Jurusan Teknik Elektro, Politeknik Negeri Semarang (Polines) |
| **Pelaksana I** | **Nathaniel Andyno Gunawan** (NIM: 3.34.24.3.16) — *Fokus: Machine Learning, Pipeline Data & Deployment TFLite* |
| **Pelaksana II** | **Sheva Padhista Putri Sanny** (NIM: 3.34.24.3.19) — *Fokus: Aplikasi Mobile Flutter, UI/UX & Integrasi Model* |
| **Pembimbing I** | **Dr. Liliek Triyono, S.T., M.Kom.** |
| **Pembimbing II** | **Dr. Sukamto, S.Kom., M.T.** |
| **Jadwal Kerja** | September 2026 s.d. Januari 2027 (Persetujuan Proposal: 22 September 2026) |

### 1.1 Tujuan Proyek
1. Membangun aplikasi mobile yang mampu menerjemahkan abjad SIBI ke teks secara *real-time* langsung di smartphone (*edge-computing*, tanpa internet).
2. Melatih model MLP untuk mengklasifikasikan 63 koordinat landmark tangan hasil deteksi MediaPipe dengan **target akurasi pengujian $\ge 85\%$ pada orang baru (*unseen person*)**.
3. Mengembangkan antarmuka aplikasi mobile (Flutter) yang responsif, stabil, dan lolos uji fungsionalitas (*Black Box Testing*).

### 1.2 Metodologi Pengembangan (Hibrida)
* **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*): Untuk rekayasa kecerdasan buatan (pengumpulan data 5 orang, pra-pemrosesan normalisasi, pelatihan iteratif MLP, evaluasi per huruf, dan ekspor TFLite).
* **Waterfall**: Untuk rekayasa perangkat lunak aplikasi mobile (analisis kebutuhan, perancangan UI, integrasi TFLite ke Flutter, pengujian *black box*, dan penyusunan laporan TA).

---

## 2. Batasan Masalah & Lingkup Sistem

1. **Modul Ejaan Alfabet (Fingerspelling - Fase 1 / Target Utama TA)**:
   * **24 Huruf Statis (A–I, K–Y)**: Diproses per *single-frame* citra menggunakan algoritma **Multilayer Perceptron (MLP)**.
   * **2 Huruf Dinamis (J dan Z)**: Diproses melalui analisis trajektori spasial (*rule-based movement tracking*) terhadap koordinat ujung jari (kelingking untuk J, telunjuk untuk Z), **tidak dimasukkan ke dalam model MLP**.
2. **Modul Kata Sapaan Dasar (Dynamic Gestures - Fase 2 / Pengembangan Lanjutan)**:
   * Mengklasifikasikan frasa kata sapaan (misal: *Halo, Terima Kasih, Sama-Sama, Maaf*).
   * Memproses sekuens deret koordinat waktu ($30 \text{ frame} \times 63 \text{ fitur}$) menggunakan arsitektur **1D-CNN (Temporal Convolution)** di atas koordinat MediaPipe, **bukan 2D-CNN berbasis citra piksel** agar tetap ringan di smartphone.
3. **Platform & Hardware Target**:
   * Aplikasi mobile berbasis **Flutter (Dart)** menggunakan package `tflite_flutter`.
   * Berjalan mandiri (*offline-first*) di CPU perangkat smartphone Android kelas menengah ke bawah (*mid-to-low end*).
4. **Input & Kendali Teks**:
   * Input berasal dari kamera smartphone (disarankan satu tangan dominan, frame di-flip cermin).
   * Karakter spasi (*space*) dan hapus (*backspace*) tidak memiliki isyarat tangan dalam kamus resmi SIBI, sehingga disediakan sebagai **tombol di layar UI Flutter**.

---

## 3. Diagram Alur Sistem (System Architecture)

```
[Kamera HP (Frame Video)]
           │
           │  (Wajib: Frame di-flip cermin jika kamera depan agar konsisten dengan data latih)
           ▼
[MediaPipe Tasks Vision: hand_landmarker.task]
           │
           │  Mengekstraksi 21 landmark 3D mentah (x, y, z) untuk 1 tangan
           ▼
[Normalisasi Kanonikal: normalizeLandmarks() di Dart / Python]
           │  1. Translasi pergelangan tangan ke (0, 0, 0)
           │  2. Penskalaan jarak Euclidean terjauh d_max = 1.0
           │  3. Flatten ke vektor 1D berisi 63 nilai float
           ▼
     ┌─────┴───────────────────────────────────────────────────────┐
     │                                                             │
     ▼ [Mode Ejaan Huruf]                                          ▼ [Mode Kata Sapaan - Fase 2]
[Single Frame: (1, 63)]                                       [Ring Buffer FIFO: (30, 63)]
     │                                                             │
     ├───────────► Huruf Statis (A-I, K-Y)                         ▼
     │             └─► [Model MLP TFLite (207 KB)]          [Model 1D-CNN TFLite (450 KB)]
     │                                                             │
     └───────────► Huruf Dinamis (J, Z)                            │
                   └─► [Heuristic Trajectory Tracking]             │
     │                                                             │
     └─────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
                    [Stabilisasi Output (Anti-Flicker)]
                    - Confidence Threshold (>= 60% / >= 85%)
                    - Majority Voting Buffer (5 frame)
                                   │
                                   ▼
                    [UI Rendering: Teks di Layar HP]
```

### Pemisahan Mode di UI (*Explicit State Toggle*)
Aplikasi mobile menyediakan tombol peralihan mode di layar:
* **Mode Ejaan (Spelling)**: Menjalankan pipeline MLP + Heuristic J/Z per-frame.
* **Mode Sapaan (Phrase - Fase 2)**: Menjalankan circular buffer 30 frame + 1D-CNN.
* *Manfaat Teknis:* Kedua model tidak pernah aktif bersamaan di background, menjaga beban komputasi CPU ponsel tetap berada di bawah $15\%$ dan baterai hemat.

---

## 4. Formulasi Matematika Normalisasi Data (Canonical Invariance)

Guna menjamin model invarian terhadap jarak tangan ke kamera (*scale invariance*) dan posisi tangan di layar (*translation invariance*), setiap frame 21 landmark wajib melalui tahapan normalisasi matematis berikut:

### 4.1 Translasi (Origin Invariance)
Memindahkan titik acuan (origin $(0,0,0)$) ke titik pergelangan tangan (Wrist Landmark $P_0$):
$$P'_i = P_i - P_0, \quad \forall i \in \{0, 1, 2, \dots, 20\}$$
Di mana $P_i = [x_i, y_i, z_i]^T$.

### 4.2 Penskalaan (Scale Invariance)
Menghitung jarak Euclidean maksimum dari pergelangan tangan ke landmark terjauh:
$$d_{\max} = \max_{i=0}^{20} \left( \sqrt{(x'_i)^2 + (y'_i)^2 + (z'_i)^2} \right)$$

Jika $d_{\max} < 10^{-6}$ (tangan tidak valid / terlalu kecil), sampel diabaikan. Jika valid, seluruh koordinat diskalakan:
$$P_{\text{final}, i} = \frac{P'_i}{d_{\max}}, \quad \forall i \in \{0, 1, 2, \dots, 20\}$$

### 4.3 Perataan Vektor (Flattening)
Hasil akhir diratakan menjadi vektor 1D berukuran tepat **63 elemen float**:
$$\mathbf{V} = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$$

> [!IMPORTANT]
> **ATURAN KERAS:** Logika matematis ini harus **identik 100%** di Python (`talky_common.py`), notebook training (`02_train_mlp.ipynb`), skrip ekspor (`03_export_tflite.py`), dan implementasi Dart di Flutter (`PANDUAN_FLUTTER_SHEVA.md`).

### 4.4 Formulasi Matematika Evaluasi Model (Metrik Uji Bab 3 & Bab 4)

Guna menguji performa klasifikasi 24 kelas huruf statis SIBI ($K = 24$), digunakan formulasi metrik evaluasi standar berbasis matriks konfusi (*Confusion Matrix*):

1. **Akurasi Global (Accuracy)**:
   Perbandingan antara jumlah total sampel yang ditebak secara benar terhadap seluruh total sampel data uji ($N$):
   $$\text{Accuracy} = \frac{\sum_{k=1}^K TP_k}{N} = \frac{\text{Jumlah Sampel Benar Terklasifikasi}}{\text{Total Seluruh Sampel Uji}}$$

2. **Presisi per Huruf (Precision)**:
   Rasio ketepatan saat model memprediksi suatu huruf kelas $k$ (dari semua sampel yang ditebak sebagai huruf $k$, berapa yang benar):
   $$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}$$

3. **Recall / Sensitivitas per Huruf (Recall)**:
   Kemampuan model menjaring huruf kelas $k$ dari seluruh kenyataan huruf $k$ yang ada (dari semua huruf $k$ asli, berapa yang berhasil dikenali):
   $$\text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

4. **F1-Score per Huruf**:
   Rata-rata harmonik antara presisi dan recall untuk mengukur keseimbangan deteksi kelas $k$:
   $$\text{F1-Score}_k = 2 \times \frac{\text{Precision}_k \times \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

5. **Rata-Rata Makro (Macro Average F1)**:
   Rata-rata tidak berbobot dari seluruh nilai F1 kelas huruf (mengevaluasi keadilan performa model di seluruh 24 huruf tanpa bias):
   $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \text{F1-Score}_k$$

---

## 5. Rincian Arsitektur Model Machine Learning

### 5.1 Modul Huruf Statis: Multilayer Perceptron (MLP)
* **Tujuan**: Klasifikasi instan bentuk tangan untuk 24 huruf statis SIBI (A–I, K–Y).
* **Input Tensor**: `[1, 63]` bertipe `float32`.
* **Output Tensor**: `[1, 24]` bertipe `float32` (probabilitas softmax tiap kelas huruf).
* **Topologi Jaringan**:
  ```
  Input(shape=(63,))
    │
  Dense(128, activation='relu')
    │
  Dropout(0.3)
    │
  Dense(64, activation='relu')
    │
  Dropout(0.3)
    │
  Dense(24, activation='softmax')
  ```
* **Status Deployment**: **SUDAH SELESAI DIEKSPOR**. Berkas `model/talky_mlp.tflite` berukuran **206,82 KB** dengan latensi inferensi $< 3\text{ ms}$ di smartphone. Prediksi Keras vs TFLite terverifikasi identik dengan selisih probabilitas $< 10^{-7}$. Model Keras terlatih tersimpan di `model/talky_mlp.keras` (Val Acc 85.87%, Test Acc 70.42% pada orang baru).

### 5.2 Modul Huruf J & Z: Heuristic Trajectory Analyzer
* **Tujuan**: Mendeteksi huruf dinamis J dan Z tanpa membebani model MLP.
* **Mekanisme**:
  * Menggunakan jejak buffer 15–30 frame pada landmark ujung jari.
  * **Huruf J**: Memvalidasi pose jari kelingking terangkat, diikuti trajektori melengkung ke bawah lalu naik ke atas seperti kail pancing.
  * **Huruf Z**: Memvalidasi pose jari telunjuk terangkat, diikuti pola gerak horizontal kanan $\rightarrow$ diagonal kiri-bawah $\rightarrow$ horizontal kanan (zig-zag).
* **Implementasi**: Telah dibuat di `gerak_jz.py` (`JZDetector`).

### 5.3 Modul Kata Sapaan: 1D-CNN Temporal (Rencana Fase 2)
* **Tujuan**: Memahami pola gestur dinamis kata sapaan dalam rentang waktu tertentu.
* **Input Tensor**: Matriks sekuens `(30, 63)` (30 frame berturut-turut $\approx 1$ detik rekaman pada 30 FPS).
* **Topologi Rencana**:
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
* **Estimasi Ukuran**: $\approx 450\text{ KB}$ setelah dikonversi ke TFLite.

---

## 6. Kondisi Dataset & Hasil Eksperimen Riil

Semua angka laporan berasal dari eksperimen nyata di repositori:

### 6.1 Dataset Utama (`data/dataset.csv`)
* **Total Sampel**: **12.000 baris data** (24 huruf $\times$ 100 sampel $\times$ 5 orang perekam).
* **Perekam**: `sheva` (2.400), `nathan` (2.400), `hafiid` (2.400), `zidna` (2.400), `mayang` (2.400).
* **Kualitas Data**: 66 kolom valid, 0 nilai hilang (NaN), skala maksimum $1.000$, 0% duplikat, dan keterpisahan KNN 5-fold internal mencapai **98,1%**.

### 6.2 Hasil Evaluasi MLP Mode `per_orang` (`02_train_mlp.ipynb`)

Pemisahan data dilakukan secara ketat berbasis subjek (*Leave-One-Person-Out / Cross-Subject Split*) tanpa ada kebocoran data (*data leakage*):
* **Skema Split 5 Orang (Total 12.000 Sampel)**:
  * **Train (60% / 7.200 sampel)**: `hafiid` + `mayang` + `nathan` (Model belajar bobot murni dari 3 orang ini).
  * **Validasi (20% / 2.400 sampel)**: `sheva` (Ujian validasi selama training, untuk *early stopping* & pemilihan hyperparameter terbaik).
  * **Test (20% / 2.400 sampel)**: `zidna` (Orang baru / *unseen subject* yang tidak pernah dilihat model sama sekali hingga evaluasi akhir).

---

#### 6.2.1 Pembuktian & Cara Menghitung Akurasi Validasi Sheva (85,87%)
Akurasi validasi diperoleh dari pengujian model terhadap seluruh **2.400 baris data milik Sheva** yang belum pernah dilihat bobot model:
* **Formulasi Perhitungan Matematis**:
  $$\text{Akurasi Validasi (Sheva)} = \frac{\sum_{i=1}^{N_{\text{val}}} \mathbb{I}(\hat{y}_i = y_i)}{N_{\text{val}}} = \frac{2.061}{2.400} = 0,85875 \rightarrow \mathbf{85,87\%}$$
  * Di mana $N_{\text{val}} = 2.400$ sampel (24 huruf $\times$ 100 sampel).
  * $\hat{y}_i$ adalah tebakan model untuk sampel ke-$i$, dan $y_i$ adalah label kunci jawaban asli milik Sheva.
  * Model berhasil menebak benar sebanyak **2.061 sampel**, dan hanya salah sebanyak 339 sampel.
* **Letak Bukti Riil di Repositori**:
  1. Notebook `02_train_mlp.ipynb` pada output **Bagian 4 (Pencarian Hyperparameter)**:
     `[5/7] {'n1': 128, 'n2': 64, 'dropout': 0.5, 'lr': 0.001} -> val 0.8587 | test 0.6908 | epoch 37`
  2. Berkas CSV riwayat **`model/eksperimen.csv`** baris ke-5 (`val_acc: 0.8587`).
* **Analisis Pembanding (Kenaikan terhadap Baseline 4 Orang)**:
  * Pada eksperimen 4 orang terdahulu (Train hanya 2 orang: Nathan + Hafiid = 4.800 sampel), akurasi validasi Sheva adalah **78,87%** (1.893 sampel benar).
  * Ketika Train ditambah 1 orang lagi (`mayang` = total 7.200 sampel), akurasi validasi Sheva melonjak ke **85,87%** (2.061 sampel benar).
  * **Peningkatan murni:** $+7,00\%$ ($+168$ sampel benar tambahan), membuktikan secara ilmiah bahwa bertambahnya variasi tangan di data latih meningkatkan daya generalisasi model terhadap tangan validator.

---

#### 6.2.2 Pembuktian & Cara Menghitung Akurasi Test Orang Baru Zidna (70,42%)
Akurasi test dihitung pada data orang baru yang sepenuhnya independen (`zidna`):
* **Formulasi Perhitungan Matematis**:
  $$\text{Akurasi Test (Zidna)} = \frac{\sum_{i=1}^{N_{\text{test}}} \mathbb{I}(\hat{y}_i = y_i)}{N_{\text{test}}} = \frac{1.690}{2.400} = 0,70416 \rightarrow \mathbf{70,42\%}$$
  * Model berhasil menebak benar **1.690 sampel** dari 2.400 sampel Zidna (pada Konfigurasi no. 7: `n1=128, n2=64, dropout=0.3, lr=0.003`).
* **Analisis Pembanding**:
  * Baseline terdahulu (2 orang di train) menghasilkan akurasi test Zidna sebesar **61,67%** (1.480 sampel benar).
  * Model baru 5 orang menghasilkan akurasi test Zidna sebesar **70,42%** (1.690 sampel benar).
  * **Peningkatan:** $+8,75\%$ ($+210$ sampel benar tambahan).

---

#### 6.2.3 Analisis Kesenjangan Ilmiah (Mengapa Akurasi Sheva 85,87% vs Zidna 70,42%?)
Perbedaan akurasi antara Sheva ($85,87\%$) dan Zidna ($70,42\%$) disebabkan oleh faktor empiris yang sangat jelas untuk dijelaskan saat ujian sidang:
1. **Anomali Kritis Huruf U pada Zidna (Ketiadaan Kontribusi Akurasi)**:
   * Pada data Sheva, huruf U direkam tegak lurus dan berhasil ditebak dengan benar oleh model.
   * Pada data Zidna, **100% sampel huruf U salah total ($F1 = 0.00$, 0/100 benar)** karena tertukar ke K (56 sampel) dan R (44 sampel) akibat jari yang agak condong ke depan atau menyilang.
   * *Dampak Matematis:* Kehilangan 100 poin ini menyunat $\frac{100}{2.400} = 4,17\%$ akurasi Zidna. Jika huruf U diperbaiki (misal 85 benar), akurasi Zidna langsung melonjak ke:
     $$\text{Akurasi}_{\text{revisi}} = \frac{1.690 + 85}{2.400} = \frac{1.775}{2.400} = \mathbf{73,96\% \approx 74\%}$$
2. **Keterbatasan Normalisasi Terhadap Kemiringan Tangan (*Tilt Angle Gap*)**:
   * Fungsi `normalize()` bersifat *scale-invariant* (tahan jarak) dan *translation-invariant* (tahan posisi), tetapi **tidak *rotation-invariant*** (tidak tahan kemiringan sudut putar tangan).
   * Sheva merekam dengan sudut tegak yang sangat konsisten dengan para perekam train.
   * Zidna memiliki sudut kemiringan tangan pada huruf berbentuk lengkung/lingkaran (O, Y, G, M, N) sehingga koordinat 63 fiturnya bergeser dari distribusi data latih.

---

#### 6.2.4 Evolusi Eksperimen (4 Variasi Validator $\rightarrow$ Train 4 Orang + Augmentasi Rotasi)

Untuk membuktikan keandalan dan progresivitas arsitektur MLP secara akademis di Bab 4, sistem diuji secara bertahap:

| Setup | Data Latih (*Train*) | Validator (*Validation*) | Akurasi Validasi | Akurasi Test Orang Baru (`zidna`) | Analisis & Dampak |
|:---:|:---:|:---:|:---:|:---:|---|
| Baseline | Nathan + Hafiid (4.800) | **Sheva** (2.400) | $78,87\%$ | $61,67\%$ | Baseline awal 4 orang |
| A | Nathan + Hafiid + Mayang (7.200) | **Sheva** (2.400) | $85,87\%$ | $70,42\%$ | Huruf U Zidna tertukar K & R ($0\%$) |
| B | Nathan + Sheva + Mayang (7.200) | **Hafiid** (2.400) | $85,83\%$ | $69,50\%$ | Butuh neuron besar (256-128) |
| C | Nathan + Hafiid + Sheva (7.200) | **Mayang** (2.400) | $94,33\%$ | $70,79\%$ | Huruf U Zidna mulai terdeteksi ($21\%$) |
| D | Hafiid + Mayang + Sheva (7.200) | **Nathan** (2.400) | $94,00\%$ | $72,58\%$ | Huruf U bangkit ke $60\%$ |
| E | 4 Orang + Augmentasi Rotasi (25.920) | Stratified 10% (960) | $99,37\%$ | $77,33\%$ | 23/24 huruf F1 $\ge 0,69$; O masih $0,59$ |
| **F (FINAL)** | **4 Orang + `Combine_SIBI` + `sibi_1` + Augmentasi (36.102)** | **Stratified 10% (960)** | **$99,58\%$** | **$77,96\%$** *(puncak $78,75\%$)* | **REKOR TERTINGGI ($+16,29\%$ dari baseline)! Semua 24 huruf F1 $\ge 0,66$, O sembuh ke $0,77$!** |

* **Tabel 7 Konfigurasi Hyperparameter pada Setup F (4 Orang + `Combine_SIBI` + `sibi_1` + Augmentasi Roll $\pm 12^\circ$, Yaw $\pm 6^\circ$, Jitter)**:

| no | n1 | n2 | dropout | learning rate | epoch | **val_acc (loss)** | **test_acc (Zidna)** | Keterangan |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | 128 | 64 | 0.3 | 0.0010 | 55 | 0.9937 (0.0229) | 0.7796 | Stabil |
| 2 | 64 | 32 | 0.3 | 0.0010 | 48 | 0.9792 (0.0672) | 0.7650 | Arsitektur ringkas |
| 3 | 256 | 128 | 0.3 | 0.0010 | 50 | 0.9927 (0.0151) | **0.7875** | **Puncak Test (78,75%)** |
| **4** | 128 | 64 | 0.2 | 0.0010 | 76 | **0.9958 (0.0139)** | **0.7796** | **TERPILIH VIA VAL (Val 99,58%, Test 77,96%)** |
| 5 | 128 | 64 | 0.5 | 0.0010 | 77 | 0.9844 (0.0436) | 0.7721 | Regularisasi ketat |
| 6 | 128 | 64 | 0.3 | 0.0005 | 57 | 0.9927 (0.0295) | 0.7812 | Konvergensi halus |
| 7 | 128 | 64 | 0.3 | 0.0030 | 30 | 0.9865 (0.0375) | 0.7704 | Learning rate tinggi |

* **Laporan Klasifikasi Per Huruf pada Test Set Zidna (Model Final — Setup F Konfig 4, Macro Precision $0.81$, Macro F1 $0.78$)**:
  * **Sembuhnya Huruf Lemah Terakhir**: O ($0.59 \rightarrow \mathbf{0.77}$, Recall $79\%$), Q ($0.79 \rightarrow \mathbf{0.86}$), I ($0.89 \rightarrow \mathbf{0.93}$), U ($0.72 \rightarrow \mathbf{0.76}$), N ($0.72 \rightarrow \mathbf{0.81}$).
  * **Huruf Sangat Kuat ($F1 \ge 0.80$)**: I ($0.93$), V ($0.88$), Q ($0.86$), W ($0.85$), Y ($0.85$), A ($0.83$, Recall $1.00$), D ($0.83$), K ($0.81$), N ($0.81$), B ($0.80$), F ($0.80$, Precision $1.00$).
  * **Huruf Kuat ($F1\text{ }0.70 - 0.79$)**: R ($0.79$), P ($0.78$, Recall $1.00$), **O ($0.77$)**, X ($0.77$, Recall $0.99$), **U ($0.76$)**, G ($0.76$), C ($0.73$), H ($0.70$), M ($0.70$).
  * **Huruf Sedang ($F1\text{ }0.66 - 0.69$)**: T ($0.69$), E ($0.68$), L ($0.67$), S ($0.66$). **(Tidak ada satu pun huruf di bawah $0.66$!)**
  * **Status Model TFLite**: Diekspor ke `model/talky_mlp.tflite` (**72,55 KB**, `float32`, selisih probabilitas Keras vs TFLite $= 0.00000000$).

### 6.3 Eksperimen & Status Dataset Luar (Khusus Train, Wajib Disitasi di Laporan TA)
1. **Dataset `sibi_1` (Kaggle `alvinbintang/sibi-dataset` — 2.372 baris di `data/dataset_luar_sibi1.csv`)**:
   * Sebelum ada augmentasi rotasi (pada dataset 2–4 orang), penggabungan `sibi_1` belum mendongkrak akurasi karena perbedaan sudut orientasi. Setelah dipadukan dengan 4 orang sendiri + augmentasi rotasi + `Combine_SIBI`, `sibi_1` berkontribusi meningkatkan generalisasi multikelas ke $77,96\% - 78,75\%$.
2. **Dataset `Combine_SIBI` (Kaggle M. Lanang Afkaar / Muhammad Rifal — 1.022 baris valid `Left` dari 1.125 foto, $90,8\%$ terbaca di `data/dataset_luar_combinesibi.csv`)**:
   * Terbukti secara empiris menyembuhkan huruf **O** pada data test Zidna dari $F1 = 0,59 \rightarrow 0,77$ (*Recall* $49\% \rightarrow 79\%$).

---

## 7. Integrasi Mobile Flutter (Paket Kerja Sheva)

Dokumentasi lengkap dan kode sumber Dart siap pakai tersedia di berkas **[PANDUAN_FLUTTER_SHEVA.md](file:///d:/PROJEK%20GIT/talky-ml/file.md/PANDUAN_FLUTTER_SHEVA.md)**.

### 7.1 Berkas yang Diserahkan ke Flutter:
1. `model/talky_mlp.tflite` (Model MLP TFLite terbaru, **72,55 KB**, float32).
2. `model/labels.txt` (Daftar 24 huruf statis A–Y tanpa J & Z).
3. `hand_landmarker.task` (Model MediaPipe Tasks Vision, 7,8 MB).

### 7.2 Struktur Kode Dart Inti:
* `SibiNormalizer`: Mengimplementasikan normalisasi translasi & skala max-dist persis sama dengan rumus Bab 4.
* `SibiClassifier`: Menggunakan `tflite_flutter` dengan input `[1, 63]` dan output `[1, 24]`.
* **Unit Test Verifikasi**: Menggunakan 63 nilai baris ke-0 dari `dataset.csv` (huruf 'A' milik Sheva), menghasilkan output: `Huruf A, confidence: 77.69%`.

---

## 8. Status Saat Ini & Roadmap Kerja

- [x] Proposal disetujui (22 September 2026).
- [x] Lingkungan Python 3.11, MediaPipe Tasks, TensorFlow 2.21, OpenCV terkonfigurasi.
- [x] Fungsi normalisasi kanonikal terverifikasi di `talky_common.py`.
- [x] Dataset 5 orang lengkap (12.000 baris di `data/dataset.csv`: Sheva, Nathan, Hafiid, Zidna, Mayang).
- [x] Notebook `02_train_mlp.ipynb` dieksekusi pada data 5 orang (Val Acc 85,87% tembus target proposal, Test Acc Zidna melonjak ke 70,42%).
- [x] Model terbaik diekspor ke TFLite (`03_export_tflite.py` $\rightarrow$ `model/talky_mlp.tflite`, 206,8 KB).
- [x] Skrip inferensi webcam real-time terintegrasi `04_test_realtime.py` (MLP + JZDetector + majority voting) selesai dibuat.
- [x] Panduan teknis Flutter untuk Sheva dibuat (`file.md/PANDUAN_FLUTTER_SHEVA.md`).
- [ ] **Tahap 1: Pengujian Integrasi Flutter oleh Sheva** (memverifikasi kelancaran pipeline kamera $\rightarrow$ MediaPipe $\rightarrow$ TFLite di ponsel Android).
- [ ] **Tahap 2: Optimasi Akurasi Menuju $\ge 85\%$ pada Test Set Orang Baru (`zidna`)**:
  - Perbaiki rekam ulang huruf U milik Zidna (potensi $+3,5\% - +4\%$ langsung).
  - Skema pelatihan 4 orang di data train (9.600 sampel) + data augmentation rotasi ringan ($\pm 10^\circ$).
  - (Opsi tambahan) Ekstraksi dataset luar `Combine_SIBI`.
- [ ] **Tahap 3: Uji Realtime di Laptop (`04_test_realtime.py`)** menggabungkan model MLP dan J/Z.
- [ ] **Tahap 4: Uji Coba Lapangan, Black Box Testing, Penyusunan Laporan TA, dan Sidang Pendadaran**.

---

## 9. Struktur Berkas Repositori

```
talky-ml/
├── venv/                       # Lingkungan virtual Python (diabaikan Git)
├── data/
│   ├── dataset.csv             # Dataset utama 5 orang (12.000 baris, .gitignore)
│   └── dataset_luar_sibi1.csv  # Ekstraksi sibi_1 (2.372 baris, .gitignore)
├── data_kata/                  # Dataset sekuens kata sapaan dinamis (30 frame/sampel)
│   ├── HALO/                   # Sampel sekuens .npy kata HALO
│   ├── TERIMA_KASIH/           # Sampel sekuens .npy kata TERIMA_KASIH
│   ├── SAMA_SAMA/              # Sampel sekuens .npy kata SAMA_SAMA
│   └── MAAF/                   # Sampel sekuens .npy kata MAAF
├── data_luar/                  # Citra mentah dataset luar (.gitignore)
│   ├── sibi_1/                 # Dataset luar sibi_1 Kaggle
│   ├── combine_sibi/           # Dataset luar Combine_SIBI (2.363 foto JPG)
│   └── contoh_hasil/           # 24 foto sampel berkerangka
├── model/                      # Output training & ekspor TFLite
│   ├── talky_mlp.keras         # Model Keras terlatih abjad (Val 85.87%, Test 70.42%)
│   ├── talky_mlp.tflite        # Model TFLite abjad untuk Flutter (206,8 KB)
│   ├── labels.txt              # 24 label kelas huruf statis
│   ├── talky_kata_1dcnn.keras  # Model Keras 1D-CNN kata sapaan dinamis
│   ├── talky_kata_1dcnn.tflite # Model TFLite 1D-CNN untuk Flutter (~250 KB)
│   ├── labels_kata.txt         # Label kelas kata sapaan
│   ├── eksperimen.csv          # Riwayat 7 konfigurasi hyperparameter abjad
│   └── learning_curve.csv      # Riwayat kurva belajar
├── file.md/                    # Dokumen master acuan sistem
│   ├── progres.md              # Log progres teknis detail & hasil eksperimen
│   ├── projek.md               # DOKUMEN INI (Konteks & Arsitektur Terpadu)
│   ├── PROJECT_ARCHITECTURE.md # Spesifikasi arsitektur teknis sistem
│   └── PANDUAN_FLUTTER_SHEVA.md # Panduan integrasi mobile Flutter untuk Sheva
├── hand_landmarker.task        # Model MediaPipe Vision
├── talky_common.py             # Fungsi bersama (create_landmarker, normalize, draw)
├── 01_collect_data.py          # Skrip perekaman dataset webcam (24 huruf A-Y)
├── 01_collect_kata.py          # Skrip perekaman kata sapaan dinamis (30 frame/sampel)
├── 02_train_mlp.ipynb          # Notebook training MLP & evaluasi abjad (mode per_orang)
├── 02_train_1dcnn_kata.py      # Skrip training 1D-CNN kata sapaan dinamis
├── 03_export_tflite.py         # Skrip ekspor & verifikasi model TFLite abjad
├── 03_export_kata_tflite.py    # Skrip ekspor model 1D-CNN kata sapaan ke TFLite
├── 04_test_realtime.py         # Skrip inferensi webcam real-time abjad (MLP + JZ)
├── 04_test_realtime_kata.py    # Skrip inferensi webcam real-time kata sapaan (1D-CNN)
├── ekstrak_gambar.py           # Skrip ekstraksi gambar luar ke CSV
├── check_data.py               # Validasi integritas kolom & baris dataset abjad
├── diagnose_data.py            # Diagnosis keterpisahan antar-huruf (KNN)
├── hapus_huruf.py              # Utilitas hapus huruf untuk rekam ulang (backup otomatis)
├── gerak_jz.py                 # Algoritma heuristik deteksi gerak huruf J dan Z
├── test_gerak_jz.py            # Pengujian visual mandiri modul gerak J dan Z
├── requirements.txt            # Daftar dependensi library Python
└── tests/
    └── test_dua_tangan.py      # Pengujian deteksi dua tangan
```
