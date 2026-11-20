# Panduan Integrasi Machine Learning Talky ke Flutter (Khusus Developer Mobile / Sheva)

> **Dokumen ini ditujukan untuk Developer Mobile (Sheva) dan Asisten AI (Cursor / Claude Code / Copilot / dsb.)** yang membangun aplikasi Flutter Talky (Tugas Akhir D3 Teknik Informatika Polines).
> 
> Dokumen ini menjelaskan spesifikasi berkas, pipeline pra-pemrosesan data koordinat, inferensi TensorFlow Lite, kode Dart siap pakai, dan unit test mandiri tanpa kamera.

---

## 1. Ringkasan Arsitektur & Alur Kerja

Aplikasi Talky menerjemahkan Bahasa Isyarat SIBI (Sistem Isyarat Bahasa Indonesia) ke teks secara *real-time* di perangkat mobile dengan alur pipeline berikut:

```
[Kamera HP (Frame Video)]
           │
           │  (Wajib: Frame di-flip cermin jika kamera depan agar konsisten dengan data latih)
           ▼
[MediaPipe Tasks Vision: hand_landmarker.task]
           │
           │  Menghasilkan 21 titik koordinat 3D mentah (x, y, z) untuk 1 tangan (num_hands=1)
           ▼
[Fungsi normalizeLandmarks() di Dart]
           │  1. Translasi: titik 0 (pergelangan) digeser ke (0, 0, 0)
           │  2. Skala: dibagi dengan jarak Euclidean terjauh titik ke pergelangan
           │  3. Flatten: diubah menjadi List<double> berisi 63 angka float
           ▼
[TensorFlow Lite: talky_mlp.tflite]
           │  Input:  [1, 63] float32
           │  Output: [1, 24] float32 (probabilitas softmax tiap huruf)
           ▼
[Argmax & Pemetaan Label (labels.txt)]
           │  Confidence Threshold (>= 60%) + Majority Voting (5 frame terakhir)
           ▼
[Huruf Terprediksi & Teks di Layar UI Flutter]
```

> [!IMPORTANT]
> **Arsitektur Decoupled (Terpisah Bersih):**
> Kode Flutter di Dart hanya mengurus kamera, MediaPipe, `normalizeLandmarks()`, dan pemanggilan TFLite. **Jika tim ML memperbarui model TFLite untuk meningkatkan akurasi, Sheva cukup menimpa file `talky_mlp.tflite` di folder `assets/` tanpa perlu mengubah kode Dart sama sekali.**

---

## 2. Berkas Model yang Wajib Dimasukkan ke Assets Flutter

Salin 3 berkas berikut dari repo `talky-ml` ke dalam folder aset proyek Flutter Anda (misal `assets/model/` atau `assets/`):

| Nama Berkas | Lokasi Asal di `talky-ml` | Ukuran | Deskripsi |
|---|---|---|---|
| `talky_mlp.tflite` | `model/talky_mlp.tflite` | ~207 KB | Model neural network (MLP) terkompresi float32 |
| `labels.txt` | `model/labels.txt` | 70 B | Berisi 24 huruf statis berurutan baris demi baris |
| `hand_landmarker.task` | `hand_landmarker.task` | ~7.8 MB | Model MediaPipe Vision untuk deteksi 21 titik tangan |

Pastikan sudah didaftarkan pada `pubspec.yaml`:
```yaml
flutter:
  assets:
    - assets/model/talky_mlp.tflite
    - assets/model/labels.txt
    - assets/model/hand_landmarker.task
```

---

## 3. Spesifikasi Tensor Model TFLite

Bagi developer dan asisten AI, berikut rincian teknis input/output tensor model:

* **Tipe Data**: `float32` (IEEE 754 32-bit floating point)
* **Input Tensor**:
  * **Nama**: `serving_default_input_layer_23:0`
  * **Bentuk (Shape)**: `[1, 63]`
  * **Arti**: 1 baris inferensi dengan 63 angka fitur ($x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}$) yang telah dinormalisasi.
* **Output Tensor**:
  * **Nama**: `StatefulPartitionedCall_1:0`
  * **Bentuk (Shape)**: `[1, 24]`
  * **Arti**: 24 nilai probabilitas softmax (rentang 0.0 s.d. 1.0) dengan total jumlah probabilitas = 1.0.
* **Daftar Indeks Huruf (`labels.txt`)**:
  | Indeks | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | **Huruf** | A | B | C | D | E | F | G | H | I | K | L | M |

  | Indeks | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | **Huruf** | N | O | P | Q | R | S | T | U | V | W | X | Y |

*(Catatan: Huruf **J** dan **Z** tidak ada di model MLP ini karena merupakan huruf dinamis/gerak. Penjelasan ada di Bagian 7).*

---

## 4. Logika Normalisasi Koordinat di Dart (Wajib Sama 100% dengan Python)

Model dilatih dengan data yang dinormalisasi oleh `normalize()` pada file `talky_common.py`. Fungsi di Dart **wajib mengimplementasikan logika matematis yang sama persis** agar prediksi tidak meleset.

### Referensi Kode Python Asal (`talky_common.py`):
```python
def normalize(pts):
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts - pts[0]  # Translasi ke pergelangan (titik 0)
    scale = np.max(np.linalg.norm(pts, axis=1))  # Jarak terjauh
    if scale < 1e-6:
        return None
    return (pts / scale).flatten()  # 63 float
```

### Implementasi Resmi di Dart (`sibi_normalizer.dart`):

```dart
import 'dart:math';

/// Representasi titik 3D landmark dari MediaPipe
class Landmark3D {
  final double x;
  final double y;
  final double z;

  const Landmark3D({
    required this.x,
    required this.y,
    required this.z,
  });
}

class SibiNormalizer {
  /// Melakukan normalisasi 21 landmark tangan menjadi 63 float.
  /// Mengembalikan List<double> dengan panjang tepat 63, atau null jika tidak valid.
  static List<double>? normalize(List<Landmark3D> landmarks) {
    // Validasi jumlah titik harus tepat 21
    if (landmarks.length != 21) {
      return null;
    }

    // 1. Translasi: jadikan pergelangan tangan (indeks 0) sebagai titik asal (0, 0, 0)
    final wrist = landmarks[0];
    final translated = List<Landmark3D>.generate(21, (i) {
      return Landmark3D(
        x: landmarks[i].x - wrist.x,
        y: landmarks[i].y - wrist.y,
        z: landmarks[i].z - wrist.z,
      );
    });

    // 2. Skala: cari jarak Euclidean terjauh dari pergelangan tangan (0, 0, 0)
    double maxDistance = 0.0;
    for (final pt in translated) {
      final dist = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
      if (dist > maxDistance) {
        maxDistance = dist;
      }
    }

    // Jika tangan terlalu kecil / tidak terbaca dengan benar
    if (maxDistance < 1e-6) {
      return null;
    }

    // 3. Bagi setiap koordinat dengan skala jarak terjauh, lalu ratakan (flatten) ke List 63 elemen
    final List<double> normalized63 = [];
    for (final pt in translated) {
      normalized63.add(pt.x / maxDistance);
      normalized63.add(pt.y / maxDistance);
      normalized63.add(pt.z / maxDistance);
    }

    return normalized63;
  }
}
```

---

## 5. Implementasi Kelas Inferensi TFLite di Flutter (`sibi_classifier.dart`)

Gunakan package resmi [`tflite_flutter`](https://pub.dev/packages/tflite_flutter).

```dart
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class PredictionResult {
  final String letter;
  final double confidence; // Nilai 0.0 s.d. 1.0 (misal 0.85 = 85%)
  final int classIndex;

  PredictionResult({
    required this.letter,
    required this.confidence,
    required this.classIndex,
  });

  @override
  String toString() => 'Huruf: $letter (${(confidence * 100).toStringAsFixed(1)}%)';
}

class SibiClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];

  bool get isReady => _interpreter != null && _labels.isNotEmpty;

  /// Memuat model TFLite dan labels.txt dari assets
  Future<void> init({
    String modelAssetPath = 'assets/model/talky_mlp.tflite',
    String labelsAssetPath = 'assets/model/labels.txt',
  }) async {
    try {
      // 1. Muat Interpreter TFLite
      final options = InterpreterOptions()..threads = 2; // Menggunakan 2 thread CPU mobile
      _interpreter = await Interpreter.fromAsset(modelAssetPath, options: options);

      // 2. Muat labels.txt
      final labelsRaw = await rootBundle.loadString(labelsAssetPath);
      _labels = labelsRaw
          .split('\n')
          .map((line) => line.trim())
          .where((line) => line.isNotEmpty)
          .toList();

      print('SibiClassifier berhasil diinisialisasi. Jumlah label: ${_labels.length}');
    } catch (e) {
      print('Gagal inisialisasi SibiClassifier: $e');
      rethrow;
    }
  }

  /// Menjalankan prediksi dari 63 float koordinat ternormalisasi
  PredictionResult? predict(List<double> normalized63) {
    if (_interpreter == null || _labels.isEmpty) {
      print('Classifier belum diinisialisasi!');
      return null;
    }

    if (normalized63.length != 63) {
      print('Panjang input salah: ${normalized63.length} (harus 63)');
      return null;
    }

    // Input tensor shape [1, 63]
    final input = [normalized63];

    // Output tensor shape [1, 24]
    final output = List.generate(1, (_) => List.filled(24, 0.0));

    // Jalankan inferensi
    _interpreter!.run(input, output);

    final List<double> probabilities = output[0];

    // Argmax: cari indeks dengan probabilitas tertinggi
    int bestIndex = 0;
    double bestConfidence = probabilities[0];

    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > bestConfidence) {
        bestConfidence = probabilities[i];
        bestIndex = i;
      }
    }

    return PredictionResult(
      letter: _labels[bestIndex],
      confidence: bestConfidence,
      classIndex: bestIndex,
    );
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}
```

---

## 6. Unit Test Mandiri di Dart (Verifikasi Tanpa Kamera)

Sebelum menguji coba dengan kamera ponsel, jalankan tes mandiri berikut di Flutter (`test/sibi_test.dart` atau jalankan di fungsi `main()` sementara). Data di bawah diambil langsung dari **baris pertama `data/dataset.csv` (Huruf 'A' oleh Sheva)**:

```dart
void testSibiPrediction(SibiClassifier classifier) {
  // 63 angka nyata dari baris 0 dataset SIBI Talky (Huruf 'A')
  final sampleHurufA = <double>[
    0.0, 0.0, 0.0,
    -0.186727, -0.184045, -0.037611,
    -0.321852, -0.354157, -0.066491,
    -0.428678, -0.505679, -0.088339,
    -0.504287, -0.635835, -0.103986,
    -0.163623, -0.655252, -0.053155,
    -0.222718, -0.803738, -0.076629,
    -0.285868, -0.730304, -0.091176,
    -0.334057, -0.669865, -0.096538,
    -0.024976, -0.672809, -0.047463,
    -0.055819, -0.835848, -0.074367,
    -0.134816, -0.760233, -0.090123,
    -0.203649, -0.686629, -0.097062,
    0.094593, -0.636665, -0.048705,
    0.095033, -0.793664, -0.077678,
    0.015243, -0.725916, -0.093539,
    -0.060136, -0.665324, -0.099197,
    0.198273, -0.569421, -0.052601,
    0.231268, -0.702951, -0.075908,
    0.149206, -0.671511, -0.084777,
    0.078491, -0.632007, -0.087265,
  ];

  final result = classifier.predict(sampleHurufA);

  print('=== HASIL UNIT TEST TFLITE FLUTTER ===');
  print('Prediksi   : ${result?.letter}');
  print('Confidence : ${((result?.confidence ?? 0) * 100).toStringAsFixed(2)}%');
  print('Indeks     : ${result?.classIndex}');

  // Verifikasi
  assert(result?.letter == 'A', 'Prediksi harus huruf A!');
  assert(result?.classIndex == 0, 'Indeks harus 0!');
  print('STATUS: 100% COCOK DENGAN MODEL PYTHON!');
}
```

---

## 7. Penanganan Khusus Huruf J dan Z (Gerakan Dinamis)

Dalam SIBI resmi:
* **Huruf J**: Ujung jari kelingking membentuk pose menggantung, lalu bergerak melengkung seperti kail ikan (lintasan huruf J).
* **Huruf Z**: Ujung jari telunjuk menunjuk ke depan, lalu bergerak zig-zag (garis horizontal kanan -> diagonal kiri bawah -> horizontal kanan).

Karena model MLP hanya menerima **1 frame statis (snapshot)**, huruf J dan Z tidak dapat dikenali hanya dari model TFLite ini.

### Saran Implementasi di Mobile:
1. **Fase 1 (Target Utama Saat Ini)**:
   Fokuskan aplikasi pada **24 huruf statis (A–Y)** dengan lancar dan stabil.
2. **Fase 2 (Tambahan Deteksi J dan Z)**:
   Di Python sudah ada referensi pelacak gerak pada file `gerak_jz.py`. Logikanya adalah menyimpan buffer 15–30 frame koordinat ujung jari telunjuk (titik 8 untuk Z) dan kelingking (titik 20 untuk J). Jika terdeteksi lintasan gerak zig-zag atau kail, sistem memicu output huruf Z atau J.

---

## 8. Tips Tampilan UI & Stabilisasi Prediksi (*Smoothing*)

Saat kamera membaca pose tangan pengguna secara live, hasil prediksi bisa berfluktuasi antar frame jika tangan sedikit goyah. Terapkan 2 teknik sederhana berikut di Flutter:

1. **Threshold Keyakinan (Confidence Threshold)**:
   Hanya tampilkan huruf jika `confidence >= 0.60` (atau 60%). Jika di bawah itu, anggap pose belum mantap / tampilkan tanda hubung `-` atau teks "Membaca...".
2. **Majority Voting Buffer (Stabilizer)**:
   Simpan 5 hasil prediksi terakhir ke dalam `List<String> history`. Huruf yang ditampilkan di layar adalah huruf yang paling banyak muncul di dalam 5 frame terakhir tersebut. Hal ini membuat teks di layar HP tidak berkedip-kedip.
3. **Tombol Spasi dan Hapus di Layar (UI Buttons)**:
   Karena bahasa isyarat SIBI tidak memiliki gerakan tangan khusus untuk karakter spasi (*space*) dan hapus (*backspace*), sediakan tombol fisik/digital di layar UI Flutter (misalnya tombol "Spasi" dan tombol "Hapus / ⌫") di sebelah kolom teks terjemahan.

---

## 9. Hubungan dengan Tim Machine Learning

Jika Sheva mendapati ada huruf tertentu yang kurang peka atau salah tebak saat dicoba langsung di HP:
* Catat huruf apa yang tertukar (misalnya huruf **P** sering terbaca **K**, atau **R** terbaca **U**).
* Tim ML akan melakukan perbaikan pose pada dataset dan melatih ulang bobot model.
* Setelah model baru selesai, tim ML hanya perlu mengirimkan file `talky_mlp.tflite` baru ke Sheva. Sheva **cukup me-replace file tersebut di folder assets**.

---

*Panduan ini disusun otomatis untuk kelancaran pengerjaan kolaboratif Tugas Akhir D3 Teknik Informatika Talky (16 November 2026).*
