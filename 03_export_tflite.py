"""
03_export_tflite.py
Mengekspor model Keras (model/talky_mlp.keras) ke format TensorFlow Lite (model/talky_mlp.tflite)
untuk digunakan pada aplikasi mobile Flutter (tflite_flutter).

Alur:
1. Validasi keberadaan file model/talky_mlp.keras dan model/labels.txt
2. Konversi ke TensorFlow Lite (float32 standar)
3. Simpan file model/talky_mlp.tflite
4. Verifikasi inferensi Keras vs TFLite Interpreter
5. Tampilkan spesifikasi tensor (input/output) untuk referensi developer Flutter
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# Path penting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
KERAS_PATH = os.path.join(MODEL_DIR, "talky_mlp.keras")
TFLITE_PATH = os.path.join(MODEL_DIR, "talky_mlp.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

# 63 nama fitur landmark
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]


def export_tflite():
    print("=" * 60)
    print("TALKY - EXPORT MODEL KE TENSORFLOW LITE (.tflite)")
    print("=" * 60)

    # 1. Validasi file Keras
    if not os.path.exists(KERAS_PATH):
        print(f"[ERROR] Model Keras tidak ditemukan di: {KERAS_PATH}")
        print("Silakan jalankan notebook '02_train_mlp.ipynb' terlebih dahulu.")
        sys.exit(1)

    print(f"1. Memuat model Keras dari: {KERAS_PATH}")
    model = tf.keras.models.load_model(KERAS_PATH)
    model.summary()

    # Baca label kelas jika ada
    labels = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"   Label kelas ({len(labels)} huruf): {', '.join(labels)}")
    else:
        print("   [PERINGATAN] labels.txt tidak ditemukan di folder model/")

    # 2. Konversi ke TFLite
    print("\n2. Mengonversi ke TensorFlow Lite (float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimasi default (opsional, tetap float32 presisi penuh)
    tflite_model = converter.convert()

    # 3. Simpan berkas .tflite
    print(f"\n3. Menyimpan model TFLite ke: {TFLITE_PATH}")
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"   Ukuran berkas TFLite: {size_kb:.2f} KB ({len(tflite_model):,} bytes)")

    # 4. Verifikasi dengan TFLite Interpreter
    print("\n4. Memverifikasi model TFLite dengan TFLite Interpreter...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_shape = input_details[0]["shape"].tolist()
    in_type = np.dtype(input_details[0]["dtype"]).name
    out_shape = output_details[0]["shape"].tolist()
    out_type = np.dtype(output_details[0]["dtype"]).name

    print("   [Input Tensor]")
    print(f"   - Name : {input_details[0]['name']}")
    print(f"   - Shape: {in_shape} (batch_size, 63 fitur)")
    print(f"   - Type : {in_type}")
    print("   [Output Tensor]")
    print(f"   - Name : {output_details[0]['name']}")
    print(f"   - Shape: {out_shape} (batch_size, {len(labels) if labels else 'num_classes'} kelas probabilitas)")
    print(f"   - Type : {out_type}")

    # 5. Uji coba inferensi: bandingkan Keras vs TFLite
    print("\n5. Uji konsistensi prediksi (Keras vs TFLite)...")
    test_sample = None
    sample_label = "Uji Coba"

    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH, nrows=5)
            if all(col in df.columns for col in FEATURE_COLUMNS):
                row = df.iloc[0]
                test_sample = row[FEATURE_COLUMNS].values.astype(np.float32).reshape(1, 63)
                sample_label = f"Baris 0 dataset (Huruf '{row.get('label', '?')}' oleh '{row.get('person_id', '?')}')"
        except Exception as e:
            print(f"   (Catatan: Gagal membaca sampel CSV: {e})")

    if test_sample is None:
        # Gunakan vektor acak dummy jika CSV tidak dapat dibaca
        test_sample = np.random.randn(1, 63).astype(np.float32)
        sample_label = "Vektor dummy (63 float)"

    print(f"   Menggunakan sampel: {sample_label}")

    # Prediksi via Keras
    keras_pred = model.predict(test_sample, verbose=0)[0]
    keras_idx = int(np.argmax(keras_pred))
    keras_conf = float(keras_pred[keras_idx])
    keras_huruf = labels[keras_idx] if labels and keras_idx < len(labels) else str(keras_idx)

    # Prediksi via TFLite
    interpreter.set_tensor(input_details[0]["index"], test_sample)
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(output_details[0]["index"])[0]
    tflite_idx = int(np.argmax(tflite_pred))
    tflite_conf = float(tflite_pred[tflite_idx])
    tflite_huruf = labels[tflite_idx] if labels and tflite_idx < len(labels) else str(tflite_idx)

    # Hitung selisih maksimal antar probabilitas
    max_diff = float(np.max(np.abs(keras_pred - tflite_pred)))

    print(f"   - Keras  : Prediksi '{keras_huruf}' (indeks {keras_idx}) | Confidence = {keras_conf*100:.2f}%")
    print(f"   - TFLite : Prediksi '{tflite_huruf}' (indeks {tflite_idx}) | Confidence = {tflite_conf*100:.2f}%")
    print(f"   - Selisih maksimum probabilitas: {max_diff:.8f}")

    if keras_idx == tflite_idx and max_diff < 1e-4:
        print("\n[SUKSES] Konversi berhasil 100%! Prediksi Keras dan TFLite identik.")
    else:
        print("\n[PERINGATAN] Terdapat perbedaan nilai, periksa kembali arsitektur model.")

    print("=" * 60)
    print("BERKAS SIAP DIBERIKAN KE DEVELOPER FLUTTER (SHEVA):")
    print(f"1. Model TFLite : {os.path.relpath(TFLITE_PATH, BASE_DIR)}")
    print(f"2. Label Huruf  : {os.path.relpath(LABELS_PATH, BASE_DIR)}")
    print(f"3. MediaPipe Task: hand_landmarker.task")
    print("=" * 60)


if __name__ == "__main__":
    export_tflite()
