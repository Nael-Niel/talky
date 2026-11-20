"""
03_export_kata_tflite.py
Mengonversi model 1D-CNN (Keras) ke format TFLite (Float32) untuk aplikasi mobile Flutter.

Input TFLite  : [1, 30, 63] (Float32)
Output TFLite : [1, NUM_CLASSES] (Float32)
Berkas target : model/talky_kata_1dcnn.tflite
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf

FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(FOLDER, "model")
KERAS_PATH = os.path.join(MODEL_DIR, "talky_kata_1dcnn.keras")
TFLITE_PATH = os.path.join(MODEL_DIR, "talky_kata_1dcnn.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels_kata.txt")


def main():
    print("=" * 60)
    print("TALKY - Ekspor Model 1D-CNN Kata Sapaan ke TFLite")
    print("=" * 60)

    if not os.path.exists(KERAS_PATH):
        print(f"Error: Model Keras {KERAS_PATH} belum ada.")
        print("Jalankan 'python 02_train_1dcnn_kata.py' terlebih dahulu.")
        sys.exit(1)

    print(f"1. Memuat model Keras: {KERAS_PATH}")
    model = tf.keras.models.load_model(KERAS_PATH)
    model.summary()

    print("\n2. Mengonversi ke TensorFlow Lite (Float32) ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Target float32 murni untuk kompatibilitas Flutter tflite_flutter
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    ukuran_kb = len(tflite_model) / 1024
    print(f"   Model TFLite tersimpan di: {TFLITE_PATH}")
    print(f"   Ukuran berkas: {ukuran_kb:.2f} KB")

    print("\n3. Memvalidasi model TFLite via Interpreter ...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_shape = input_details[0]["shape"]
    in_type = input_details[0]["dtype"]
    out_shape = output_details[0]["shape"]
    out_type = output_details[0]["dtype"]

    print(f"   Input Details : shape={in_shape}, dtype={in_type}")
    print(f"   Output Details: shape={out_shape}, dtype={out_type}")

    # Uji inferensi dummy (1 sekuens dummy 30x63)
    dummy_input = np.random.randn(1, 30, 63).astype(np.float32)

    # Prediksi Keras
    keras_pred = model.predict(dummy_input, verbose=0)

    # Prediksi TFLite
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    t0 = time.perf_counter()
    interpreter.invoke()
    latensi_ms = (time.perf_counter() - t0) * 1000
    tflite_pred = interpreter.get_tensor(output_details[0]["index"])

    selisih_maks = np.max(np.abs(keras_pred - tflite_pred))
    print(f"\n   Latensi Inferensi CPU : {latensi_ms:.2f} ms")
    print(f"   Selisih Keras vs TFLite : {selisih_maks:.8f}")

    if selisih_maks < 1e-5:
        print("\n[SUKSES] Model TFLite 1D-CNN 100% valid dan identik dengan Keras!")
        print(f"Berkas siap diserahkan ke Sheva untuk Flutter: {TFLITE_PATH}")
    else:
        print("\n[PERINGATAN] Ada selisih numerik yang cukup besar.")


if __name__ == "__main__":
    main()
