"""
02_train_1dcnn_kata.py
Melatih model 1D-CNN (Temporal Convolution) untuk klasifikasi sekuens kata sapaan dasar SIBI.

Input  : Tensor [batch, 30, 63] (30 frame rentang waktu x 63 koordinat landmark)
Output : Probabilitas softmax kelas kata (HALO, TERIMA_KASIH, SAMA_SAMA, MAAF)

Model disimpan di:
  - model/talky_kata_1dcnn.keras
  - model/labels_kata.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_KATA_DIR = os.path.join(FOLDER, "data_kata")
MODEL_DIR = os.path.join(FOLDER, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63
SEED = 42
keras.utils.set_random_seed(SEED)


def load_dataset():
    """Membaca semua file .npy dari subfolder data_kata/<LABEL>/"""
    if not os.path.exists(DATA_KATA_DIR):
        print(f"Folder dataset {DATA_KATA_DIR} belum ada. Rekam data dengan 01_collect_kata.py dulu.")
        sys.exit(1)

    X_list = []
    y_list = []
    person_list = []
    file_list = []

    subfolders = [f for f in os.listdir(DATA_KATA_DIR) if os.path.isdir(os.path.join(DATA_KATA_DIR, f))]
    if not subfolders:
        print("Belum ada data kata sapaan yang terekam.")
        sys.exit(1)

    print(f"Membaca data kata dari: {DATA_KATA_DIR}")
    for label in sorted(subfolders):
        folder = os.path.join(DATA_KATA_DIR, label)
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        print(f"  - {label:<15}: {len(files)} sampel")

        for f in files:
            path = os.path.join(folder, f)
            arr = np.load(path)
            # Validasi dimensi
            if arr.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                # Ambil person_id dari nama file (format: person_id_LABEL_idx.npy)
                parts = f.split("_")
                p_id = parts[0] if len(parts) >= 3 else "unknown"

                X_list.append(arr)
                y_list.append(label)
                person_list.append(p_id)
                file_list.append(f)
            else:
                print(f"    [SKIP] Dimensi salah pada {f}: {arr.shape} (seharusnya {(SEQUENCE_LENGTH, NUM_FEATURES)})")

    if len(X_list) == 0:
        print("Error: Tidak ada sampel valid yang ditemukan!")
        sys.exit(1)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list)
    persons = np.asarray(person_list)

    print(f"\nTotal dataset dimuat: {len(X)} sampel.")
    return X, y, persons


def build_1d_cnn(num_classes):
    """
    Arsitektur 1D-CNN Temporal Convolution
    Input: [batch, 30, 63] -> melakukan konvolusi sepanjang dimensi waktu (30 frame).
    """
    inputs = keras.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES), name="sequence_input")

    # Blok Konvolusi 1
    x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.2)(x)

    # Blok Konvolusi 2
    x = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Global Average Pooling mereduksi dimensi temporal ke representasi fitur global
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Dense classifier
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Talky_Kata_1DCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X, y_raw, persons = load_dataset()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    num_classes = len(classes)

    print(f"Kelas kata sapaan ({num_classes} kelas): {classes}")

    # Simpan labels_kata.txt
    labels_path = os.path.join(MODEL_DIR, "labels_kata.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(f"{c}\n")
    print(f"Daftar label tersimpan: {labels_path}")

    # Strategi Split Data:
    # Jika data masih sedikit atau 1-2 orang, gunakan stratified split 80% train / 20% test.
    # Jika sudah >= 3 orang, bisa split berdasarkan person_id.
    unique_persons = np.unique(persons)
    print(f"Perekam terdeteksi ({len(unique_persons)} orang): {unique_persons}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"\nData Latih : {len(X_train)} sampel")
    print(f"Data Uji   : {len(X_test)} sampel")

    model = build_1d_cnn(num_classes)
    model.summary()

    # Callbacks
    model_save_path = os.path.join(MODEL_DIR, "talky_kata_1dcnn.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print("\nMemulai pelatihan model 1D-CNN ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluasi akhir pada Data Uji
    print("\n" + "=" * 60)
    print("EVALUASI PERFORMA MODEL 1D-CNN PADA DATA UJI")
    print("=" * 60)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Akurasi Uji: {acc:.4f} ({acc * 100:.2f}%)")

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    print(df_cm.to_string())

    print(f"\nModel Keras tersimpan di: {model_save_path}")
    print("Langkah selanjutnya: Jalankan 'python 03_export_kata_tflite.py' untuk mengekspor ke format TFLite.")


if __name__ == "__main__":
    main()
