import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. Setup Direktori Dataset
DATASET_DIR = 'dataset'
IMG_SIZE = (64, 64) # Resolusi sakti, enteng tapi akurat
BATCH_SIZE = 32
EPOCHS = 25 # Digeber 25 kali biar bener-bener hafal polanya

print("Membaca dataset MRI dari folder...")
# Menggunakan mode grayscale (hitam putih) agar sesuai dengan sifat alami MRI
train_dataset = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale', # Memaksa gambar jadi 1 channel (hitam putih)
    class_names=['no', 'yes']
)

validation_dataset = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_names=['no', 'yes']
)

class_names = train_dataset.class_names
print(f"Kelas target yang dideteksi: {class_names}")

# 2. Membangun Arsitektur CNN Murni Khusus Medis
print("\nMembangun arsitektur CNN Grayscale...")
model = keras.Sequential([
    # Input shape menyesuaikan: 64x64 piksel dengan 1 channel (grayscale)
    keras.Input(shape=(64, 64, 1)),
    
    # Normalisasi otomatis ditanam di model
    keras.layers.Rescaling(1./255),
    
    # Layer Ekstraksi Fitur
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Layer Klasifikasi
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3), # Regularisasi ideal
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. Melatih Model
print(f"\n[START] Memulai proses training {EPOCHS} Epochs...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# 4. Simpan Model
print("\n[FINISH] Menyimpan model cerdas...")
model.save('cnn_mri_model.h5')
print("Model berhasil disimpan dengan nama 'cnn_mri_model.h5'")

# 5. MEMBUAT INDIKATOR GRAFIK EVALUASI
print("\nMembuat grafik indikator evaluasi...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Akurasi Training dan Validasi MRI')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss Training dan Validasi MRI')

plt.savefig('grafik_mri.png')
print("Grafik evaluasi berhasil disimpan sebagai 'grafik_mri.png'")