import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Setup path dan kategori menyesuaikan folder dataset Kaggle
DATA_DIR = 'dataset'
CATEGORIES = ['no', 'yes'] # 'no' = Normal, 'yes' = Tumor
IMG_SIZE = 64 # Resize gambar biar komputasi ringan

data = []
labels = []

print("Mulai membaca citra MRI...")
for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    class_num = CATEGORIES.index(category)
    
    # Looping semua gambar di dalam folder
    for img_name in os.listdir(path): 
        try:
            img_path = os.path.join(path, img_name)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            
            # Flatten: Ubah matriks 2D gambar jadi array 1D buat Random Forest
            flattened_array = resized_array.flatten()
            
            data.append(flattened_array)
            labels.append(class_num)
        except Exception as e:
            pass

X = np.array(data)
y = np.array(labels)

print("Membagi data latih dan uji...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Melatih model Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluasi Model
y_pred = rf_model.predict(X_test)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Simpan Model
joblib.dump(rf_model, 'rf_mri_model.pkl')
print("Mantap! Model disimpan sebagai 'rf_mri_model.pkl'")