from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load model yang udah dilatih dari train.py
model = joblib.load('rf_mri_model.pkl')

IMG_SIZE = 64
CATEGORIES = ['Otak Normal', 'Terindikasi Tumor']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah ada file yang diunggah
    if 'file' not in request.files:
        return render_template('index.html', error="Tidak ada file gambar.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="Pilih citra MRI terlebih dahulu.")

    if file:
        # Baca gambar dari memory (tanpa disave ke folder static)
        file_bytes = np.fromfile(file, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Preprocessing biar sama persis kayak pas training
        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        flattened_img = resized_img.flatten().reshape(1, -1)
        
        # Eksekusi Prediksi
        prediction = model.predict(flattened_img)
        result = CATEGORIES[prediction[0]]
        
        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)