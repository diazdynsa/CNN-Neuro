from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load model khusus klasifikasi MRI tumor otak grayscale
model = tf.keras.models.load_model('cnn_mri_model.h5')

# Urutan kelas: 'no' (0), 'yes' (1)
CLASS_NAMES = ['NORMAL (TIDAK TERINDIKASI TUMOR)', 'TERINDIKASI TUMOR OTAK'] 
IMG_SIZE = (64, 64) # Mengikuti resolusi training

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="Tidak ada file gambar.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="Pilih citra MRI terlebih dahulu.")

    if file:
        try:
            img_bytes = file.read()
            # KUNCI JAWABAN: Konversi gambar user ke format hitam putih otomatis
            img = image.load_img(io.BytesIO(img_bytes), target_size=IMG_SIZE, color_mode='grayscale')
            
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) 
            # Normalisasi udah ditanam di model, jadi langsung diprediksi aja
            
            # Eksekusi Prediksi
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            
            result = CLASS_NAMES[predicted_class_index]
            confidence = np.max(predictions) * 100
            
            return render_template('index.html', result=result, confidence=f"{confidence:.2f}%")
        except Exception as e:
            return render_template('index.html', error=f"Gagal memproses gambar mri: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)