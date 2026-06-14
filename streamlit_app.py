import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Setup Halaman
st.set_page_config(page_title="NeuroScan AI | Brain MRI", page_icon="🧠", layout="centered")
st.title("🧠 NeuroScan AI")
st.markdown("Sistem klasifikasi citra MRI Tumor Otak berbasis Convolutional Neural Network (CNN).")

# Load Model (di-cache biar ringan)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cnn_mri_model.h5')

model = load_model()
CLASS_NAMES = ['NORMAL (TIDAK TERINDIKASI)', 'TERINDIKASI TUMOR OTAK']

# Area Upload
uploaded_file = st.file_uploader("Unggah Citra Pemindaian MRI (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Langsung konversi ke Grayscale (Hitam Putih) sesuai training lu
    image = Image.open(uploaded_file).convert('L') 
    st.image(image, caption='Citra MRI yang diunggah', width=300)

    if st.button('Jalankan Komputasi'):
        with st.spinner('Memproses citra dengan CNN...'):
            try:
                # Preprocessing 64x64
                img_resized = image.resize((64, 64))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Eksekusi Prediksi
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions)
                result = CLASS_NAMES[predicted_class_index]
                confidence = np.max(predictions) * 100
                
                # Menampilkan Hasil
                st.markdown("---")
                if predicted_class_index == 1:
                    st.error(f"### **Hasil: {result}**")
                else:
                    st.success(f"### **Hasil: {result}**")
                st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")
                
                # Penafian Medis
                st.caption("⚠️ Penafian Medis: Hasil komputasi ini murni berfungsi sebagai alat uji awal riset akademis dan tidak menggantikan validasi operasional klinis dari spesialis radiologi medis.")
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")