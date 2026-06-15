# Gunakan Python versi 3.9
FROM python:3.9

# Setup direktori kerja
WORKDIR /app

# Copy requirement dan install library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install library tambahan untuk OpenCV di server linux
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy semua file proyek lu
COPY . .

# Hugging Face pakai port 7860
EXPOSE 7860
ENV FLASK_APP=app.py

# Perintah buat jalanin Flask di server
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]