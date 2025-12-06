# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import pickle
import cv2
import os
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from PIL import Image as Img
from numpy import asarray, expand_dims
from pyngrok import ngrok
import nest_asyncio

# 1. Inisialisasi App & Load Model
# Model dimuat di luar fungsi agar hanya diload sekali saat server nyala (efisien)
app = FastAPI(title="FaceNet Presensi API")

print("⏳ Sedang memuat model AI... (Mohon tunggu)")
detector = MTCNN()
embedder = FaceNet()
print("✅ Model AI siap.")

# 2. Load Database Wajah
# Pastikan file .pkl ada di folder yang sama dengan main.py     <= INI PENTING!!!...
DB_FILE = "datafacenet_aug.pkl"
database = {}

if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        database = pickle.load(f)
    print(f"✅ Database dimuat: {len(database)} wajah.")
else:
    print("❌ PERINGATAN: File database tidak ditemukan!")

# 3. Endpoint API (Pintu Masuk)
@app.get("/")
def index():
    return {"message": "Sistem Presensi Wajah Online"}

@app.post("/presensi")
async def presensi_wajah(file: UploadFile = File(...)):
# A. Validasi apakah file adalah gambar (Versi Aman)
    # Kita izinkan jika content_type None (kosong) atau diawali image/
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    try:
        # B. Baca File Gambar dari Upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # C. Konversi BGR (OpenCV) ke RGB (FaceNet)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # D. Deteksi Wajah (MTCNN)
        results = detector.detect_faces(rgb_img)
        
        if not results:
            return {"status": "gagal", "pesan": "Wajah tidak terdeteksi"}

        # Ambil wajah terbesar
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        face_crop = rgb_img[y1:y2, x1:x2]

        # E. Preprocessing (Resize ke 160x160)
        face_pil = Img.fromarray(face_crop).resize((160, 160))
        face_arr = asarray(face_pil)
        face_input = expand_dims(face_arr, axis=0)

        # F. Generate Embedding (FaceNet)
        new_embedding = embedder.embeddings(face_input)

        # G. Pencocokan dengan Database
        min_dist = 100
        identity = "Tidak Dikenal"
        
        for (name, db_enc) in database.items():
            dist = np.linalg.norm(new_embedding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        # Bersihkan nama (hilangkan _aug)
        if "_aug" in identity:
            identity = identity.split("_aug")[0]

        # Threshold (Ambang Batas)
        THRESHOLD = 0.8
        match = True
        if min_dist > THRESHOLD:
            identity = "Tidak Dikenal"
            match = False

        # H. Return Hasil (JSON)
        return {
            "status": "sukses",
            "nama": identity,
            "jarak_kemiripan": float(min_dist),
            "dikenali": match
        }

    except Exception as e:
        return {"status": "error", "pesan": str(e)}

# 4. Blok untuk menjalankan script langsung (Opsional)
if __name__ == "__main__":
    # Aktifkan nest_asyncio untuk kompatibilitas
    nest_asyncio.apply()
    
    # Buka tunnel ngrok
    ngrok_tunnel = ngrok.connect(8000)
    print('🌐 Public URL:', ngrok_tunnel.public_url)
    
    # Jalankan server
    uvicorn.run(app, host="127.0.0.1", port=8000)