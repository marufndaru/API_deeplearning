# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import uvicorn
import numpy as np
import pickle
import cv2
import os
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from PIL import Image as Img
from numpy import asarray, expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pyngrok import ngrok
import nest_asyncio

# ==========================================
# 1. INISIALISASI & KONFIGURASI
# ==========================================
app = FastAPI(title="FaceNet Presensi API")

# Load Model AI
print("⏳ Sedang memuat model AI (FaceNet & MTCNN)...")
detector = MTCNN()
embedder = FaceNet()
print("✅ Model AI siap.")

# Load Database
DB_FILE = "datafacenet_aug.pkl"
database = {}

def load_database():
    global database
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                database = pickle.load(f)
            print(f"✅ Database dimuat: {len(database)} vektor wajah.")
        except Exception as e:
            print(f"❌ Error memuat database: {e}")
            database = {}
    else:
        print("⚠️ File database belum ada, akan dibuat baru saat registrasi.")
        database = {}

# Panggil fungsi load saat awal
load_database()

# Konfigurasi Augmentasi (Sama seperti saat training)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# ==========================================
# 2. FUNGSI BANTUAN (HELPER)
# ==========================================
def process_image_to_face(image_bytes):
    """
    Mengubah bytes gambar menjadi array wajah (1, 160, 160, 3)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None
        
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_img)
    
    if not results:
        return None
        
    # Ambil wajah terbesar
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    face_crop = rgb_img[y1:y2, x1:x2]
    
    # Resize ke 160x160
    face_pil = Img.fromarray(face_crop).resize((160, 160))
    face_arr = asarray(face_pil)
    face_input = expand_dims(face_arr, axis=0) # (1, 160, 160, 3)
    
    return face_input

def save_database_to_disk():
    """Menyimpan dictionary database ke file .pkl"""
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)
    print(f"💾 Database tersimpan! Total data: {len(database)}")

# ==========================================
# 3. ENDPOINT API
# ==========================================
@app.get("/")
def index():
    return {
        "message": "Sistem Presensi Wajah Online",
        "total_data": len(database)
    }

# --- ENDPOINT 1: INPUT DATA BARU (REGISTER) ---
# --- ENDPOINT 1: INPUT DATA BARU (REGISTER) ---
@app.post("/register")
async def daftar_karyawan(
    nama: str = Form(...), 
    nik: int = Form(...),  # <--- UBAH DI SINI (str jadi int)
    file: UploadFile = File(...)
):
    """
    Mendaftarkan wajah baru ke dalam model.
    NIK wajib berupa angka.
    """
    # Validasi File
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    try:
        # 1. Baca & Proses Wajah
        contents = await file.read()
        face_input = process_image_to_face(contents)
        
        if face_input is None:
            return {"status": "gagal", "pesan": "Wajah tidak terdeteksi pada foto."}

        # 2. Format Nama untuk Key Database
        clean_nama = nama.strip().replace(" ", "_")
        
        # Konversi NIK (int) balik ke string agar bisa jadi nama file/key
        clean_nik = str(nik).strip() 
        
        base_key = f"{clean_nik}_{clean_nama}_api"

        # 3. Generate Embedding Asli
        original_embedding = embedder.embeddings(face_input)
        database[base_key] = original_embedding

        # 4. Lakukan Augmentasi (Membuat 5 variasi)
        aug_iter = datagen.flow(face_input, batch_size=1)
        for i in range(5):
            aug_img = next(aug_iter)
            aug_embedding = embedder.embeddings(aug_img)
            
            aug_key = f"{base_key}_aug{i}"
            database[aug_key] = aug_embedding

        # 5. Simpan ke File Pickle
        save_database_to_disk()

        return {
            "status": "sukses",
            "pesan": f"Berhasil mendaftarkan {clean_nama} (NIK: {clean_nik})",
            "tipe_data_nik": "integer",
            "total_variasi_disimpan": 6, 
            "total_database_sekarang": len(database)
        }

    except Exception as e:
        return {"status": "error", "pesan": str(e)}

# --- ENDPOINT 2: PRESENSI (RECOGNITION) ---
@app.post("/presensi")
async def presensi_wajah(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    try:
        # 1. Baca & Proses Wajah
        contents = await file.read()
        face_input = process_image_to_face(contents) # Pakai fungsi helper
        
        if face_input is None:
            return {"status": "gagal", "pesan": "Wajah tidak terdeteksi"}

        # 2. Generate Embedding
        new_embedding = embedder.embeddings(face_input)

        # 3. Pencocokan
        min_dist = 100
        identity = "Tidak Dikenal"
        
        for (name, db_enc) in database.items():
            dist = np.linalg.norm(new_embedding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        # Bersihkan nama (hilangkan _aug, _api, dll)
        # Contoh key: 123_Budi_api_aug0 -> Jadi: 123_Budi
        raw_identity = identity # Simpan key asli untuk debug
        
        if "_aug" in identity:
            identity = identity.split("_aug")[0]
        if "_api" in identity: # Membersihkan suffix _api jika ada
            identity = identity.replace("_api", "")

        # Threshold
        THRESHOLD = 0.8 # Bisa disesuaikan (0.7 lebih ketat, 0.9 lebih longgar)
        match = True
        if min_dist > THRESHOLD:
            identity = "Tidak Dikenal"
            match = False

        return {
            "status": "sukses",
            "nama": identity,
            "jarak_kemiripan": float(min_dist),
            "dikenali": match,
            # "debug_key": raw_identity # Uncomment jika ingin melihat key asli
        }

    except Exception as e:
        return {"status": "error", "pesan": str(e)}

# ==========================================
# 4. JALANKAN SERVER
# ==========================================
if __name__ == "__main__":
    nest_asyncio.apply()
    
    # Setup Ngrok
    public_url = ngrok.connect(8000).public_url
    print(f"🚀 API Public URL: {public_url}")
    print(f"📄 Dokumentasi API: {public_url}/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)