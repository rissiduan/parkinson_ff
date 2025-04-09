import cv2
import numpy as np
import pickle
from tkinter import Tk, filedialog
from PIL import Image

# --------- ฟังก์ชันเตรียมภาพ ----------
def prepare_image(image_path):
    print("[1] โหลดภาพและปรับขนาด...")
    img = Image.open(image_path).convert("L")  # แปลงเป็น grayscale
    img = img.resize((128, 128))               # ปรับขนาด
    img_np = np.array(img).flatten()           # flatten เป็น 1 มิติ
    return img_np.reshape(1, -1)               # reshape สำหรับโมเดล

# --------- โหลดโมเดล ----------
model_path = 'model/model_check/svm_model_wave_check.pkl'
print(f"[2] กำลังโหลดโมเดลจาก {model_path} ...")
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# --------- เลือกรูปภาพ ----------
Tk().withdraw()  # ปิดหน้าต่าง root
file_path = filedialog.askopenfilename(title="เลือกภาพ Spiral ที่จะตรวจสอบ", 
                                       filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")])
if not file_path:
    print("❌ ไม่ได้เลือกรูปภาพ")
    exit()

# --------- ทำนายภาพ ----------
features = prepare_image(file_path)
prediction = model.predict(features)[0]

# confidence ถ้ามี
try:
    confidence = max(model.predict_proba(features)[0])
except:
    confidence = None

# --------- แสดงผลลัพธ์ ----------
result = "Yes" if prediction == 1 else "No"
print(f"\n✅ ผลลัพธ์การตรวจสอบ: {result}")
if confidence is not None:
    print(f"📊 ความมั่นใจ: {confidence*100:.2f}%")
