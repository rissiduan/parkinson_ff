import cv2
import numpy as np
from skimage.feature import hog
import pickle
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.secret_key = 'a_random_and_secure_string'  # Secret key for session

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/drawnSP')
def drawn_sp():
    return render_template("drawn_Sp.html")

@app.route('/drawnWave')
def drawn_wave_page():
    return render_template("drawn_wave.html")

@app.route('/upload')
def upload_page():
    return render_template("upload.html")

@app.route('/result')
def result_page():
    return render_template("result.html")

@app.route('/sp_check', methods=['POST'])
def sp_check():
    def extract_hog_features_from_image(image):
        print("[1] เริ่มแปลงภาพเป็น grayscale และปรับขนาด...")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128, 128))
        print("[2] เริ่มดึง HOG features...")
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys')
        return hog_features

    try:
        print("======== SPIRAL CHECK เริ่มต้น ========")

        print("[0] รับข้อมูลจาก client แล้ว")
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("No image data found in the request")

        print("[1] แปลงข้อมูล Base64 กลับเป็นภาพ")
        image_data = data['image'].split(',')[1]
        image_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_data))

        print("[2] โหลดโมเดลตรวจสอบก้นหอย")
        model_path = 'model/model_check/svm_model_sp_check.pkl'
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        print("[3] ดึงคุณลักษณะ HOG จากภาพ")
        hog_features = extract_hog_features_from_image(image)
        hog_features = np.array(hog_features).reshape(1, -1)

        print("[4] ทำการพยากรณ์ผล...")
        prediction = loaded_model.predict(hog_features)
        confidence = max(loaded_model.predict_proba(hog_features)[0])

        result = "Healthy" if prediction[0] == 1 else "Parkinson"

        print(f"[5] ผลลัพธ์การตรวจสอบ: {result} (Confidence: {confidence:.4f})")
        session['sp_result'] = (result, confidence)

        print("======== SPIRAL CHECK เสร็จสิ้น ========")
        return redirect(url_for('get_results'))

    except Exception as e:
        print(f"[ERROR] เกิดข้อผิดพลาด: {str(e)}")
        return jsonify({'error': str(e)})



@app.route('/wave_check', methods=['POST'])
def wave_check():
    def extract_hog_features_from_image(image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128, 128))
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        return hog_features

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("No image data found in the request")
        
        image_data = data['image'].split(',')[1]
        image_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_data))

        # Load the wave model
        model_path = 'model/model_check/svm_model_wave_check.pkl'
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        # Extract HOG features from the image
        hog_features = extract_hog_features_from_image(image)
        hog_features = np.array(hog_features).reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(hog_features)
        confidence = max(loaded_model.predict_proba(hog_features)[0])

        # Determine the result based on prediction
        result = "Healthy" if prediction[0] == 0 else "Parkinson"
        
        # Save the result and confidence
        session['wave_result'] = (result, confidence)
        print(f"Wave Check Result: {result} (Confidence: {confidence})")
        return redirect(url_for('get_results'))

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/sp', methods=['POST'])
def cal_Sp():
    def extract_hog_features_from_image(image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128, 128))
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        return hog_features

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("No image data found in the request")
        
        image_data = data['image'].split(',')[1]
        image_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_data))

        model_path = 'model\svm_spiral_model_ff4.pkl'
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        hog_features = extract_hog_features_from_image(image)
        hog_features = np.array(hog_features).reshape(1, -1)

        prediction = loaded_model.predict(hog_features)
        confidence = max(loaded_model.predict_proba(hog_features)[0])

        result = "Healthy" if prediction[0] == 1 else "Parkinson"
        session['sp_result'] = (result, confidence)
        print(f"1. {result} (Confidence: {confidence})")
        return redirect(url_for('get_results'))

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})


@app.route('/Wave', methods=['POST'])
def cal_Wave():
    def extract_hog_features_from_image(image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128, 128))
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        return hog_features

    try:
        data = request.form['image']
        data = data.split(',')[1]
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data))

        model_path = 'model\svm_WAVE_model2.pkl'
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        hog_features = extract_hog_features_from_image(image)
        hog_features = np.array(hog_features).reshape(1, -1)

        prediction = loaded_model.predict(hog_features)
        confidence = max(loaded_model.predict_proba(hog_features)[0])

        result = "Healthy" if prediction[0] == 0 else "Parkinson"
        session['wave_result'] = (result, confidence)
        print(f"2. {result} (Confidence: {confidence})")
        return redirect(url_for('get_results'))

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/results', methods=['GET'])
def get_results():
    sp_result, sp_confidence = session.get('sp_result', ('No result', 0))
    wave_result, wave_confidence = session.get('wave_result', ('No result', 0))

    # ปรับค่าน้ำหนักตามความแม่นยำของโมเดล
    weight_sp = 0.48  # น้ำหนักสำหรับโมเดล SP (accuracy 0.9248)
    weight_wave = 0.52  # น้ำหนักสำหรับโมเดล Wave (accuracy 0.9429)
    
    confidence_threshold = 0.6  # กำหนด threshold ความมั่นใจขั้นต่ำ

    if sp_confidence == 0 and wave_confidence == 0:
        final_result = "No result available"
    else:
        # หากมีความมั่นใจต่ำกว่า threshold ให้ใช้โมเดลที่มั่นใจสูงสุด
        if sp_confidence >= confidence_threshold and wave_confidence >= confidence_threshold:
            # การคำนวณคะแนนแบบถ่วงน้ำหนัก
            weighted_score = (weight_sp * sp_confidence) + (weight_wave * wave_confidence)

            if sp_result == "Parkinson" and wave_result == "Parkinson":
                final_result = "Parkinson"
            elif sp_result == "Healthy" and wave_result == "Healthy":
                final_result = "Healthy"
            else:
                # ถ้าผลลัพธ์ไม่ตรงกัน ใช้คะแนนแบบถ่วงน้ำหนักเพื่อตัดสิน
                final_result = "Parkinson" if weighted_score > 0.5 else "Healthy"
        else:
            # ใช้โมเดลที่มีความมั่นใจสูงสุดในการตัดสินใจ
            if sp_confidence > wave_confidence:
                final_result = "Healthy" if sp_result == "Healthy" else "Parkinson"
            else:
                final_result = "Healthy" if wave_result == "Healthy" else "Parkinson"

    print(f"3. {sp_result} (Confidence: {sp_confidence})")
    print(f"4. {wave_result} (Confidence: {wave_confidence})")
    print(f"Final Result: {final_result}")

    return render_template('result.html', final_result=final_result)






if __name__ == "__main__":
    app.run(debug=True)
