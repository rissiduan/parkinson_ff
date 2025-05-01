import os
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify, session
from io import BytesIO
from PIL import Image
import base64
import joblib
from flask import send_from_directory, send_file
from skimage.feature import hog
from flask import session

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'a_random_and_secure_string'

# ตรวจสอบ static folder
print(f"Static folder path: {app.static_folder}")
print(f"Static folder exists: {os.path.exists(app.static_folder)}")
if os.path.exists(app.static_folder):
    print(f"Files in static folder: {os.listdir(app.static_folder)}")

# ---------- Utility Functions ----------
def extract_flattened_features(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    return img.flatten()

def is_image_openable(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def extract_hog_features_from_image(image):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

# ---------- Web Routes ----------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/drawnSP')
def drawn_sp():
    return render_template("drawn_Sp.html")

@app.route('/drawnWave')
def drawn_wave():
    return render_template("drawn_wave.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/result')
def result():
    return render_template("result.html")

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/test-image')
def test_image():
    return f"<h1>ทดสอบรูปภาพ</h1><img src='/static/i.png' alt='Test image'>"

@app.route('/check-static')
def check_static():
    static_path = app.static_folder
    if os.path.exists(static_path):
        files = os.listdir(static_path)
        return f"Static folder: {static_path}<br>Files found: {files}"
    else:
        return f"Static folder not found at: {static_path}"

@app.route('/get-image')
def get_image():
    try:
        image_path = os.path.join(app.static_folder, 'i.png')
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/create-test-image')
def create_test_image():
    try:
        img = Image.new('RGB', (100, 100), color='red')
        test_img_path = os.path.join(app.static_folder, 'test.png')
        img.save(test_img_path)
        return f"Test image created at {test_img_path}<br><img src='/static/test.png'>"
    except Exception as e:
        return f"Error: {str(e)}"

# ---------- SP Precheck ----------
@app.route('/sp_check', methods=['POST'])
def sp_check():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path_sp = os.environ.get('MODEL_PATH_SVM_SP', 'model/model_check/svm_model_sp_check-new2.pkl')
        if not os.path.exists(model_path_sp):
            raise FileNotFoundError(f"Model file not found at {model_path_sp}")

        model = joblib.load(open(model_path_sp, 'rb'))
        features = extract_flattened_features(image).reshape(1, -1)
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0
        result = "Yes" if prediction[0] == 1 else "No"

        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- Wave Precheck ----------
@app.route('/wave_check', methods=['POST'])
def wave_check():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path_wave = os.environ.get('MODEL_PATH_SVM_WAVE', 'model/model_check/svm_model_wave_check.pkl')
        if not os.path.exists(model_path_wave):
            raise FileNotFoundError(f"Model file not found at {model_path_wave}")

        model = joblib.load(open(model_path_wave, 'rb'))
        features = extract_flattened_features(image).reshape(1, -1)
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0
        result = "Yes" if prediction[0] == 1 else "No"
        
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- Spiral Upload ----------
@app.route('/sp-upload', methods=['POST'])
def sp_upload():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = 'model/svm_spiral_model_ff4-2.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = joblib.load(open(model_path, 'rb'))
        features = extract_hog_features_from_image(image).reshape(1, -1)
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0
        result = "Healthy" if prediction[0] == 0 else "Parkinson"
        from flask import session

        session['sp_result'] = (result, confidence)  

        print(f"📊 [SP] Features extracted: {result} {confidence}")
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ---------- Wave Upload ----------
@app.route('/wave-upload', methods=['POST'])
def wave_upload():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = 'model/svm_WAVE_model2.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = joblib.load(open(model_path, 'rb'))
        features = extract_hog_features_from_image(image).reshape(1, -1)
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0
        result = "Healthy" if prediction[0] == 0 else "Parkinson" 
        session['wave_result'] = (result, confidence)
        print(f"📊 [WAVE] Features extracted: {result} {confidence}")
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ---------- Final Spiral ----------
@app.route('/sp', methods=['POST'])
def final_spiral():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = 'model/model_sprial_SVM_new.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = joblib.load(open(model_path, 'rb'))
        features = extract_flattened_features(image).reshape(1, -1)
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0
        result = "Healthy" if prediction[0] == 0 else "Parkinson"

        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ---------- Final Wave ----------
@app.route('/Wave', methods=['POST'])
def final_wave():
    try:
        print("📥 [WAVE] Received request")
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))
        print("🖼️ [WAVE] Image decoded successfully")

        model_path = os.path.join('model', 'model_wave_SVM_new.pkl')
        print(f"🔍 [WAVE] Model path: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        print("✅ [WAVE] Model loaded successfully")

        features = extract_flattened_features(image).reshape(1, -1)
        print("📊 [WAVE] Features extracted")

        prediction = model.predict(features)
        print(f"🧠 [WAVE] Raw prediction output: {prediction}")

        result = "Healthy" if prediction[0] == 0 else "Parkinson"
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0
        print(f"✅ [WAVE] Prediction result: {result} (Confidence: {confidence*100:.2f}%)")

        session['wave_result'] = (result, confidence)
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        print(f"❌ [WAVE] Error occurred: {str(e)}")
        return jsonify({'error': str(e)})





# ---------- Combined Result ----------
@app.route('/results', methods=['GET'])
def get_results():
    # ดึงผลลัพธ์จาก session (บันทึกจากการทำนายก่อนหน้า)
    sp_result, sp_prob = session.get('sp_result', ('No result', 0.0))
    wave_result, wave_prob = session.get('wave_result', ('No result', 0.0))

    # สเต็ป 1: แปลง prob ให้เป็น "โอกาสที่จะเป็น Parkinson" เท่านั้น
    sp_parkinson_prob = sp_prob if sp_result == "Parkinson" else 1 - sp_prob
    wave_parkinson_prob = wave_prob if wave_result == "Parkinson" else 1 - wave_prob

    # สเต็ป 2: ปรับน้ำหนักตามความน่าเชื่อถือของแต่ละโมเดล
    # ให้ Spiral มี accuracy 95.64% → 0.9564
    # ให้ Wave มี accuracy 94.29% → 0.9429
    acc_sp = 0.9564
    acc_wave = 0.9429

    total_acc = acc_sp + acc_wave
    weight_sp = acc_sp / total_acc
    weight_wave = acc_wave / total_acc

    # สเต็ป 3: คำนวณ weighted probability แบบ Soft Voting
    weighted_probability = (weight_sp * sp_parkinson_prob) + (weight_wave * wave_parkinson_prob)

    # สเต็ป 4: ตัดสินผลลัพธ์
    threshold = 0.5
    final_result = "Parkinson" if weighted_probability >= threshold else "Healthy"

    # สำหรับ debug และดูค่าทั้งหมด
    print("Spiral model: result =", sp_result, ", raw prob =", sp_prob, ", Parkinson prob =", sp_parkinson_prob)
    print("Wave model: result =", wave_result, ", raw prob =", wave_prob, ", Parkinson prob =", wave_parkinson_prob)
    print(f"Weighted probability: {weighted_probability:.4f} → Final result: {final_result}")

    # สเต็ป 5: ส่งผลลัพธ์ไปยังหน้าเว็บ
    return render_template("result.html",
                           final_result=final_result,
                           overall_conf=round(weighted_probability * 100, 2),
                           sp_result=sp_result,
                           sp_conf=round(sp_prob * 100, 2),
                           wave_result=wave_result,
                           wave_conf=round(wave_prob * 100, 2))



# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
