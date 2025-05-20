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
        predicted_class = prediction[0]
        probs = model.predict_proba(features)[0]
        confidence = probs[predicted_class] if hasattr(model, 'predict_proba') else 1.0
        print(f"🧠 [SP] Raw prediction class: {predicted_class}")
        result = "Healthy" if predicted_class == 0 else "Parkinson"

        session['sp_result'] = (result, confidence)

        print(f"📊 [SP] Features extracted: {result} {confidence}")
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence),'predicted_class': predicted_class})
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
        predicted_class = prediction[0]
        probs = model.predict_proba(features)[0]
        confidence = probs[predicted_class] if hasattr(model, 'predict_proba') else 1.0
        print(f"🧠 [wave] Raw prediction class: {predicted_class}")
        result = "Healthy" if predicted_class == 0 else "Parkinson"

        session['wave_result'] = (result, confidence)
        print(f"📊 [WAVE] Features extracted: {result} {confidence}")
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence),'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ---------- Draw Spiral ----------
@app.route('/sp', methods=['POST'])
def final_spiral():
    try:
        print("📥 [SPIRAL] Received request")
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))
        print("🖼️ [SPIRAL] Image decoded successfully")

        model_path = os.path.join('model', 'model_sprial_SVM_new.pkl')
        print(f"🔍 [SPIRAL] Model path: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        print("✅ [SPIRAL] Model loaded successfully")

        features = extract_flattened_features(image).reshape(1, -1)
        print("📊 [SPIRAL] Features extracted")

        prediction = model.predict(features)
        predicted_class = prediction[0]
        probs = model.predict_proba(features)[0]
        confidence = probs[predicted_class] if hasattr(model, 'predict_proba') else 1.0
        print(f"🧠 [SPIRAL] Raw prediction output: {prediction}")

        result = "Healthy" if predicted_class == 0 else "Parkinson"
        print(f"✅ [SPIRAL] Prediction result: {result} (Confidence: {confidence*100:.2f}%)")

        session['sp_result'] = (result, confidence)
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence),'predicted_class': predicted_class})
    except Exception as e:
        print(f"❌ [SPIRAL] Error occurred: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# ---------- Draw Wave ----------
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
        predicted_class = prediction[0]
        probs = model.predict_proba(features)[0]
        confidence = probs[predicted_class] if hasattr(model, 'predict_proba') else 1.0
        print(f"🧠 [WAVE] Raw prediction output: {prediction}")

        result = "Healthy" if predicted_class == 0 else "Parkinson"
        print(f"✅ [WAVE] Prediction result: {result} (Confidence: {confidence*100:.2f}%)")

        session['wave_result'] = (result, confidence)
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence),'predicted_class': predicted_class})
    except Exception as e:
        print(f"❌ [WAVE] Error occurred: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/results', methods=['GET'])
def get_results():
    # ดึงผลลัพธ์จาก session (บันทึกจากการทำนายก่อนหน้า)
    sp_result, sp_prob = session.get('sp_result', ('No result', 0.0))
    wave_result, wave_prob = session.get('wave_result', ('No result', 0.0))

    # สร้างตัวแปรรับ predicted_class ของ Spiral และ Wave
    predicted_class_sp = 1 if sp_result == "Parkinson" else 0
    predicted_class_wave = 1 if wave_result == "Parkinson" else 0

    # กำหนด accuracy ของแต่ละโมเดล (น้ำหนัก)
    acc_sp = 0.9564
    acc_wave = 0.9429

    total_acc = acc_sp + acc_wave
    weight_sp = acc_sp / total_acc
    weight_wave = acc_wave / total_acc

    # แปลง prob ของแต่ละโมเดล ให้เป็นความมั่นใจในผลลัพธ์ที่โมเดลเลือก
    sp_conf_final = sp_prob if sp_result == "Parkinson" else 1 - sp_prob
    wave_conf_final = wave_prob if wave_result == "Parkinson" else 1 - wave_prob

    # คำนวณ weighted confidence โดยรวม (ใช้สำหรับแสดงความมั่นใจในผลลัพธ์ที่แต่ละโมเดลเลือก)
    weighted_confidence = (weight_sp * sp_conf_final) + (weight_wave * wave_conf_final)

    # คำนวณ weighted probability ของ Parkinson (ความน่าจะเป็นที่เป็น Parkinson)
    sp_parkinson_prob = sp_prob if sp_result == "Parkinson" else 1 - sp_prob
    wave_parkinson_prob = wave_prob if wave_result == "Parkinson" else 1 - wave_prob

    weighted_parkinson_prob = (weight_sp * sp_parkinson_prob) + (weight_wave * wave_parkinson_prob)

    # กำหนด threshold เพื่อแยก Parkinson หรือ Healthy
    threshold = 0.5
    final_result = "Parkinson" if weighted_parkinson_prob >= threshold else "Healthy"

    # แก้ไขการคำนวณ overall confidence ให้สอดคล้องกับผลลัพธ์ final_result
    if final_result == "Parkinson":
        overall_conf = weighted_parkinson_prob
    else:
        overall_conf = 1 - weighted_parkinson_prob

    # Debug print
    print("Spiral model: result =", sp_result, ", raw prob =", sp_prob, ", confidence final =", sp_conf_final)
    print("Wave model: result =", wave_result, ", raw prob =", wave_prob, ", confidence final =", wave_conf_final)
    print(f"Weighted confidence (final result): {weighted_confidence:.4f}")
    print(f"Weighted Parkinson prob: {weighted_parkinson_prob:.4f} → Final result: {final_result}")
    print("Overall confidence (adjusted):", overall_conf)
    print("predicted_class_sp: ", predicted_class_sp)
    print("predicted_class_wave: ", predicted_class_wave)

    # ส่งผลลัพธ์ไปยังหน้าเว็บ พร้อมแสดงความมั่นใจในผลลัพธ์สุดท้าย
    return render_template("result.html",
        final_result=final_result,
        overall_conf=round(overall_conf * 100, 2),  # แสดงความมั่นใจใน final result แบบถูกต้อง
        sp_result=sp_result,
        sp_conf=round(sp_prob * 100, 2),
        wave_result=wave_result,
        wave_conf=round(wave_prob * 100, 2),
        sp_prob=sp_prob,
        wave_prob=wave_prob,
        predicted_class_sp=predicted_class_sp,
        predicted_class_wave=predicted_class_wave,
    )




# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
