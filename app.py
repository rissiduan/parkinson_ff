import os
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify, session
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)
app.secret_key = 'a_random_and_secure_string'

# ---------- Utility Functions ----------
def extract_flattened_features(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    return img.flatten()

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

# ---------- Spiral Precheck ----------
@app.route('/sp_check', methods=['POST'])
def sp_check():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = os.path.join('model', 'model_check', 'svm_model_sp_check.pk')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        features = extract_flattened_features(image).reshape(1, -1)
        prediction = model.predict(features)
        result = "Yes" if prediction[0] == 1 else "No"
        print(result)
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0
        print(result)
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

        model_path = os.path.join('model', 'model_check', 'svm_model_wave_check.pk')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        features = extract_flattened_features(image).reshape(1, -1)
        prediction = model.predict(features)
        result = "Yes" if prediction[0] == 1 else "No"

        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- Final Spiral ----------
@app.route('/sp', methods=['POST'])
def final_spiral():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = os.path.join('model', 'model_sprial_SVM_new.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        features = extract_flattened_features(image).reshape(1, -1)
        prediction = model.predict(features)
        result = "Healthy" if prediction[0] == 1 else "Parkinson"
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0

        session['sp_result'] = (result, confidence)
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- Final Wave ----------
@app.route('/Wave', methods=['POST'])
def final_wave():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = os.path.join('model', 'model_wave_SVM_new.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        features = extract_flattened_features(image).reshape(1, -1)
        prediction = model.predict(features)
        result = "Healthy" if prediction[0] == 0 else "Parkinson"
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else 1.0

        session['wave_result'] = (result, confidence)
        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- Combined Result ----------
@app.route('/results', methods=['GET'])
def get_results():
    sp_result, sp_conf = session.get('sp_result', ('No result', 0))
    wave_result, wave_conf = session.get('wave_result', ('No result', 0))

    weight_sp = 0.48
    weight_wave = 0.52
    confidence_threshold = 0.6

    if sp_conf == 0 and wave_conf == 0:
        final_result = "No result available"
    elif sp_conf >= confidence_threshold and wave_conf >= confidence_threshold:
        weighted_score = weight_sp * sp_conf + weight_wave * wave_conf
        if sp_result == wave_result:
            final_result = sp_result
        else:
            final_result = "Parkinson" if weighted_score > 0.5 else "Healthy"
    else:
        final_result = sp_result if sp_conf > wave_conf else wave_result

    return render_template("result.html", final_result=final_result)

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
