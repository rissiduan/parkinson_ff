import os
import cv2
import numpy as np
from skimage.feature import hog
import pickle
from flask import Flask, render_template, request, jsonify, session
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.secret_key = 'a_random_and_secure_string'  # Secret key for session

# ----------- ROUTES ----------
@app.route('/')
def index():
    print("Navigating to the index page...")
    return render_template("index.html")

@app.route('/drawnSP')
def drawn_sp():
    print("Navigating to drawn spiral page...")
    return render_template("drawn_Sp.html")

@app.route('/drawnWave')
def drawn_wave_page():
    print("Navigating to drawn wave page...")
    return render_template("drawn_wave.html")

@app.route('/upload')
def upload_page():
    print("Navigating to upload page...")
    return render_template("upload.html")

@app.route('/result')
def result_page():
    print("Navigating to result page...")
    return render_template("result.html")


# ----------- UTIL FUNCTION ----------
def extract_gray_flatten(image):
    print("Extracting gray scale and flattening image...")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    return img.flatten()

def extract_hog_features(image):
    print("Extracting HOG features...")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features


# ----------- API: Spiral Precheck ----------
@app.route('/sp_check', methods=['POST'])
def sp_check():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("No image data found in request")

        print("Received image data for spiral precheck...")

        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = os.path.join('model', 'model_check', 'svm_model_sp_check.pkl')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        features = extract_gray_flatten(image).reshape(1, -1)
        print(f"Extracted features for spiral: {features}")

        prediction = model.predict(features)
        print(f"Spiral prediction: {prediction}")

        confidence = max(model.predict_proba(features)[0])
        print(f"Spiral prediction confidence: {confidence}")

        result = "Yes" if prediction[0] == 1 else "No"
        print(f"Spiral result: {result}")

        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})
    
    except Exception as e:
        print(f"Error in spiral precheck: {str(e)}")
        return jsonify({'error': str(e)})


# ----------- API: Wave Precheck ----------
@app.route('/wave_check', methods=['POST'])
def wave_check():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("No image data found in request")

        print("Received image data for wave precheck...")

        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = os.path.join('model', 'model_check', 'svm_model_wave_check.pkl')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        features = extract_gray_flatten(image).reshape(1, -1)
        print(f"Extracted features for wave: {features}")

        prediction = model.predict(features)
        print(f"Wave prediction: {prediction}")

        confidence = max(model.predict_proba(features)[0])
        print(f"Wave prediction confidence: {confidence}")

        result = "Yes" if prediction[0] == 1 else "No"
        print(f"Wave result: {result}")

        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})

    except Exception as e:
        print(f"Error in wave precheck: {str(e)}")
        return jsonify({'error': str(e)})


# ----------- API: Spiral Final ----------
@app.route('/sp', methods=['POST'])
def cal_Sp():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            raise ValueError("No image data found in the request")
        
        print("Received image data for final spiral check...")

        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = os.path.join('model', 'svm_spiral_model_ff4.pkl')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        features = extract_hog_features(image).reshape(1, -1)
        print(f"Extracted HOG features for spiral: {features}")

        prediction = model.predict(features)
        print(f"Spiral final prediction: {prediction}")

        confidence = max(model.predict_proba(features)[0])
        print(f"Spiral final prediction confidence: {confidence}")

        result = "Healthy" if prediction[0] == 1 else "Parkinson"
        print(f"Spiral final result: {result}")

        session['sp_result'] = (result, confidence)

        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})

    except Exception as e:
        print(f"Error in final spiral check: {str(e)}")
        return jsonify({'error': str(e)})


# ----------- API: Wave Final ----------
@app.route('/Wave', methods=['POST'])
def cal_Wave():
    try:
        if 'image' in request.form:
            image_b64 = request.form['image']
        else:
            image_b64 = request.get_json()['image']
        
        print("Received image data for final wave check...")

        image_data = base64.b64decode(image_b64.split(',')[1])
        image = Image.open(BytesIO(image_data))

        model_path = os.path.join('model', 'svm_WAVE_model2.pkl')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        features = extract_hog_features(image).reshape(1, -1)
        print(f"Extracted HOG features for wave: {features}")

        prediction = model.predict(features)
        print(f"Wave final prediction: {prediction}")

        confidence = max(model.predict_proba(features)[0])
        print(f"Wave final prediction confidence: {confidence}")

        result = "Healthy" if prediction[0] == 0 else "Parkinson"
        print(f"Wave final result: {result}")

        session['wave_result'] = (result, confidence)

        return jsonify({'status': 'success', 'result': result, 'confidence': float(confidence)})

    except Exception as e:
        print(f"Error in final wave check: {str(e)}")
        return jsonify({'error': str(e)})


# ----------- Final Combined Result ----------
@app.route('/results', methods=['GET'])
def get_results():
    sp_result, sp_confidence = session.get('sp_result', ('No result', 0))
    wave_result, wave_confidence = session.get('wave_result', ('No result', 0))

    print(f"Final results - Spiral: {sp_result} with confidence {sp_confidence}")
    print(f"Final results - Wave: {wave_result} with confidence {wave_confidence}")

    weight_sp = 0.48
    weight_wave = 0.52
    confidence_threshold = 0.6

    if sp_confidence == 0 and wave_confidence == 0:
        final_result = "No result available"
    elif sp_confidence >= confidence_threshold and wave_confidence >= confidence_threshold:
        weighted_score = (weight_sp * sp_confidence) + (weight_wave * wave_confidence)
        print(f"Weighted score: {weighted_score}")

        if sp_result == "Parkinson" and wave_result == "Parkinson":
            final_result = "Parkinson"
        elif sp_result == "Healthy" and wave_result == "Healthy":
            final_result = "Healthy"
        else:
            final_result = "Parkinson" if weighted_score > 0.5 else "Healthy"
    else:
        if sp_confidence > wave_confidence:
            final_result = sp_result
        else:
            final_result = wave_result

    print(f"Final result after processing: {final_result}")
    return render_template('result.html', final_result=final_result)


# ----------- RUN APP ----------
if __name__ == "__main__":
    app.run(debug=True)
