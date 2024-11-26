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

@app.route('/result')
def result_page():
    return render_template("result.html")

@app.route('/sp', methods=['POST'])
def cal_Sp():
    def extract_hog_features_from_image(image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128, 128))  # Resize image
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        return hog_features

    try:
        # Receive the JSON data from the request
        data = request.get_json()  # Use get_json() for JSON request body
        if not data or 'image' not in data:
            raise ValueError("No image data found in the request")
        
        image_data = data['image'].split(',')[1]  # Remove the prefix part of the Base64 string
        image_data = base64.b64decode(image_data)  # Decode Base64 data
        image = Image.open(BytesIO(image_data))

        # Load the model using pickle
        model_path = 'model\svm_spiral_model_ff.pkl'
        try:
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
        except Exception as e:
            raise ValueError(f"Error loading the model: {str(e)}")

        # Extract features from the image
        hog_features = extract_hog_features_from_image(image)
        hog_features = np.array(hog_features).reshape(1, -1)

        # Predict the result
        prediction = loaded_model.predict(hog_features)
        result = "Healthy" if prediction[0] == 1 else "Parkinson"

        # Save the result in session
        session['sp_result'] = result
        print("1. "+result)
        # Redirect to results page
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
        # Receive image data from the request
        data = request.form['image']
        data = data.split(',')[1]  # Remove unwanted part
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data))

        # Load the model using pickle
        model_path = 'model\svm_WAVE_model.pkl'
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        # Extract features from the image
        hog_features = extract_hog_features_from_image(image)
        hog_features = np.array(hog_features).reshape(1, -1)

        # Predict the result
        prediction = loaded_model.predict(hog_features)
        result = "Healthy" if prediction[0] == 0 else "Parkinson"

        # Save the result in session
        session['wave_result'] = result
        print("2. "+result)
        # Redirect to results page
        return redirect(url_for('get_results'))

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/results', methods=['GET'])
def get_results():
    # Retrieve results from session
    sp_result = session.get('sp_result', 'No result')
    wave_result = session.get('wave_result', 'No result')
    print(f"3. {sp_result}")
    print(f"4. {wave_result}")

    # Display results
    return render_template('result.html', sp_result=sp_result, wave_result=wave_result)


if __name__ == "__main__":
    app.run(debug=True)
