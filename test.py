from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import cv2
from cvzone.HandTrackingModule import HandDetector

# Ensure TensorFlow is used properly
from tensorflow.keras.models import load_model

import numpy as np
import math
import base64
import os
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No training configuration found in the save file")

app = Flask(__name__)

# Enable CORS for specific origin
CORS(app, resources={r"/*": {"origins": "https://salinterpret.vercel.app"}})

# Path to model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new.h5')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load the model and initialize hand detector
classifier = load_model(model_path, compile=False)
detector = HandDetector(maxHands=1)

# Define label mapping for ASL
asl_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

def translate_image(img):
    """Processes the input image and predicts the corresponding ASL letter."""
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgSize = 300
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure crop stays within bounds
        y1, y2 = max(0, y - 20), min(img.shape[0], y + h + 20)
        x1, x2 = max(0, x - 20), min(img.shape[1], x + w + 20)
        imgCrop = img[y1:y2, x1:x2]

        aspectRatio = h / w if w != 0 else 1  # Prevent division by zero

        if aspectRatio > 1:
            k = imgSize / h
            wCal = max(1, math.ceil(k * w))  # Avoid zero width
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = max(1, math.ceil(k * h))  # Avoid zero height
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Make prediction
        prediction = classifier.predict(np.expand_dims(imgWhite, axis=0))
        index = np.argmax(prediction)

        # Map the prediction to the corresponding ASL letter
        translation = asl_labels.get(index, '?')  # '?' if index not found
    else:
        translation = ''
    return translation

@app.route('/translate', methods=['POST', 'OPTIONS'])
@cross_origin(origin='https://salinterpret.vercel.app')  # Ensure CORS for this endpoint
def translate_asl():
    """Endpoint for translating ASL images."""
    if request.method == 'OPTIONS':
        # Return a response for the preflight request
        response = jsonify({'status': 'Preflight handled'})
        response.headers.add("Access-Control-Allow-Origin", "https://salinterpret.vercel.app")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        img_data = base64.b64decode(data['image'])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        translation = translate_image(img)
        response = jsonify({'translation': translation})
        response.headers.add("Access-Control-Allow-Origin", "https://salinterpret.vercel.app")
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
