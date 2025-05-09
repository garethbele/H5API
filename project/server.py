from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import base64
import io
from PIL import Image

app = Flask(__name__)
# Enable CORS for all origins to allow requests from any frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
model = None
class_labels = ['Pneumonia', 'Normal', 'Lung Opacity']

def load_model():
    """Load the h5 model."""
    global model
    try:
        model = tf.keras.models.load_model('your_model.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image_data):
    """Preprocess the image for model prediction."""
    # Convert base64 to image
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to match the model's expected input
    img = img.resize((128, 128))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale
    
    return img_array

@app.route('/', methods=['GET'])
def index():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "Chest X-ray classifier API is running",
        "endpoints": {
            "/predict": "POST - Submit an image for classification"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive image and return predictions."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image data from request
        image_data = base64.b64decode(request.json['image'].split(',')[1])
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Convert predictions to list and pair with class labels
        result = {
            'predictions': [
                {'label': label, 'probability': float(prob)} 
                for label, prob in zip(class_labels, predictions[0].tolist())
            ]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    load_model()  # Load the model when server starts
    app.run(host='0.0.0.0', port=port)
