import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import time
import h5py

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rice_leaf_deficiency_model.h5')

# Define class names based on your labels
class_names = ['Nitrogen', 'Phosphorus', 'Potassium']

# Initialize model as None
model = None

def load_model():
    global model
    try:
        print(f"Loading model from: {MODEL_PATH}")
        print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        
        # First try loading with custom_objects
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("Model loaded successfully with compile=False")
        except Exception as e1:
            print(f"First attempt failed: {str(e1)}")
            
            # If that fails, try loading with custom_objects and custom_metrics
            try:
                model = tf.keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                    custom_objects={
                        'InputLayer': tf.keras.layers.InputLayer,
                        'EfficientNetB0': tf.keras.applications.EfficientNetB0
                    }
                )
                print("Model loaded successfully with custom_objects")
            except Exception as e2:
                print(f"Second attempt failed: {str(e2)}")
                
                # If that fails, try loading the model architecture and weights separately
                try:
                    # Create the model architecture
                    base_model = tf.keras.applications.EfficientNetB0(
                        weights=None,
                        include_top=False,
                        input_shape=(224, 224, 3)
                    )
                    x = base_model.output
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    x = tf.keras.layers.Dense(3, activation='softmax')(x)
                    model = tf.keras.Model(inputs=base_model.input, outputs=x)
                    
                    # Load the weights
                    model.load_weights(MODEL_PATH)
                    print("Model loaded successfully by recreating architecture")
                except Exception as e3:
                    print(f"Third attempt failed: {str(e3)}")
                    raise Exception("Failed to load model after multiple attempts")
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model compiled successfully")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image_data):
    # Convert base64 to image
    try:
        # Remove the data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 string
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/')
def home():
    return jsonify({'message': 'Server is running', 'status': 'ok'})

@app.route('/test')
def test():
    return jsonify({'message': 'Test endpoint working', 'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")  # Debug log
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get the JSON data from the request
        data = request.get_json()
        
        if not data or 'image' not in data:
            print("No image data in request")  # Debug log
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get the base64 encoded image
        image_data = data['image']
        
        print("Processing image data...")  # Debug log
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        start_time = time.time()
        predictions = model.predict(processed_image, verbose=0)
        end_time = time.time()
        
        # Get prediction details
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        inference_time = end_time - start_time
        
        return jsonify({
            'class': int(predicted_class_idx),  # 0, 1, or 2
            'class_name': class_names[predicted_class_idx],
            'confidence': confidence,
            'inference_time': inference_time
        })
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    
    # Load the model
    load_model()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port)
