import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import time

app = Flask(_name_)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Get the absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'rice_leaf_deficiency_model.h5')

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names based on your labels
class_names = ['Nitrogen', 'Phosphorus', 'Potassium']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess for EfficientNetB0
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image
            processed_image = preprocess_image(filepath)
            
            # Make prediction
            start_time = time.time()
            predictions = model.predict(processed_image, verbose=0)
            end_time = time.time()
            
            # Get prediction details
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]) * 100)
            inference_time = end_time - start_time
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'class': int(predicted_class_idx),  # 0, 1, or 2
                'class_name': class_names[predicted_class_idx],
                'confidence': confidence,
                'inference_time': inference_time
            })
            
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if _name_ == '_main_':
    app.run(debug=True, port=5000)
