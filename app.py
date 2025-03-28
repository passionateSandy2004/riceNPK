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
import gc
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU usage and optimize memory
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the absolute path to the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rice_leaf_deficiency_model.h5')

# Define class names based on your labels
class_names = ['Nitrogen', 'Phosphorus', 'Potassium']

# Initialize model as None
model = None

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def create_model():
    """Create the model architecture"""
    try:
        logger.info("Creating model architecture...")
        # Create the base model with the same configuration as the saved model
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',  # Use ImageNet weights
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        # Create the model
        model = tf.keras.Model(inputs=base_model.input, outputs=x)
        
        # Compile the model with CPU-optimized settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False,  # Disable eager execution for better performance
            jit_compile=True  # Enable XLA compilation for better performance
        )
        
        logger.info("Model architecture created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def load_model():
    global model
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        logger.info(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        log_memory_usage()
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Clear any existing model from memory
        if model is not None:
            del model
            gc.collect()
            log_memory_usage()
        
        # First try to load the entire model
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("Model loaded successfully as complete model")
            
            # Recompile the model with CPU-optimized settings
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                run_eagerly=False,
                jit_compile=True
            )
            log_memory_usage()
        except Exception as e1:
            logger.error(f"Complete model loading failed: {str(e1)}")
            
            # If that fails, create the model and load weights
            try:
                # Create the model architecture
                model = create_model()
                
                # Load the weights
                model.load_weights(MODEL_PATH)
                logger.info("Model weights loaded successfully")
                log_memory_usage()
            except Exception as e2:
                logger.error(f"Model creation and weight loading failed: {str(e2)}")
                raise
        
        # Verify the model is working with a small test input
        test_input = np.random.random((1, 224, 224, 3))
        _ = model.predict(test_input, verbose=0)
        logger.info("Model verification successful")
        
        # Clear test input from memory
        del test_input
        gc.collect()
        log_memory_usage()
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/')
def home():
    return jsonify({'message': 'Server is running', 'status': 'ok'})

@app.route('/test')
def test():
    return jsonify({'message': 'Test endpoint working', 'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    log_memory_usage()
    
    try:
        # Check if model is loaded
        if model is None:
            logger.info("Model not loaded, attempting to load...")
            load_model()
            if model is None:
                return jsonify({'error': 'Failed to load model'}), 500
        
        # Get the JSON data from the request
        data = request.get_json()
        
        if not data or 'image' not in data:
            logger.error("No image data in request")
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get the base64 encoded image
        image_data = data['image']
        
        logger.info("Processing image data...")
        
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
        
        # Clear processed image from memory
        del processed_image
        gc.collect()
        log_memory_usage()
        
        return jsonify({
            'class': int(predicted_class_idx),  # 0, 1, or 2
            'class_name': class_names[predicted_class_idx],
            'confidence': confidence,
            'inference_time': inference_time
        })
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Load the model when the application starts
logger.info("Starting Flask server...")
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Model exists: {os.path.exists(MODEL_PATH)}")
logger.info("Using CPU for inference")
logger.info("Optimizing TensorFlow for CPU...")
log_memory_usage()

try:
    load_model()
    logger.info("Model loaded successfully at startup")
except Exception as e:
    logger.error(f"Error loading model at startup: {str(e)}")

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port)
