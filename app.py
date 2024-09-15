import os
from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import shutil

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Set the upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained models
classification_model = load_model('leaves_classification_model.h5')
object_detection_model = YOLO('bestt.pt')  # Update with correct YOLOv8 model path

# Class mapping for the classification model
class_mapping = {0: "amla leaf", 1: "babul leaf", 2: "sammi leaf"}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image for classification
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values to 0-1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Home route to select task
@app.route('/')
def home():
    return render_template('index.html')

# Update classify function
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image_classification', filename)
        file.save(file_path)

        # Preprocess the image and make prediction
        image = preprocess_image(file_path)
        predictions = classification_model.predict(image)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        predicted_label = class_mapping[predicted_class]

        # Save image in static folder with correct path format
        static_image_path = os.path.join('static', 'images', filename).replace('\\', '/')
        shutil.copy(file_path, static_image_path)

        return render_template('result.html', label=predicted_label, image_path=static_image_path)

    else:
        flash('Allowed image types are png, jpg, jpeg')
        return redirect(request.url)

# Update detect function similarly
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'object_detection', filename)
        file.save(file_path)

        # Perform object detection
        results = object_detection_model(file_path)

        # Iterate through results and save the image with correct path
        static_image_path = os.path.join('static', 'images', filename).replace('\\', '/')
        for result in results:
            result.save(static_image_path)

        return render_template('object_detection.html', image_path=static_image_path)

    else:
        flash('Allowed image types are png, jpg, jpeg')
        return redirect(request.url)

if __name__ == '__main__':
    # Ensure the folders for uploads exist
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'image_classification'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'object_detection'), exist_ok=True)
    app.run(debug=True)
