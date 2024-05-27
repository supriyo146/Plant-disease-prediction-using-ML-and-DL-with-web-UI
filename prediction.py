import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and labels
MODEL_PATH = 'model.h5'
model = None
labels = {0: 'pumpkin early blight',
1: 'pumpkin healthy',
2: 'pumpkin initial stage',
3: 'pumpkin late blight',
4: 'chilli early blight',
5: 'chilli healthy',
6: 'chilli initial stage',
7: 'chilli late blight'}

def load_model_from_file():
    """Load the pre-trained model from file."""
    global model
    model = load_model(MODEL_PATH)
    print('Model loaded. Check http://127.0.0.1:5000/')

def predict_image(image_path):
    """Predict the class of the image."""
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return labels[np.argmax(predictions)]

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        file_path = os.path.join(upload_path, secure_filename(file.filename))
        file.save(file_path)
        try:
            predicted_label = predict_image(file_path)
            return predicted_label
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    return 'Unexpected error occurred'

if __name__ == '__main__':
    load_model_from_file()
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1'])
