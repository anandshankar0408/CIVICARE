from flask import Flask, request, jsonify
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from gevent.pywsgi import WSGIServer
from pathlib import Path
app = Flask(__name__)
# Adjust the filename and construct the new path using .\ explicitly 
file_name = "transfer_learning_model_with_auto_fine_tuning.h5"
current_directory = Path(__file__).resolve().parent
# Adjust the filename and construct the new path using .\ explicitly 
file_path = current_directory / file_name 
model = tf.keras.models.load_model(str(file_path))
@app.route('/classify', methods=['POST'])
def classify_image():
    image_url = request.json['url']
    response = requests.get(image_url)
    if response.status_code == 200:
        image1 = Image.open(BytesIO(response.content))
        image1 = image1.resize((224, 224))
        image_array=image.img_to_array(image1)
        image_array=np.expand_dims(image_array,axis=0)
        image_array /=255.0
        predictions = model.predict(image_array)
        class_indices = np.argmax(predictions, axis=1)
        class_labels = ['Electrical Department','Garbage Department','Road Department','Sewage Department','Water Department']  # Replace with your actual class labels
        predicted_class = class_labels[class_indices[0]]
        predicted_prob = float(predictions[0, class_indices])
        return jsonify({'class': predicted_class, 'probability': predicted_prob})
    else:
        return jsonify({'error': 'Failed to fetch image from URL'})
if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000, debug=True)
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()