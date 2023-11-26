from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Pour charger le model
model = load_model('final_model.h5') 

# Pour charger les classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')


# Traitement de l'image à partir de l'url
def traitement(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_data = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (224, 224)) / 255.0
            prediction = model.predict(np.array([img]))
            return label_encoder.inverse_transform([np.argmax(prediction)])[0]
        else:
            return "Impossible de décoder l'image."
    else:
        return "Echec de téléchargement de l'image."


# Requête
@app.route('/predict', methods=['POST'])
def predict_from_url():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'erreur': 'aucune url'}), 400

    image_url = data['url']
    prediction = traitement(image_url)

    return jsonify({'category': prediction})


if __name__ == '__main__':
    app.run(debug=True) 
