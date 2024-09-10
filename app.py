from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import librosa
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import soundfile as sf
import pandas as pd
from werkzeug.utils import secure_filename
import os
import logging
from sklearn.preprocessing import LabelEncoder

cls = joblib.load('police.pkl')
en = joblib.load('label_encoder.pkl')  



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained model
model = joblib.load('human_vs_animal.pkl')

# Load crime data
df2 = pd.read_csv('districtwise-crime-against-women (1).csv')
df2 = df2[['registeration_circles', 'total_crime_against_women']]

# Define function to classify crime alert
def crime_indicator(crime_count):
    if crime_count < 50:
        return '🟢Green'
    elif 50 <= crime_count <= 500:
        return '🟡Yellow'
    else:
        return '🔴Red'

# Apply classification to crime data
df2['indicator'] = df2['total_crime_against_women'].apply(crime_indicator)

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/nearestPoliceStation', methods=['POST'])
def nearest_police_station():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    # Predict the nearest police station using the trained model
    try:
        nearest_station = en.inverse_transform(cls.predict([[latitude, longitude]]))
        return jsonify({'police_station': nearest_station[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emergency', methods=['POST'])
def emergency():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    address = data.get('address')
    
    # Log received location and address
    logger.info(f'Received emergency location: Latitude {latitude}, Longitude {longitude}, Address {address}')
    
    return jsonify({'status': 'success', 'latitude': latitude, 'longitude': longitude, 'address': address})

@app.route('/getCrimeAlert', methods=['GET'])
def get_crime_alert():
    city = request.args.get('city')
    crime_alert = 'low'  # Default value
    for i in range(len(df2)):
        if city.lower() in df2['registeration_circles'][i].lower():
            crime_alert = df2['indicator'][i]
            break
    return jsonify({'alert': crime_alert})

if __name__ == '__main__':
    app.run(debug=True)
