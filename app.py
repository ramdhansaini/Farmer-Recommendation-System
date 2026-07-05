from flask import Flask, render_template, request
import pickle, numpy as np
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "standard_scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
ferti_model_path = os.path.join(BASE_DIR, "models", "fertilizer_model.pkl")
ferti_scaler_path = os.path.join(BASE_DIR, "models", "scaler_fertilizer.pkl")
ferti_encoder_path = os.path.join(BASE_DIR, "models", "label_encoder_fertilizer.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)
with open(ferti_model_path, 'rb') as f:
    ferti_model = pickle.load(f)
with open(ferti_scaler_path, 'rb') as f:
    ferti_scaler = pickle.load(f)
with open(ferti_encoder_path, 'rb') as f:
    ferti_encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        return predict_crop()
    return render_template('crop_recommendation.html')

@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if request.method == 'POST':
        return predict_fertilizer()
    return render_template('fertilizer_recommendation.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_data_scaled = scaler.transform(input_data)
    prediction_encoded = model.predict(input_data_scaled)
    prediction = encoder.inverse_transform(prediction_encoded)

    return render_template('result.html', prediction=prediction[0])

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    crop_type = request.form['crop']
    temperature = float(request.form['temperature'])
    moisture = float(request.form['Moisture'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    carbon_content = float(request.form['Carbon'])
    soil_type = request.form['soil']
    

    input_data = np.array([[N, P, K, crop_type, temperature, moisture, ph, rainfall, carbon_content, soil_type]])
    input_data_scaled = ferti_scaler.transform(input_data)
    prediction_encoded = ferti_model.predict(input_data_scaled)
    prediction = ferti_encoder.inverse_transform(prediction_encoded)

    return render_template('result2.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
