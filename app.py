import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Define the paths explicitly
model_path = r'C:\Users\anukr\OneDrive\Documents\Anu\Final year project\Final Minor Project\random_forest_model.pkl'
scaler_path = r'C:\Users\anukr\OneDrive\Documents\Anu\Final year project\Final Minor Project\scaler.pkl'
columns_path = r'C:\Users\anukr\OneDrive\Documents\Anu\Final year project\Final Minor Project\model_columns.pkl'

# Load the trained model, scaler, and feature columns
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_columns = joblib.load(columns_path)  # Load the saved columns

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    price = float(request.form['Price'])
    ram = float(request.form['RAM'])
    rom = float(request.form['ROM'])
    height = float(request.form['Height'])
    width = float(request.form['Width'])
    main_camera = float(request.form['Main_Cam'])
    extra_camera = float(request.form['Extra_Cam'])
    front_camera = float(request.form['Front_Cam'])
    battery = float(request.form['Battery'])

    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'Price': [price],
        'RAM': [ram],
        'ROM': [rom],
        'Height': [height],
        'Width': [width],
        'Main_Cam': [main_camera],
        'Extra_Cam': [extra_camera],
        'Front_Cam': [front_camera],
        'Battery': [battery],
        'Color_Black': [1 if request.form['Color'] == 'Black' else 0],
        'Color_White': [1 if request.form['Color'] == 'White' else 0],
        'Processor_MediaTek': [1 if request.form['Processor'] == 'MediaTek' else 0],
        'Processor_Qualcomm': [1 if request.form['Processor'] == 'Qualcomm' else 0],
    })

    # Create dummy variables and align input with the model's expected input
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Scale the features for prediction
    features_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(features_scaled)

    return jsonify({'predicted_review': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
