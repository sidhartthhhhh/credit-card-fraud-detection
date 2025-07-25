import os
import sqlite3
from datetime import datetime

import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Define paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lgbm_fraud_detector.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'robust_scaler.pkl')

app = Flask(__name__)
CORS(app)

# Define lazy-loading variables
model = None
scaler = None

# Load model only when needed
def load_model():
    global model
    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            model = None
    return model

# Load scaler only when needed
def load_scaler():
    global scaler
    if scaler is None:
        try:
            scaler = joblib.load(SCALER_PATH)
        except FileNotFoundError:
            scaler = None
    return scaler


# --- Feature Order ---
FEATURE_ORDER = [
    'scaled_amount', 'hour_of_day', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
    'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
    'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
    'V27', 'V28'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model()
        if model is None or scaler is None:
            return jsonify({'error': 'Model or Scaler file not found'}), 500

        data = request.get_json(force=True)
        df_new = pd.DataFrame(data, index=[0])

        # Preprocess input
        amount_val = df_new['Amount'].values.reshape(-1, 1)
        df_new['scaled_amount'] = scaler.transform(amount_val)
        df_new['hour_of_day'] = (df_new['Time'] // 3600) % 24
        df_processed = df_new.reindex(columns=FEATURE_ORDER)

        # Make Prediction
        prediction = model.predict(df_processed)
        prediction_proba = model.predict_proba(df_processed)[:, 1]
        is_fraud_val = int(prediction[0])
        fraud_proba_val = float(prediction_proba[0])

        # Log to database
        with sqlite3.connect('predictions.db') as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                'INSERT INTO predictions (timestamp, is_fraud, fraud_probability) VALUES (?, ?, ?)',
                (timestamp, is_fraud_val, fraud_proba_val)
            )
            conn.commit()

        return jsonify({
            'is_fraud': is_fraud_val,
            'fraud_probability': fraud_proba_val
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    init_db()
    app.run(port=5000, debug=True, use_reloader=False)
