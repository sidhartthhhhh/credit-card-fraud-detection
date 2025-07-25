from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# --- Database Setup ---
def init_db():
    with sqlite3.connect('predictions.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                is_fraud INTEGER NOT NULL,
                fraud_probability REAL NOT NULL
            )
        ''')
        conn.commit()

# --- Load Model and Scaler ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'lgbm_fraud_detector.pkl')
scaler_path = os.path.join(base_dir, '..', 'models', 'robust_scaler.pkl')

model = None
def load_model():
    global model
    if model is None:
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            model = None
    return model

scaler = None

def load_scaler():
    global scaler
    if scaler is None:
        try:
            scaler = joblib.load(scaler_path)
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
