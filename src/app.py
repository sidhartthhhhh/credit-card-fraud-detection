import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lgbm_fraud_detector.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'robust_scaler.pkl')

app = Flask(__name__)

model = None
scaler = None

FEATURE_ORDER = [
    'scaled_amount', 'hour_of_day', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
    'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17'
]

def load_dependencies():
    global model, scaler
    if model is None or scaler is None:
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        except FileNotFoundError:
            model = None
            scaler = None
    return model, scaler

@app.route('/')
def home():
    return "<h1>Fraud Detection API</h1><p>Send a POST request to /predict</p>"

@app.route('/predict', methods=['POST'])
def predict():
    model, scaler = load_dependencies()
    if not all([model, scaler]):
        return jsonify({'error': 'Model or scaler not found on server.'}), 500

    try:
        transaction_data = request.get_json(force=True)
        df = pd.DataFrame(transaction_data, index=[0])

        df['scaled_amount'] = scaler.transform(df[['Amount']])
        df['hour_of_day'] = (df['Time'] // 3600) % 24

        for col in FEATURE_ORDER:
            if col not in df.columns:
                df[col] = 0
        
        processed_df = df[FEATURE_ORDER]

        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)[:, 1]

        return jsonify({
            'is_fraud': int(prediction[0]),
            'fraud_probability': float(probability[0])
        })
    except KeyError as e:
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
