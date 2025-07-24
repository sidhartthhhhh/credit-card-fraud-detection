from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- Load Model and Scaler ---
# Construct paths relative to the current file
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'lgbm_fraud_detector.pkl')
scaler_path = os.path.join(base_dir, '..', 'models', 'robust_scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# --- Define feature order ---
# This list must match the columns the model was trained on
# (excluding the target variable 'Class')
FEATURE_ORDER = [
    'scaled_amount', 'hour_of_day', 'V1', 'V2', 'V3', 'V4', 'V5', 
    'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 
    'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
    'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)
        
        # --- Preprocess input ---
        # The input JSON must contain 'Amount' and 'Time' along with V1-V28
        df_new = pd.DataFrame(data, index=[0])
        
        # Apply scaling to 'Amount'
        amount_val = df_new['Amount'].values.reshape(-1, 1)
        df_new['scaled_amount'] = scaler.transform(amount_val)
        
        # Create 'hour_of_day' from 'Time'
        df_new['hour_of_day'] = (df_new['Time'] // 3600) % 24
        
        # Ensure all required features are present and in the correct order
        df_processed = df_new.reindex(columns=FEATURE_ORDER)
        
        # --- Make Prediction ---
        prediction = model.predict(df_processed)
        prediction_proba = model.predict_proba(df_processed)[:, 1]
        
        # --- Return Response ---
        return jsonify({
            'is_fraud': int(prediction[0]),
            'fraud_probability': float(prediction_proba[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # To test, use a command like:
    # curl -X POST -H "Content-Type: application/json" -d '{"Time": 406, "V1": -2.31, ..., "V28": -0.15, "Amount": 3.99}' http://127.0.0.1:5000/predict
    app.run(port=5000, debug=True)