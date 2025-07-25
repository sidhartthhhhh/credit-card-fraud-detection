import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from src.app import app as flask_app

# --- Define paths ---
# These paths assume a 'models' directory exists at the same level as the 'src' directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lgbm_fraud_detector.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'robust_scaler.pkl')

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Define lazy-loading variables ---
# These are kept in memory as None until the first prediction request
model = None
scaler = None

# --- Feature Order (ensure this matches your model's training) ---
FEATURE_ORDER = [
    'scaled_amount', 'hour_of_day', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
    'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17'
    # NOTE: Ensure you have all the columns your model expects! 
    # This list seems shorter than the standard 28 'V' columns.
]

# --- Lazy Loading Function ---
# This function loads the model and scaler into memory on the first call
# and then reuses them for subsequent calls.
def load_model_and_scaler():
    """
    Loads the model and scaler from disk if they haven't been loaded yet.
    Uses global variables to cache them in memory.
    """
    global model, scaler
    if model is None or scaler is None:
        print("Loading model and scaler for the first time...")
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and scaler loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model/scaler: {e}")
            # In a real app, you would log this error more formally
            model = None
            scaler = None
    return model, scaler

# --- Routes ---

@app.route('/')
def home():
    """
    A simple home route. For a real application, you would render a form here.
    """
    # Note: This requires a 'templates' folder with an 'index.html' file.
    # If you don't have it, this route will error, but the /predict route will still work.
    return "<h1>Fraud Detection API</h1><p>Send a POST request to /predict</p>"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives transaction data, preprocesses it, and returns a fraud prediction.
    """
    try:
        # Lazy load the model and scaler
        model, scaler = load_model_and_scaler()
        
        # Check if loading failed
        if model is None or scaler is None:
            return jsonify({'error': 'Model or Scaler file not found on the server. Ensure they exist in the /models directory.'}), 500

        # Get data from the POST request
        data = request.get_json(force=True)
        df_new = pd.DataFrame(data, index=[0])

        # --- Preprocess Input Data ---
        # 1. Scale the transaction amount
        amount_val = df_new['Amount'].values.reshape(-1, 1)
        df_new['scaled_amount'] = scaler.transform(amount_val)
        
        # 2. Extract the hour of the day from the 'Time' feature
        df_new['hour_of_day'] = (df_new['Time'] // 3600) % 24
        
        # 3. Ensure all required columns from FEATURE_ORDER are present, fill missing with 0
        for col in FEATURE_ORDER:
            if col not in df_new.columns:
                df_new[col] = 0
        
        # 4. Reorder columns to match the model's training order
        df_processed = df_new[FEATURE_ORDER]

        # --- Make Prediction ---
        prediction = model.predict(df_processed)
        prediction_proba = model.predict_proba(df_processed)[:, 1]

        # --- Return Response ---
        return jsonify({
            'is_fraud': int(prediction[0]),
            'fraud_probability': float(prediction_proba[0])
        })

    except KeyError as e:
        return jsonify({'error': f'Missing required field in input data: {e}'}), 400
    except Exception as e:
        print(f"An error occurred: {e}") # Log the full error to the console
        return jsonify({'error': 'An internal error occurred. Check the server logs.'}), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    # The app runs on port 5000 and is accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)

