from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# --- Load Model and Scaler ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'lgbm_fraud_detector.pkl')
scaler_path = os.path.join(base_dir, '..', 'models', 'robust_scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# --- Define feature order ---
FEATURE_ORDER = [
    'scaled_amount', 'hour_of_day', 'V1', 'V2', 'V3', 'V4', 'V5', 
    'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 
    'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
    'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
]

# --- Add a route for the homepage ---
@app.route('/')
def home():
    """Renders the HTML file for the UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df_new = pd.DataFrame(data, index=[0])
        
        # Preprocess input
        amount_val = df_new['Amount'].values.reshape(-1, 1)
        df_new['scaled_amount'] = scaler.transform(amount_val)
        df_new['hour_of_day'] = (df_new['Time'] // 3600) % 24
        
        # Ensure all required features are present and in the correct order
        df_processed = df_new.reindex(columns=FEATURE_ORDER)
        
        # Make Prediction
        prediction = model.predict(df_processed)
        prediction_proba = model.predict_proba(df_processed)[:, 1]
        
        # Return Response
        return jsonify({
            'is_fraud': int(prediction[0]),
            'fraud_probability': float(prediction_proba[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)