# Credit Card Fraud Detection System

This project is a machine learning system to detect fraudulent credit card transactions. It uses a LightGBM classifier trained on a highly imbalanced dataset from Kaggle.

## Project Structure

credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
├── models/
│   ├── lgbm_fraud_detector.pkl
│   └── robust_scaler.pkl
├── notebooks/
│   └── 1_eda_and_modeling.ipynb
├── src/
│   ├── train.py
│   └── app.py
├── .gitignore
├── README.md
└── requirements.txt


## How to Run

### 1. Setup

First, clone the repository and install the required dependencies.

```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
pip install -r requirements.txt

2. Download Data
Download the creditcard.csv dataset from Kaggle and place it inside the data/ directory.

3. Train the Model
Run the training script to process the data, train the LightGBM model, and save the artifacts (model and scaler) to the models/ directory.


# Navigate to the src directory
cd src

# Run the training script
python train.py

4. Start the Prediction API
Once the model is trained, start the Flask API to serve predictions.

# From the src directory
python app.py
The API will be available at http://127.0.0.1:5000.

5. Make a Prediction
You can send a POST request to the /predict endpoint with transaction data to get a fraud prediction.

Example using cURL:
curl -X POST -H "Content-Type: application/json" \
-d '{
    "Time": 86300,
    "V1": -0.5, "V2": 1.2, "V3": 1.5, "V4": -0.3, "V5": 0.8,
    "V6": -0.1, "V7": 0.4, "V8": 0.2, "V9": -0.6, "V10": -0.4,
    "V11": 1.6, "V12": -1.8, "V13": 0.3, "V14": -0.2, "V15": -1.1,
    "V16": 0.4, "V17": 0.1, "V18": 0.2, "V19": -0.0, "V20": 0.1,
    "V21": -0.2, "V22": -0.5, "V23": 0.1, "V24": 0.4, "V25": -0.1,
    "V26": 0.1, "V27": 0.2, "V28": 0.1,
    "Amount": 7.99
}' \
[http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)