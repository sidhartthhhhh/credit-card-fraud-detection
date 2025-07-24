import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, average_precision_score
import joblib
import os

def train_model():
    """
    Trains the fraud detection model and saves the artifacts.
    """
    print("Starting training process...")

    # Define paths
    data_path = '../data/creditcard.csv'
    model_dir = '../models'
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load Data
    df = pd.read_csv(data_path)

    # 2. Preprocess Data
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['hour_of_day'] = (df['Time'] // 3600) % 24
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Handle Imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Train Model
    print("Training LightGBM model...")
    model = LGBMClassifier(objective='binary', metric='auc', n_estimators=1000, random_state=42)
    model.fit(X_train_res, y_train_res)

    # 6. Evaluate Model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation on Test Set ---")
    print(classification_report(y_test, y_pred))
    print(f"Area Under PR Curve: {average_precision_score(y_test, y_proba):.4f}")

    # 7. Save Artifacts
    model_path = os.path.join(model_dir, 'lgbm_fraud_detector.pkl')
    scaler_path = os.path.join(model_dir, 'robust_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print("\nTraining process finished.")


if __name__ == '__main__':
    train_model()