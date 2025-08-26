import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, average_precision_score, recall_score, precision_score
import joblib
import os
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

def train_model():
    print("Starting training process...")

    data_path = 'data/creditcard.csv'
    

    # Load Data
    df = pd.read_csv(data_path)

    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['hour_of_day'] = (df['Time'] // 3600) % 24
    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



    # --- Start MLflow Run ---
    mlflow.set_experiment("Credit Card Fraud Detection")
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        print("Training LightGBM model...")
        model = LGBMClassifier(objective='binary', metric='auc', n_estimators=1000, random_state=42)
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auprc = average_precision_score(y_test, y_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        print("\n--- Model Evaluation on Test Set ---")
        print(classification_report(y_test, y_pred))
        print(f"Area Under PR Curve: {auprc:.4f}")

        print("Logging experiment to MLflow...")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_estimators", 1000)

        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("recall_fraud", recall)
        mlflow.log_metric("precision_fraud", precision)

        mlflow.lightgbm.log_model(model, "lgbm_model")
        mlflow.sklearn.log_model(scaler, "robust_scaler")

        print("\nMLflow logging complete.")
        print("To view your experiments, run 'mlflow ui --port 5001' in your terminal.")

if __name__ == '__main__':
    train_model()