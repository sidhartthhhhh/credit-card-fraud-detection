import pytest
import json
import os
from src.app import app as flask_app, init_db # Import your Flask app and init_db

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    
    # --- SETUP ---
    # This ensures the database and table are created before tests run
    with flask_app.app_context():
        init_db()
    
    yield flask_app
    
    # --- TEARDOWN ---
    # Optional: clean up the database file after tests are done
    # This runs in the root directory, so it won't touch the one in src/
    if os.path.exists('predictions.db'):
        os.remove('predictions.db')


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_predict_endpoint_non_fraud(client):
    """
    Test the /predict endpoint with a sample known non-fraudulent transaction.
    """
    # ARRANGE: Define a sample payload
    sample_data = {
        "Time": 86300, "V1": -0.5, "V2": 1.2, "V3": 1.5, "V4": -0.3, "V5": 0.8,
        "V6": -0.1, "V7": 0.4, "V8": 0.2, "V9": -0.6, "V10": -0.4, "V11": 1.6,
        "V12": -1.8, "V13": 0.3, "V14": -0.2, "V15": -1.1, "V16": 0.4, "V17": 0.1,
        "V18": 0.2, "V19": -0.0, "V20": 0.1, "V21": -0.2, "V22": -0.5, "V23": 0.1,
        "V24": 0.4, "V25": -0.1, "V26": 0.1, "V27": 0.2, "V28": 0.1, "Amount": 7.99
    }

    # ACT: Send a POST request to the endpoint
    response = client.post(
        '/predict',
        data=json.dumps(sample_data),
        content_type='application/json'
    )

    # ASSERT: Check if the response is correct
    assert response.status_code == 200 # This should now pass
    response_data = response.get_json()
    assert response_data['is_fraud'] == 0
    assert 'fraud_probability' in response_data
    assert isinstance(response_data['fraud_probability'], float)