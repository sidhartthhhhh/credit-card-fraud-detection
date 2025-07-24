import pytest
import json
from src.app import app as flask_app # Import your Flask app

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    yield flask_app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_predict_missing_field(client):
    """
    Test the /predict endpoint with a payload missing the 'Amount' field.
    """
    # ARRANGE: Define a payload that is missing the "Amount" key
    bad_data = {
        "Time": 86300, "V1": -0.5, "V2": 1.2, "V3": 1.5, "V4": -0.3, "V5": 0.8,
        "V6": -0.1, "V7": 0.4, "V8": 0.2, "V9": -0.6, "V10": -0.4, "V11": 1.6,
        "V12": -1.8, "V13": 0.3, "V14": -0.2, "V15": -1.1, "V16": 0.4, "V17": 0.1,
        "V18": 0.2, "V19": -0.0, "V20": 0.1, "V21": -0.2, "V22": -0.5, "V23": 0.1,
        "V24": 0.4, "V25": -0.1, "V26": 0.1, "V27": 0.2, "V28": 0.1
        # "Amount" key is intentionally missing
    }

    # ACT: Send a POST request
    response = client.post(
        '/predict',
        data=json.dumps(bad_data),
        content_type='application/json'
    )

    # ASSERT: Check that the server returns a 400 Bad Request error
    assert response.status_code == 400
    response_data = response.get_json()
    assert 'error' in response_data # Check if an error message is in the response