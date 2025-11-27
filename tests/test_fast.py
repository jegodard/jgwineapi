from fastapi.testclient import TestClient
from main import app

# Initialize a FastAPI client
client = TestClient(app)

# Test your root endpoint
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello stranger! This API allow you to evaluate the quality of red wine. Go to the /docs for more details."}

# Test your predict endpoint
def test_predict():
    response = client.post(
        "/predict",
        headers={'accept': 'application/json', 'Content-Type': 'application/json'},
        json={'alcohol':9.4, 'volatile_acidity': 0.7},
    )
    print(response.json())
    assert response.status_code == 200
    result = response.json()
    assert 'prediction' in result
    assert 'probability' in result
    assert result['prediction'] == 0
    assert len(result['probability']) == 2