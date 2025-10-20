from mlops_housing.train import train_and_register
from mlops_housing.config import DEFAULT_DATA_PATH
from mlops_housing.api.app import app, load_model  
from fastapi.testclient import TestClient

def test_predict_endpoint():

    #  Entrenar antes de levantar la app
    train_and_register(str(DEFAULT_DATA_PATH), tag="test_api")

    # Forzar recarga del modelo en la app 
    load_model()

    # 3) Usar context manager para respetar lifespan/startup/shutdown
    with TestClient(app) as client:
        payload = {
            "CRIM": 0.1, "ZN": 18, "INDUS": 2.3, "CHAS": 0, "NOX": 0.5,
            "RM": 6.2, "AGE": 45, "DIS": 4.2, "RAD": 1, "TAX": 300,
            "PTRATIO": 15, "B": 390, "LSTAT": 5.0
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], float)
