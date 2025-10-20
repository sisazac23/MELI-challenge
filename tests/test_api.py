
from fastapi.testclient import TestClient
from mlops_housing.train import train_and_register
from mlops_housing.config import DEFAULT_DATA_PATH
from mlops_housing.api.app import app  


def test_predict_endpoint():
    # Entrenamos ANTES de levantar la app
    train_and_register(str(DEFAULT_DATA_PATH), tag="test_api")

    # Ahora cargamos la app (evento startup encontrará el modelo activo)
    client = TestClient(app)

    payload = {
        "CRIM": 0.1,
        "ZN": 18,
        "INDUS": 2.3,
        "CHAS": 0,
        "NOX": 0.5,
        "RM": 6.2,
        "AGE": 45,
        "DIS": 4.2,
        "RAD": 1,
        "TAX": 300,
        "PTRATIO": 15,
        "B": 390,
        "LSTAT": 5.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
