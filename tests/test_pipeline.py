import pandas as pd
from mlops_housing.pipeline import build_pipeline
from mlops_housing.config import FEATURES, TARGET
import numpy as np


def test_pipeline_fit_and_predict():
    """
    Test básico para asegurar que el pipeline se entrena
    y predice correctamente con una fila simple.
    """

    # Creamos un DataFrame mínimo simulado (2 filas para entrenamiento)
    data = {
        "CRIM": [0.1, 0.2],
        "ZN": [18, 0],
        "INDUS": [2.3, 7.0],
        "CHAS": [0, 1],
        "NOX": [0.5, 0.6],
        "RM": [6.2, 5.5],
        "AGE": [45, 60],
        "DIS": [4.2, 3.5],
        "RAD": [1, 2],
        "TAX": [300, 330],
        "PTRATIO": [15, 17],
        "B": [390, 380],
        "LSTAT": [5.0, 12.3],
        "MEDV": [24.0, 19.5],  # Target
    }
    df = pd.DataFrame(data)

    X = df[FEATURES]
    y = df[TARGET]

    # Construimos el pipeline
    pipeline = build_pipeline(FEATURES)

    # Entrenamos el pipeline
    pipeline.fit(X, y)

    # Probamos predicción con una fila
    test_row = X.iloc[[0]]
    prediction = pipeline.predict(test_row)

    # Validamos el resultado
    assert prediction is not None
    assert len(prediction) == 1
    assert isinstance(prediction[0], (float, np.floating))
