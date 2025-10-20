import pandas as pd
from mlops_housing.registry import load_current
from mlops_housing.config import FEATURES


def test_load_current_model():
    """
    Verifica que load_current() carga un modelo válido.
    También prueba que el modelo puede predecir una fila dummy.
    """
    model, run_dir = load_current()
    
    # Creamos una fila dummy con ceros o valores típicos
    dummy_data = pd.DataFrame([[0 for _ in FEATURES]], columns=FEATURES)

    prediction = model.predict(dummy_data)

    assert prediction is not None
    assert len(prediction) == 1
    assert isinstance(prediction[0], float)
