"""
app.py
------
API REST para predicción de precios de viviendas usando FastAPI.
Carga automáticamente el modelo activo desde artifacts/version.json.
"""

from fastapi import FastAPI, HTTPException, status
from loguru import logger
import pandas as pd

from .schemas import PredictRequest, PredictResponse
from src.mlops_housing.registry import load_current  # Carga el modelo entrenado
from src.mlops_housing.config import FEATURES

# Crear instancia de FastAPI
app = FastAPI(
    title="Housing Price Prediction API",
    version="1.0.0",
    description="API para predecir precios de viviendas entrenada con RandomForest y registrada con MLflow."
)

# Variable global del modelo
MODEL = None


@app.on_event("startup")
def load_model():
    """
    Intenta cargar el modelo en el arranque de la API.
    Si falla, deja un flag indicando que el modelo no está disponible.
    """
    global MODEL, MODEL_LOADED
    try:
        MODEL, run_dir = load_current()
        MODEL_LOADED = True
        logger.info(f"Modelo cargado desde: {run_dir}")
    except Exception as e:
        MODEL_LOADED = False
        logger.error(f"Error al cargar el modelo: {str(e)}")


@app.get("/healthz")
def health():
    """
    Endpoint de verificación del estado de la API.
    Retorna un mensaje simple para comprobar disponibilidad.
    """
    return {"status": "ok", "message": "API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Entrene un modelo antes de predecir."
        )
    
    try:
        feature_values = [getattr(payload, feature) for feature in FEATURES]
        X_input = pd.DataFrame([feature_values], columns=FEATURES)

        pred = MODEL.predict(X_input)[0]
        pred = round(float(pred), 2)
        return PredictResponse(predicted_price=float(pred))

    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hubo un error procesando la predicción. Verifica los datos o intenta nuevamente."
        )
