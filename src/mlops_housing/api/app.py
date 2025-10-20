"""
app.py
------
API REST para predicción de precios de viviendas usando FastAPI.
Carga automáticamente el modelo activo desde artifacts/version.json.
"""

from fastapi import FastAPI, HTTPException, status, Response
from loguru import logger
import pandas as pd
import json
from pathlib import Path
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from datetime import datetime
import uuid

from .schemas import PredictRequest, PredictResponse, FeedbackRequest
from mlops_housing.registry import load_current  # Carga el modelo entrenado
from mlops_housing.config import FEATURES

# Crear instancia de FastAPI
app = FastAPI(
    title="Housing Price Prediction API",
    version="1.0.0",
    description="API para predecir precios de viviendas entrenada con RandomForest y registrada con MLflow."
)

# Variable global del modelo
MODEL = None
MODEL_LOADED = False

# Métricas Prometheus
PRED_COUNTER = Counter("pred_requests_total", "Total de requests a /predict")
PRED_LATENCY = Histogram("pred_latency_seconds", "Latencia de /predict en segundos")

# Logs
LOG_PATH = Path("logs") / "predictions.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


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

@app.get("/version")
def version():
    try:
        _, run_dir = load_current()
        metrics_path = Path(run_dir) / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {"run_dir": str(run_dir), "metrics": metrics}
    except Exception as e:
        logger.error(f"Error leyendo versión/metrics: {e}")
        raise HTTPException(status_code=500, detail="No se pudo leer versión actual")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Entrene un modelo antes de predecir."
        )

    start_time = time.time()  
    try:
        PRED_COUNTER.inc()  

        # Extraer valores en el orden correcto
        feature_values = [getattr(payload, feature) for feature in FEATURES]
        X_input = pd.DataFrame([feature_values], columns=FEATURES)

        # Hacer predicción
        pred = MODEL.predict(X_input)[0]
        pred_float = round(float(pred), 3)
        

        # Generar ID
        prediction_id = str(uuid.uuid4())

        # Loggear predicción
        row = {
            "id": prediction_id,
            "timestamp": datetime.utcnow().isoformat(),
            **payload.dict(),
            "predicted_price": pred_float,
            "real_price": None
        }

        df_log = pd.DataFrame([row])
        if LOG_PATH.exists():
            df_log.to_csv(LOG_PATH, mode="a", header=False, index=False)
        else:
            df_log.to_csv(LOG_PATH, index=False)

        # Devolver resultado
        return PredictResponse(
            predicted_price=pred_float,
            id=prediction_id  
        )

    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hubo un error procesando la solicitud."
        )
    finally:
         PRED_LATENCY.observe(time.time() - start_time)  


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    try:
        df = pd.read_csv(LOG_PATH)

        if payload.id not in df["id"].values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="ID no encontrado en el registro de predicciones."
            )

        df.loc[df["id"] == payload.id, "real_price"] = payload.real_price
        df.to_csv(LOG_PATH, index=False)

        return {"message": "Valor real actualizado correctamente"}

    except Exception as e:
        logger.error(f"Error al procesar feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo actualizar el valor real."
        )

