"""
train.py
--------
Entrena el modelo con el pipeline de producción, registra métricas en MLflow
y guarda la versión final del modelo en artifacts/.
"""

from __future__ import annotations
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger
import mlflow
import mlflow.sklearn

from .config import FEATURES, TARGET, DEFAULT_DATA_PATH
from .pipeline import build_pipeline
from .registry import save_run


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series, k: int = 5) -> Tuple[float, float]:
    """
    Calcula RMSE y R2 mediante validación cruzada.
    """
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv).mean()
    r2 = cross_val_score(model, X, y, scoring="r2", cv=cv).mean()
    return float(rmse), float(r2)


def train_and_register(data_path: str, tag: str = "rf") -> Dict[str, float]:
    """
    Pipeline completo de entrenamiento con MLflow:
      1. Cargar datos
      2. CV para métricas
      3. Entrenar sobre todo el dataset
      4. Logear en MLflow
      5. Guardar artefacto en artifacts/
    """
    logger.info(f"Cargando dataset desde: {data_path}")
    df = pd.read_csv(data_path)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    logger.info("Construyendo pipeline de producción...")
    model = build_pipeline(FEATURES)

    with mlflow.start_run():
        # Evaluación con CV previa al entrenamiento final
        logger.info("Ejecutando validación cruzada (CV)")
        cv_rmse, cv_r2 = evaluate_cv(model, X, y, k=5)

        # Entrenar sobre todo el dataset
        logger.info("Entrenando modelo final con todo el dataset")
        model.fit(X, y)
        y_hat = model.predict(X)

        # Métricas sobre train completo (solo para monitoreo, no para selección)
        train_rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
        train_r2 = float(r2_score(y, y_hat))

        metrics = {
            "cv_rmse": cv_rmse,
            "cv_r2": cv_r2,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
        }

        logger.info(f"Métricas finales: {metrics}")

        # Log en MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)

        # Guardamos el pipeline completo con MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Guardamos también en artifacts/ para que lo cargue la API fácilmente
        logger.info("Guardando artefactos en carpeta local artifacts/")
        run_dir = save_run(model, metrics, tag=tag)

    logger.success(f"Entrenamiento completado. Artefactos guardados en: {run_dir}")
    return metrics


def cli():
    parser = argparse.ArgumentParser(description="Entrena y registra un modelo RandomForest")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--tag", type=str, default="rf")
    args = parser.parse_args()

    train_and_register(args.data_path, args.tag)


if __name__ == "__main__":
    cli()
