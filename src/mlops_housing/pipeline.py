"""
pipeline.py
-----------
Contiene la función que construye el pipeline completo del modelo.
Incluye las etapas de preprocesamiento e inferencia en un único objeto sklearn.
"""

from __future__ import annotations
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

def build_pipeline(features: List[str]) -> Pipeline:
    """
    Construye un pipeline de sklearn con:
      1. Imputador de mediana para cada feature numérica
      2. Modelo RandomForestRegressor

    Args:
        features: Lista de columnas (features) que se usarán para el modelo.

    Returns:
        Pipeline completamente configurado (listo para .fit() / .predict()).
    """

    # Preprocesador: imputación numérica
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), features)
        ],
        remainder="drop",  # Descarta cualquier columna no especificada
        verbose_feature_names_out=False
    )

    # Modelo
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Pipeline completo
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
