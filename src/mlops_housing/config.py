"""
config.py
---------
Define constantes globales del challenge MLOps
Estas constantes permiten mantener centralizadas las variables críticas
del flujo de entrenamiento, inferencia y persistencia de artefactos
"""

from pathlib import Path
from typing import List

# Nombre de las columnas de entrada utilizadas por el modelo
FEATURES: List[str] = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

# Variable objetivo
TARGET: str = "MEDV"

# Directorio donde se almacenarán los modelos entrenados 
ARTIFACTS_DIR: Path = Path("artifacts")

# Ruta por defecto del dataset 
DEFAULT_DATA_PATH: Path = Path("data/HousingData.csv")
