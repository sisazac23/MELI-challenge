from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Variables fijas

features : List[str] = [ "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

target = "MEDV"

def load_dataset(csv_path : str | Path) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV (Boston Housing).

    Args:
        csv_path (str | Path): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados de Boston Housing.
    """

    df = pd.read_csv(csv_path)
    missing = set(features + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")
    return df

def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en conjuntos de entrenamiento y prueba.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        test_size (float, optional): Proporción del conjunto de prueba. Defaults to 0.2.
        random_state (int, optional): Semilla para la aleatoriedad. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Conjuntos de entrenamiento y prueba.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test