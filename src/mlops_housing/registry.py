"""
registry.py
-----------
Encargado de guardar y cargar el modelo entrenado en producción.
Utiliza un enfoque simple basado en filesystem y version.json.
Para usarse junto a MLflow como sistema de tracking.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
from .config import ARTIFACTS_DIR

VERSION_FILE = ARTIFACTS_DIR / "version.json"

def save_run(model: Any, metrics: Dict[str, float], tag: str) -> Path:
    """
    Guarda el modelo entrenado y sus métricas en una carpeta única (timestamp + tag).
    Actualiza 'version.json' para indicar la versión activa.
    
    Args:
        model: Pipeline entrenado (sklearn)
        metrics: Diccionario con métricas registradas (e.g., RMSE, R2)
        tag: Etiqueta (ejemplo: "rf_v1")

    Returns:
        Path del directorio del run generado.
    """
    # Generar nombre de versión basado en timestamp UTC
    run_dir = ARTIFACTS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    joblib.dump(model, run_dir / "model.joblib")

    # Guardar métricas
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Registrar versión activa
    VERSION_FILE.write_text(json.dumps({"current": str(run_dir)}, indent=2), encoding="utf-8")

    return run_dir


def load_current() -> Tuple[Any, Path]:
    """
    Carga el modelo actualmente activo según 'version.json'.

    Returns:
        model: Modelo/Pipeline sklearn cargado
        run_dir: Directorio de la versión activa

    Raises:
        FileNotFoundError si no existe un modelo registrado
    """
    if not VERSION_FILE.exists():
        raise FileNotFoundError(
            "No se encontró 'version.json'. Debes entrenar y registrar un modelo primero."
        )
    
    meta = json.loads(VERSION_FILE.read_text(encoding="utf-8"))
    run_dir = Path(meta["current"])

    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"El modelo no se encontró en {model_path}")

    model = joblib.load(model_path)
    return model, run_dir
