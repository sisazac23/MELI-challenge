import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
from mlops_housing.registry import load_current

LOG_PATH = Path("logs") / "predictions.csv"
PLOT_DIR = Path("logs") / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# env helpers
def _envfloat(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _envint(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


# debug mode
EVAL_DEBUG = os.getenv("EVAL_DEBUG", "0") == "1"

def _dbg(msg: str):
    if EVAL_DEBUG:
        print(f"[DEBUG] {msg}")


# timsetamp normalization
def _normalize_timestamp_series(ts: pd.Series) -> pd.Series:
    """
    Convierte la columna timestamp a datetime naive en UTC (sin TZ).
    Parseamos con errors='coerce' y luego removemos tz.
    """
    parsed = pd.to_datetime(ts, errors="coerce", utc=True)
    naive_utc = parsed.dt.tz_localize(None)
    return naive_utc


def _make_cutoff(days: int) -> pd.Timestamp:
    """
    Genera un cutoff naive (UTC sin tz).
    """
    return pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)


# dated filtering
def _debug_timestamps_block(df: pd.DataFrame, cutoff: pd.Timestamp):
    _dbg(f"df.dtypes:\n{df.dtypes}")
    if "timestamp" in df.columns:
        _dbg(f"timestamp head (raw):\n{df['timestamp'].head(5)}")
        _dbg(f"timestamp.isna().sum()={df['timestamp'].isna().sum()}")
    _dbg(f"cutoff type={type(cutoff)} value={cutoff!r}")


def _filter_window(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise RuntimeError("El log no tiene columna 'timestamp'.")

    df = df.copy()
    df["timestamp"] = _normalize_timestamp_series(df["timestamp"])
    cutoff = _make_cutoff(window_days)

    if EVAL_DEBUG:
        _debug_timestamps_block(df, cutoff)

    valid = df["timestamp"].notna() & df["real_price"].notna()
    df_valid = df.loc[valid]
    _dbg(f"valid rows (ts OK and real_price OK): {len(df_valid)}/{len(df)}")

    try:
        mask = df_valid["timestamp"] >= cutoff
    except Exception as e:
        _dbg(f"FAILED COMPARISON: ts dtype={df_valid['timestamp'].dtype}, cutoff={type(cutoff)}")
        _dbg(f"sample ts types: {df_valid['timestamp'].map(type).head(5).tolist()}")
        raise

    df_window = df_valid.loc[mask].copy()
    _dbg(f"rows in window (>=cutoff): {len(df_window)}")
    return df_window

def evaluate_and_decide() -> int:
    if not LOG_PATH.exists():
        print(f"[evaluate] No existe {LOG_PATH}. Por lo que no se evalúa si hay degradación del modelo")
        return 0

    df = pd.read_csv(LOG_PATH)

    if "real_price" not in df.columns:
        print("[evaluate] No hay columna real_price aún.")
        return 0

    window_days = _envint("EVAL_WINDOW_DAYS", 1)
    min_feedback = _envint("EVAL_MIN_FEEDBACK", 20)
    threshold = _envfloat("EVAL_THRESHOLD", 0.10)

    df_window = _filter_window(df, window_days)
    n_feedback = len(df_window)

    if n_feedback < min_feedback:
        print(f"[evaluate] Feedback insuficiente: {n_feedback} < {min_feedback}.")
        return 0

    y_true = df_window["real_price"].astype(float)
    y_pred = df_window["predicted_price"].astype(float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    _, run_dir = load_current()
    with open(Path(run_dir) / "metrics.json", "r", encoding="utf-8") as f:
        base_metrics = json.load(f)
    baseline_rmse = float(base_metrics.get("cv_rmse", float("nan")))

    degraded = rmse > baseline_rmse * (1.0 + threshold)

    fig_path = PLOT_DIR / f"prod_eval_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.title(f"Últimos {window_days} días | RMSE={rmse:.3f} | baseline={baseline_rmse:.3f}")
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"[evaluate] n={n_feedback}, rmse={rmse:.3f}, baseline={baseline_rmse:.3f}, degraded={degraded}")
    
    with mlflow.start_run(run_name=f"prod_eval_{datetime.utcnow().isoformat()}"):
        mlflow.log_param("window_days", window_days)
        mlflow.log_param("min_feedback", min_feedback)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("prod_rmse", rmse)
        mlflow.log_metric("prod_mae", mae)
        mlflow.log_metric("prod_r2", r2)
        mlflow.log_metric("baseline_rmse", baseline_rmse)
        mlflow.log_metric("degraded_flag", 1.0 if degraded else 0.0)
        mlflow.log_artifact(str(fig_path), artifact_path="plots")

    return 2 if degraded else 0


if __name__ == "__main__":
    exit(evaluate_and_decide())
