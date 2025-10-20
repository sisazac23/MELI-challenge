""" 
evaluate.py
-----------
Evalúa el desempeño del modelo en producción usando los logs de predicciones.
Calcula métricas como RMSE, MAE y R2, y registra los resultados en MLflow.
También genera gráficos de dispersión entre predicciones y valores reales.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from datetime import datetime
import matplotlib.pyplot as plt

LOG_PATH = Path("logs") / "predictions.csv"
PLOT_DIR = Path("logs") / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_production_performance():
    if not LOG_PATH.exists():
        print(f"No existe {LOG_PATH}. Aún no hay datos para evaluar.")
        return

    df = pd.read_csv(LOG_PATH)

    # Filtrar registros que ya tengan real_price
    df_eval = df.dropna(subset=["real_price"])
    if df_eval.empty:
        print("No hay registros con 'real_price'. No se puede evaluar.")
        return

    y_true = df_eval["real_price"]
    y_pred = df_eval["predicted_price"]

    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE (prod): {rmse}")
    print(f"MAE (prod): {mae}")
    print(f"R2 (prod): {r2}")

    # Graficar (opcional)
    fig_path = PLOT_DIR / f"prod_eval_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")  # línea ideal
    plt.title("Predicción vs Valor Real (Producción)")
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.savefig(fig_path)
    plt.close()

    print(f"Gráfico guardado en: {fig_path}")

    # Registrar en MLflow
    with mlflow.start_run(run_name=f"prod_eval_{datetime.utcnow().isoformat()}"):
        mlflow.log_metric("prod_rmse", rmse)
        mlflow.log_metric("prod_mae", mae)
        mlflow.log_metric("prod_r2", r2)
        mlflow.log_artifact(str(fig_path), artifact_path="plots")

if __name__ == "__main__":
    evaluate_production_performance()
