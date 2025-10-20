### Reto técnico MercadoLibre: MLOPs#  MLOPS Housing Price Predictor – Pipeline de Producción con CI/CD y Docker

Este proyecto implementa un flujo completo de **MLOps** para entrenar, versionar, desplegar, evaluar y reentrenar automáticamente un modelo de predicción de precios de viviendas basado en un dataset tipo Housing (Boston dataset como ejemplo académico).

El objetivo es simular un entorno **real de producción**, incorporando:

- Entrenamiento reproducible  
- API REST con FastAPI  
- Versionado y trazabilidad de modelos (MLflow + artifacts persistentes)  
- Registro de predicciones y recolección de feedback  
- Evaluación automática del rendimiento en producción  
- Reentrenamiento automático (CI/CD)  
- Construcción y publicación de imágenes Docker inmutables  
- Pipeline CI/CD completo con GitHub Actions  

---

## Tecnologías utilizadas

| Área | Herramienta |
|------|------------|
| Lenguaje | Python 3.11 |
| Framework API | FastAPI + Uvicorn |
| Modelos ML | Scikit-learn |
| Trazabilidad | MLflow |
| Contenerización | Docker (multi-stage) |
| CI/CD | GitHub Actions |
| Métricas | Compatible con Prometheus |
| Validación de inputs | Pydantic |
| Logging & Feedback | CSV persistente o base centralizable |

---

## Estructura del proyecto

```plaintext
MELI-challenge/
├── src/mlops_housing/
│   ├── pipeline.py         # Construcción del pipeline de ML
│   ├── train.py            # Entrenamiento y registro del modelo
│   ├── registry.py         # Gestión de versiones del modelo
│   ├── api/app.py          # API REST FastAPI
│   ├── schemas.py          # Validación de datos de entrada
│   ├── evaluate.py         # Evaluación del modelo en producción
│   └── ...
├── artifacts/              # Modelo activo en producción
├── logs/                   # Feedback y predicciones en runtime
├── data/                   # Dataset base
├── Dockerfile
├── requirements.txt
└── .github/workflows/
    ├── ci.yml              # Validación + build Docker
    ├── evaluate.yml        # Evalúa degradación y dispara retrain
    ├── retrain.yml         # Reentrena modelo y sube artifacts
    └── build_and_push.yml  # Genera imagen con nuevo modelo
