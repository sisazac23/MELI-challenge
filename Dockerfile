##########
# STAGE 1: BASE BUILDER (instala dependencias)
##########
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copiamos los archivos necesarios para la instalación
COPY pyproject.toml setup.cfg requirements.txt ./
COPY README.md ./
COPY src ./src

# Ahora, ejecutamos la instalación
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .
COPY src ./src

##########
# STAGE 2: TRAINER (CORRER ENTRENAMIENTO)
##########
FROM base AS trainer
# Aquí se podrá ejecutar el reentrenamiento en CI/CD antes de pasar a runtime
# Ejemplo de uso en CI:
# docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts trainer python -m mlops_housing.train --data_path data/... --tag retrain_xxx

##########
# STAGE 3: RUNTIME (IMAGEN FINAL)
##########
FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/bin /usr/local/bin

# Copiamos el código fuente de la aplicación
COPY src ./src

# Copiamos los artifacts (modelo entrenado) desde CI/CD o local
COPY artifacts ./artifacts

VOLUME /app/logs

EXPOSE 8000

CMD ["uvicorn", "mlops_housing.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
