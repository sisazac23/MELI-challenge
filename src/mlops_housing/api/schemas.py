"""
schemas.py
----------
Define los esquemas de entrada y salida para la API de predicción.
Usa Pydantic para validar y tipar los datos de entrada.
"""

from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    CRIM: float = Field(..., description="Tasa de criminalidad per cápita en la ciudad.")
    ZN: float = Field(..., description="Proporción de terrenos residenciales zonificados para lotes grandes.")
    INDUS: float = Field(..., description="Proporción de acres de negocios no minoristas por ciudad.")
    CHAS: Literal[0,1] = Field(..., description="Dummy Charles River (1 si limita con el río, 0 en caso contrario).")
    NOX: float = Field(..., description="Concentración de óxidos de nitrógeno (partes por 10 millones).")
    RM: float = Field(..., description="Número promedio de habitaciones por vivienda.")
    AGE: float = Field(..., description="Proporción de unidades ocupadas por sus propietarios construidas antes de 1940.")
    DIS: float = Field(..., description="Distancias ponderadas a cinco centros de empleo en Boston.")
    RAD: float = Field(..., description="Índice de accesibilidad a autopistas radiales.")
    TAX: float = Field(..., description="Tasa de impuesto a la propiedad por $10,000.")
    PTRATIO: float = Field(..., description="Relación alumno-profesor en la ciudad.")
    B: float = Field(..., description="1000*(Bk - 0.63)^2 donde Bk es proporción de población afroamericana.")
    LSTAT: float = Field(..., description="% de población con bajo estatus socioeconómico.")

class PredictResponse(BaseModel):
    predicted_price: float = Field(..., description="Precio estimado de la vivienda en miles de dólares.")
