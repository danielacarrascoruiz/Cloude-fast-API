"""Esquemas Pydantic: contrato de entrada y salida de la API."""

from typing import List
from pydantic import BaseModel, Field


class Observacion(BaseModel):
    """Una observacion del dataset Adult / Census Income."""
    age: int = Field(..., ge=17, le=100, description="Edad")
    workclass: str = Field(..., description="Tipo de empleador")
    education: str = Field(..., description="Nivel educacional")
    marital_status: str = Field(..., description="Estado civil")
    occupation: str = Field(..., description="Ocupacion")
    relationship: str = Field(..., description="Rol en el hogar")
    race: str = Field(..., description="Raza")
    sex: str = Field(..., description="Sexo")
    hours_per_week: int = Field(..., ge=1, le=99, description="Horas por semana")
    capital_gain: float = Field(0, ge=0, description="Ganancia de capital")
    capital_loss: float = Field(0, ge=0, description="Perdida de capital")
    native_country: str = Field(..., description="Pais de origen")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "workclass": "Private",
                    "education": "Bachelors",
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "hours_per_week": 40,
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "native_country": "United-States",
                }
            ]
        }
    }


class Lote(BaseModel):
    """Lista de observaciones para prediccion por lote."""
    observaciones: List[Observacion]


class RespuestaPrediccion(BaseModel):
    prediccion: int
    etiqueta: str
    probabilidad: float
    version_modelo: str
    timestamp: str
