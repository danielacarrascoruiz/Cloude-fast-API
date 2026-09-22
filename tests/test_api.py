"""Pruebas automatizadas de la API con TestClient de FastAPI.

Ejecutar desde la raiz del proyecto:
    pytest

Requiere que model/model.pkl exista (corre antes:  python train.py).
"""

from fastapi.testclient import TestClient
from app.main import app

OBS_VALIDA = {
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


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True


def test_predict_ok():
    with TestClient(app) as client:
        r = client.post("/predict", json=OBS_VALIDA)
        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["prediccion"] in (0, 1)
        assert 0.0 <= cuerpo["probabilidad"] <= 1.0


def test_predict_invalido_422():
    with TestClient(app) as client:
        mala = dict(OBS_VALIDA)
        mala["age"] = "cuarenta"          # tipo incorrecto
        del mala["workclass"]              # campo faltante
        r = client.post("/predict", json=mala)
        assert r.status_code == 422


def test_predict_batch():
    with TestClient(app) as client:
        payload = {"observaciones": [OBS_VALIDA, OBS_VALIDA]}
        r = client.post("/predict-batch", json=payload)
        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["n"] == 2
        assert len(cuerpo["predicciones"]) == 2


def test_model_info():
    with TestClient(app) as client:
        r = client.get("/model-info")
        assert r.status_code == 200
        assert "variables_entrada" in r.json()
