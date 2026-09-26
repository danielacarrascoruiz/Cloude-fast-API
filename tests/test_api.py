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

# ------------------------------------------------------------------
# Pruebas adicionales
# ------------------------------------------------------------------

def test_edad_fuera_de_rango_422():
    """Una edad fuera del rango permitido (17-100) debe devolver 422."""
    with TestClient(app) as client:
        mala = dict(OBS_VALIDA)
        mala["age"] = 150
        r = client.post("/predict", json=mala)
        assert r.status_code == 422


def test_batch_vacio_422():
    """Un lote sin observaciones debe devolver 422."""
    with TestClient(app) as client:
        r = client.post("/predict-batch", json={"observaciones": []})
        assert r.status_code == 422


def test_batch_mantiene_orden():
    """/predict-batch debe devolver las predicciones en el mismo orden
    que /predict aplicado a cada observacion por separado."""
    obs_2 = dict(OBS_VALIDA)
    obs_2.update({"age": 52, "education": "Masters",
                  "marital_status": "Married-civ-spouse",
                  "relationship": "Husband", "hours_per_week": 55,
                  "capital_gain": 15000})
    with TestClient(app) as client:
        individuales = [client.post("/predict", json=o).json()
                        for o in (OBS_VALIDA, obs_2)]
        lote = client.post("/predict-batch",
                           json={"observaciones": [OBS_VALIDA, obs_2]}).json()
        for ind, res in zip(individuales, lote["predicciones"]):
            assert ind["prediccion"] == res["prediccion"]
            assert ind["probabilidad"] == res["probabilidad"]


def test_model_info_incluye_metricas():
    """/model-info debe exponer las metricas del entrenamiento."""
    with TestClient(app) as client:
        metricas = client.get("/model-info").json()["metricas"]
        for m in ("accuracy", "f1_macro", "roc_auc"):
            assert m in metricas
