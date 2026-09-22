"""
app/main.py - API de inferencia con FastAPI.

Endpoints:
  GET  /health         estado del servicio y si el modelo esta cargado
  GET  /model-info     metadatos del modelo (tipo, variables, metricas, version)
  POST /predict        prediccion de una observacion (con probabilidad)
  POST /predict-batch  prediccion de una lista de observaciones
  GET  /docs           documentacion Swagger (automatica)

El modelo se carga UNA sola vez al iniciar la aplicacion (lifespan).
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import Observacion, Lote, RespuestaPrediccion

# Rutas relativas al proyecto (nunca absolutas)
BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "model" / "model.pkl"
META_PATH = BASE / "model" / "metadata.json"

ARTIFACTS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: cargar modelo y metadatos una sola vez
    ARTIFACTS["model"] = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            ARTIFACTS["metadata"] = json.load(f)
    else:
        ARTIFACTS["metadata"] = {}
    yield
    # Shutdown
    ARTIFACTS.clear()


app = FastAPI(title="API de inferencia - Adult Income",
              version="1.0.0", lifespan=lifespan)


def _etiqueta(pred: int) -> str:
    clases = ARTIFACTS.get("metadata", {}).get("classes", {"0": "0", "1": "1"})
    return clases.get(str(int(pred)), str(int(pred)))


def _version() -> str:
    return ARTIFACTS.get("metadata", {}).get("version", "desconocida")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ARTIFACTS}


@app.get("/model-info")
def model_info():
    meta = ARTIFACTS.get("metadata", {})
    modelo = ARTIFACTS.get("model")
    estimador = type(modelo).__name__ if modelo is not None else None
    return {
        "estimador": estimador,
        "tipo_modelo": meta.get("model_type"),
        "variables_entrada": meta.get("features"),
        "metricas": meta.get("metrics"),
        "sklearn": meta.get("sklearn"),
        "version": meta.get("version"),
        "entrenado_en": meta.get("trained_at"),
    }


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(obs: Observacion):
    try:
        df = pd.DataFrame([obs.model_dump()])
        modelo = ARTIFACTS["model"]
        pred = int(modelo.predict(df)[0])
        proba = float(modelo.predict_proba(df)[0].max())
    except Exception:
        raise HTTPException(status_code=500,
                            detail="Error al generar la prediccion")
    return {
        "prediccion": pred,
        "etiqueta": _etiqueta(pred),
        "probabilidad": round(proba, 4),
        "version_modelo": _version(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.post("/predict-batch")
def predict_batch(lote: Lote):
    if not lote.observaciones:
        raise HTTPException(status_code=422, detail="La lista esta vacia")
    try:
        df = pd.DataFrame([o.model_dump() for o in lote.observaciones])
        modelo = ARTIFACTS["model"]
        preds = modelo.predict(df)
        probas = modelo.predict_proba(df).max(axis=1)
    except Exception:
        raise HTTPException(status_code=500,
                            detail="Error al generar las predicciones")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    resultados = [
        {
            "prediccion": int(p),
            "etiqueta": _etiqueta(int(p)),
            "probabilidad": round(float(pr), 4),
        }
        for p, pr in zip(preds, probas)
    ]
    return {"n": len(resultados), "version_modelo": _version(),
            "timestamp": ts, "predicciones": resultados}
