"""
train.py - Entrenamiento y serializacion del modelo (Adult / Census Income).

Entrena un Pipeline COMPLETO (preprocesamiento + estimador) y lo guarda como
model/model.pkl. Tambien escribe model/metadata.json con la version de
scikit-learn, la lista ordenada de variables y las metricas.

Uso:
    python train.py
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# Definicion de variables (nombres ya normalizados con guion_bajo)
# ------------------------------------------------------------------
NUM_COLS = ["age", "hours_per_week", "capital_gain", "capital_loss"]
CAT_COLS = ["workclass", "education", "marital_status", "occupation",
            "relationship", "race", "sex", "native_country"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "class"

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
DATA_CSV = BASE / "data" / "adult.csv"   # fallback local opcional


def load_data() -> pd.DataFrame:
    """Carga el dataset Adult. Intenta OpenML; si no hay internet usa
    data/adult.csv (si existe)."""
    try:
        from sklearn.datasets import fetch_openml
        print("Descargando dataset Adult desde OpenML...")
        ds = fetch_openml("adult", version=2, as_frame=True)
        df = ds.frame.copy()
    except Exception as e:
        if DATA_CSV.exists():
            print(f"OpenML no disponible ({e}). Usando {DATA_CSV}")
            df = pd.read_csv(DATA_CSV)
        else:
            raise RuntimeError(
                "No se pudo descargar Adult desde OpenML y no existe "
                "data/adult.csv. Conectate a internet y reintenta."
            ) from e

    # Normalizar nombres de columnas: guiones y puntos -> guion_bajo
    df.columns = [c.strip().replace("-", "_").replace(".", "_") for c in df.columns]
    if TARGET not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": TARGET})
    return df


def build_pipeline() -> Pipeline:
    """Arma el pipeline completo: imputacion + escala + one-hot + modelo."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    clf = RandomForestClassifier(n_estimators=60, max_depth=18, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # Objetivo binario: 1 si ingreso > 50K
    y = df[TARGET].astype(str).str.contains(">50K").astype(int)
    X = df[FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    print("Entrenando pipeline (RandomForest)...")
    pipe.fit(X_train, y_train)

    # Evaluacion (al menos dos metricas)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1m:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    # Guardar artefacto (pipeline completo)
    model_path = MODEL_DIR / "model.pkl"
    joblib.dump(pipe, model_path)

    # Metadatos
    metadata = {
        "model_type": "RandomForestClassifier (Pipeline con preprocesamiento)",
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "target": TARGET,
        "classes": {"0": "<=50K", "1": ">50K"},
        "features": FEATURES,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "metrics": {
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1m), 4),
            "roc_auc": round(float(auc), 4),
        },
        "primary_metric": "f1_macro",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "1.0.0",
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado: {model_path}")
    print(f"Guardado: {MODEL_DIR / 'metadata.json'}")
    print("Listo.")


if __name__ == "__main__":
    main()
