# API de inferencia — Adult / Census Income (FastAPI)

Servicio web que entrena un modelo de clasificación supervisada, lo serializa como
pipeline completo (`model/model.pkl`) y lo expone como API HTTP con **FastAPI**.
Tarea de Cloud Computing — Diploma en Data Science, UAI.

- **Problema:** predecir si el ingreso anual de una persona supera los 50.000 USD
  (`>50K` vs `<=50K`) — clasificación binaria.
- **Dataset:** Adult / Census Income. Fuente: OpenML (`adult`, versión 2);
  también en UCI (https://archive.ics.uci.edu/dataset/2/adult). Se descarga
  automáticamente al entrenar. Cumple los mínimos: >500 filas, >4 predictoras y
  varias variables categóricas.
- **Modelo:** `RandomForestClassifier` dentro de un `Pipeline` que incluye
  imputación, escalado (numéricas) y one-hot (categóricas). El preprocesamiento
  viaja **dentro** del `.pkl`, así no hay diferencias entre entrenamiento e inferencia.

## Requisitos

- Python **3.12** (ver `runtime.txt`).
- Las dependencias con versión fija están en `requirements.txt`.

## Puesta en marcha (local)

```bash
# 1) Clonar y entrar a la carpeta
git clone https://github.com/danielacarrascoruiz/Cloude-fast-A.git
cd Cloude-fast-A

# 2) Entorno virtual
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / Mac:
# source .venv/bin/activate

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Entrenar y serializar el modelo (crea model/model.pkl y model/metadata.json)
python train.py

# 5) Levantar el servicio
uvicorn app.main:app --reload --port 8000
```

Abrir en el navegador: **http://localhost:8000/docs**

### Prueba interactiva con Swagger

Con la API en ejecución, ingresar a **http://localhost:8000/docs** para acceder
a Swagger UI. Desde esta interfaz se pueden probar directamente los endpoints
`/health`, `/model-info`, `/predict` y `/predict-batch`, además de verificar
la validación de las entradas de la API.

> El modelo se carga una sola vez al iniciar la app (evento `lifespan`).
> Las rutas al `.pkl` son relativas al proyecto (no hay rutas absolutas).

## Endpoints

| Método | Ruta             | Descripción |
|--------|------------------|-------------|
| GET    | `/health`        | Estado del servicio y si el modelo está cargado. |
| GET    | `/model-info`    | Estimador, variables de entrada, métricas y versión. |
| POST   | `/predict`       | Predice una observación; devuelve clase y probabilidad. |
| POST   | `/predict-batch` | Predice una lista de observaciones. |
| GET    | `/docs`          | Documentación Swagger automática. |

## Ejemplos de uso

Predicción individual:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":39,"workclass":"Private","education":"Bachelors","marital_status":"Never-married","occupation":"Adm-clerical","relationship":"Not-in-family","race":"White","sex":"Male","hours_per_week":40,"capital_gain":0,"capital_loss":0,"native_country":"United-States"}'
```

Respuesta:

```json
{"prediccion":0,"etiqueta":"<=50K","probabilidad":0.74,"version_modelo":"1.0.0","timestamp":"..."}
```

Predicción por lote:

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"observaciones":[{"age":39,"workclass":"Private","education":"Bachelors","marital_status":"Never-married","occupation":"Adm-clerical","relationship":"Not-in-family","race":"White","sex":"Male","hours_per_week":40,"capital_gain":0,"capital_loss":0,"native_country":"United-States"}]}'
```

Entrada inválida (devuelve **422**, tipo incorrecto / campo faltante):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":"cuarenta"}'
```

## Pruebas

Las pruebas automatizadas de la API se ejecutaron con el siguiente comando:

```bash
python -m pytest
```

Resultado obtenido:

```text
tests/test_api.py .....                                              [100%]

5 passed, 1 warning in 11.35s
```

## Estructura del repositorio

```
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── Docs/
├── model/
│   ├── model.pkl
│   └── metadata.json
├── tests/
│   └── test_api.py
├── exploracion.ipynb
├── train.py
├── requirements.txt
├── runtime.txt
├── Procfile
├── .gitignore
└── README.md
```

## Decisiones de diseño

- **Pipeline completo serializado:** imputación + escalado + one-hot + modelo en un
  solo objeto, para evitar *training–serving skew*.
- **Validación:** `train/test split` estratificado con semilla fija (`random_state=42`).
- **Métricas reportadas:** accuracy, F1 macro y ROC-AUC (ver `model/metadata.json`).
- **Manejo de errores:** entradas mal formadas → `422` (Pydantic); fallo interno →
  `500` con mensaje controlado, sin exponer trazas.

## Notas

- No se versiona el CSV crudo (ver `.gitignore`); `train.py` obtiene los datos de OpenML.
- `runtime.txt` y `Procfile` dejan declarado el entorno para un despliegue posterior
  en la nube (no requerido en esta entrega).
