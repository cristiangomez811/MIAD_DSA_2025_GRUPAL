# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import os

# ======================
# Cargar modelo
# ======================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

try:
    model = load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo: {e}")

# ======================
# API
# ======================
app = FastAPI(
    title="Airline Satisfaction API",
    description="API para predecir satisfacción de pasajeros usando un modelo Random Forest + Pipeline.",
    version="1.0"
)

# ======================
# Esquema del request
# ======================
class PassengerRequest(BaseModel):
    data: dict


# ======================
# ENDPOINT
# ======================
@app.post("/predict")
def predict(req: PassengerRequest):

    # Validar mínimo que venga algo
    if not req.data:
        return {
            "error": "El campo 'data' está vacío. Debe enviar un diccionario con las características del pasajero."
        }

    # Convertir a DataFrame
    try:
        X = pd.DataFrame([req.data])
    except Exception as e:
        return {
            "error": f"No se pudo convertir el input a DataFrame: {e}"
        }

    # Inferencia
    try:
        proba = model.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)
    except Exception as e:
        return {
            "error": f"Error durante la predicción: {e}",
            "received_columns": list(X.columns)
        }

    # Response estándar
    return {
        "prediction": pred,                   # 1 = Satisfecho, 0 = No satisfecho
        "prob_satisfied": float(proba),
        "received_columns": list(X.columns),  # útil para debugging
    }
