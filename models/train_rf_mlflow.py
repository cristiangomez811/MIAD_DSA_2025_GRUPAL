# models/train_rf.py
# =============================================
# Proyecto: Airline Passenger Satisfaction
# Modelo: Random Forest + MLflow Tracking
# =============================================

import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from joblib import dump

import mlflow
import mlflow.sklearn


TRAIN_PATH = os.path.join("data", "train.csv")
TEST_PATH  = os.path.join("data", "test.csv")

ARTIFACT_DIR = os.path.join("models", "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ===============================
# 0. CONFIG MLFLOW
# ===============================
# Si estás corriendo mlflow server en la misma VM:
# export MLFLOW_TRACKING_URI=http://127.0.0.1:8050
# Si no exportas nada, usa localhost por defecto.
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8050")
mlflow.set_tracking_uri(tracking_uri)

EXPERIMENT_NAME = "DSA-Airline-Satisfaction"
mlflow.set_experiment(EXPERIMENT_NAME)


# ===============================
# 1. CARGAR Y LIMPIAR DATOS
# ===============================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

drop_cols = [c for c in train_df.columns if c.lower().startswith("unnamed")]
if "id" in train_df.columns:
    drop_cols.append("id")

train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df  = test_df.drop(columns=drop_cols, errors="ignore")


# ===============================
# 2. TARGET BINARIO CONSISTENTE
# ===============================
target_map = {"satisfied": 1, "neutral or dissatisfied": 0}

train_df["satisfaction"] = train_df["satisfaction"].map(target_map)
test_df["satisfaction"]  = test_df["satisfaction"].map(target_map)

X_train = train_df.drop(columns=["satisfaction"])
y_train = train_df["satisfaction"]

X_test = test_df.drop(columns=["satisfaction"])
y_test = test_df["satisfaction"]


# ===============================
# 3. PREPROCESAMIENTO Y MODELO
# ===============================
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# Hiperparámetros claros (evita "auto")
n_estimators = 200
max_depth = None
max_features = "sqrt"
random_state = 42

rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    max_features=max_features,
    random_state=random_state,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf)
])


# ===============================
# 4-6. ENTRENAR + LOG MLFLOW + GUARDAR
# ===============================
with mlflow.start_run(run_name=f"RF_{n_estimators}_depth{max_depth}_feat{max_features}"):

    # Log de parámetros principales
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
        "random_state": random_state,
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
    })

    # Entrenar
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Métricas
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0))
    }
    print(metrics)

    # Log de métricas a MLflow
    mlflow.log_metrics(metrics)

    # Guardar métricas localmente
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Log del json como artifact en MLflow
    mlflow.log_artifact(metrics_path)

    # Guardar modelo localmente
    model_path = os.path.join(ARTIFACT_DIR, "model.joblib")
    dump(pipe, model_path)
    print(f"Modelo final guardado en {model_path}")

    # Log del modelo en MLflow (con pipeline completa)
    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path="model",
        registered_model_name=None
    )

    print("Run registrado en MLflow ✅")
