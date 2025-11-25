# models/train_rf_mlflow.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump
import json
import os
import mlflow
import mlflow.sklearn

TRAIN_PATH = os.path.join("data", "train.csv")
TEST_PATH  = os.path.join("data", "test.csv")

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

train_df["satisfaction"] = train_df["satisfaction"].map(
    {"satisfied": 1, "neutral or dissatisfied": 0}
)
test_df["satisfaction"] = test_df["satisfaction"].map(
    {"satisfied": 1, "neutral or dissatisfied": 0}
)

X_train = train_df.drop(columns=["satisfaction"])
y_train = train_df["satisfaction"]
X_test  = test_df.drop(columns=["satisfaction"])
y_test  = test_df["satisfaction"]

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

def run_experiment(n_estimators=200, max_depth=None, max_features="auto"):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", rf)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    return pipe, metrics


if __name__ == "__main__":

    # ======= CAMBIO CLAVE PARA EC2 (como diabetes) =======
    mlflow.set_tracking_uri("http://13.218.39.188:8050")
    # ejemplo real:
    # mlflow.set_tracking_uri("http://3.91.210.55:8050")

    mlflow.set_experiment("DSA-Airline-Satisfaction")

    # cada compañero cambia parámetros aquí
    n_estimators = 200
    max_depth = None
    max_features = "auto"

    run_name = f"RF_{n_estimators}_depth{max_depth}_feat{max_features}"

    with mlflow.start_run(run_name=run_name):
        pipe, metrics = run_experiment(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features
        )

        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "random_state": 42
        })

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(metrics)

        # artefactos locales (NO se suben a git)
        os.makedirs("models/artifacts", exist_ok=True)
        with open("models/artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        dump(pipe, "models/artifacts/model.joblib")
