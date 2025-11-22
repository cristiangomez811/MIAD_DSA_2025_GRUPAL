# models/train_rf.py
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

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

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

def run_experiment(n_estimators=200, max_depth=None):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
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

    # 1. Definir experimento (en local crea carpeta mlruns; en Databricks queda en Experiments)
    mlflow.set_experiment("DSA-Airline-Satisfaction")

    # 2. Run con tus parámetros (cada compañero cambia algo)
    with mlflow.start_run(run_name="RF_baseline"):
        pipe, metrics = run_experiment(n_estimators=200, max_depth=None)

        # log parámetros
        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42
        })

        # log métricas
        mlflow.log_metrics(metrics)

        # log modelo como artefacto de MLflow
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(metrics)

        # además guardas artefactos locales para el repo
        os.makedirs("models/artifacts", exist_ok=True)
        with open("models/artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        dump(pipe, "models/artifacts/model.joblib")
