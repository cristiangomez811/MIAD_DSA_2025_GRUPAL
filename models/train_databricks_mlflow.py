"""
Entrenamiento de RandomForest con MLflow listo para Databricks.

Se puede ejecutar tanto en local como dentro de un Job/Repo de Databricks:
python models/train_databricks_mlflow.py --experiment-path "/Shared/airline-sat"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cols basuras y normaliza la variable objetivo."""
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if "id" in df.columns:
        drop_cols.append("id")

    df = df.drop(columns=drop_cols, errors="ignore").copy()
    if "satisfaction" not in df.columns:
        raise ValueError("No se encontro la columna 'satisfaction' en el dataset.")

    df["satisfaction"] = df["satisfaction"].map(
        {"satisfied": 1, "neutral or dissatisfied": 0}
    )
    return df


def prepare_data(
    train_path: Path,
    test_path: Optional[Path],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Iterable[str], Iterable[str]]:
    """Carga datos, limpia y retorna splits + listas de columnas."""
    train_df = clean_dataframe(pd.read_csv(train_path))

    if test_path and test_path.exists():
        test_df = clean_dataframe(pd.read_csv(test_path))
    else:
        train_df, test_df = train_test_split(
            train_df, test_size=test_size, random_state=seed, stratify=train_df["satisfaction"]
        )

    X_train = train_df.drop(columns=["satisfaction"])
    y_train = train_df["satisfaction"]

    X_test = test_df.drop(columns=["satisfaction"])
    y_test = test_df["satisfaction"]

    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

    return X_train, y_train, X_test, y_test, cat_cols, num_cols


def build_pipeline(
    cat_cols: Iterable[str],
    num_cols: Iterable[str],
    n_estimators: int,
    max_depth: Optional[int],
    random_state: int,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def train_and_evaluate(
    train_path: Path,
    test_path: Optional[Path],
    n_estimators: int,
    max_depth: Optional[int],
    test_size: float,
    random_state: int,
) -> Tuple[Pipeline, Dict[str, float]]:
    X_train, y_train, X_test, y_test, cat_cols, num_cols = prepare_data(
        train_path=train_path,
        test_path=test_path,
        test_size=test_size,
        seed=random_state,
    )

    pipe = build_pipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    return pipe, metrics


def save_local_artifacts(model: Pipeline, metrics: Dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(model, output_dir / "model.joblib")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena RandomForest con MLflow listo para Databricks.")
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"), help="Ruta al CSV de entrenamiento.")
    parser.add_argument("--test-path", type=Path, default=Path("data/test.csv"), help="Ruta al CSV de prueba (opcional).")
    parser.add_argument("--experiment-path", type=str, default="/Shared/airline-satisfaction", help="Ruta del experimento en Databricks.")
    parser.add_argument("--run-name", type=str, default="rf-baseline", help="Nombre del run en MLflow.")
    parser.add_argument("--register-model-name", type=str, default="", help="Nombre en el Model Registry; si vacio no registra.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Cantidad de arboles del RandomForest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Profundidad maxima del RandomForest.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion para split si no hay test.csv.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed para reproducibilidad.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/artifacts"), help="Carpeta local para guardar modelo y metricas.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configurar experimento en Databricks (en local crea mlruns/)
    mlflow.set_experiment(args.experiment_path)

    with mlflow.start_run(run_name=args.run_name) as run:
        model, metrics = train_and_evaluate(
            train_path=args.train_path,
            test_path=args.test_path,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        mlflow.log_params(
            {
                "train_path": str(args.train_path),
                "test_path": str(args.test_path),
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "random_state": args.random_state,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(metrics, "metrics.json")

        mlflow.sklearn.log_model(model, artifact_path="model")

        if args.register_model_name:
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri=model_uri, name=args.register_model_name)

        save_local_artifacts(model, metrics, args.artifacts_dir)
        print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
