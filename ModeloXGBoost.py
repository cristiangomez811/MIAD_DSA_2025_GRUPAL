# -*- coding: utf-8 -*-
"""
XGBoost (GPU si está disponible) – simple y robusto a versiones
Ejecución:  python xgboost_gpu_simple.py
Requisitos: pandas, numpy, scikit-learn, xgboost, joblib
Archivos esperados: train.csv, test.csv (mismo directorio)
Salidas: best_model_xgb.joblib, model_metrics_xgb.csv, top10_feature_importance_xgb.csv
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
import xgboost as xgb

# ---------------- Config ----------------
TRAIN_FILE = "train.csv"
TEST_FILE  = "test.csv"
TARGET = "satisfaction"
RANDOM_SEED = 42
VAL_SIZE = 0.15
N_ESTIMATORS = 1000
ES_ROUNDS = 50  # si callbacks no existe en tu versión, entrenará sin early stopping

# -------------- Util --------------------
def map_target(df: pd.DataFrame, col: str):
    mapping = {'satisfied': 1, 'neutral or dissatisfied': 0}
    if col in df.columns:
        df[col] = df[col].map(mapping)
    return df

def detect_cols(df: pd.DataFrame, target: str):
    cat_cols = [c for c in df.select_dtypes(exclude='number').columns if c != target]
    num_cols = [c for c in df.select_dtypes(include='number').columns if c != target]
    return cat_cols, num_cols

def label_encode_inplace(train_df: pd.DataFrame, test_df: pd.DataFrame, cols):
    for col in cols:
        le = LabelEncoder()
        both = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(both)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))

def compute_scale_pos_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return float(neg / max(pos, 1))

def build_xgb(seed: int, spw: float) -> XGBClassifier:
    base = dict(
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric="auc",
        n_jobs=-1,
        scale_pos_weight=spw,
    )
    # 1) Intentar API moderna con device="cuda"
    try:
        model = XGBClassifier(**base, device="cuda", tree_method="hist")
        # fuerza construcción para que falle aquí si no hay cuda
        _ = model.get_xgb_params()
        print("Usando GPU (device='cuda')")
        return model
    except Exception:
        pass
    # 2) Intentar API antigua con gpu_hist
    try:
        model = XGBClassifier(**base, tree_method="gpu_hist")
        _ = model.get_xgb_params()
        print("Usando GPU (tree_method='gpu_hist')")
        return model
    except Exception:
        pass
    # 3) CPU
    print("Usando CPU (tree_method='hist')")
    return XGBClassifier(**base, tree_method="hist")

# -------------- Main --------------------
def main():
    assert os.path.exists(TRAIN_FILE), f"No se encontró {TRAIN_FILE}"
    assert os.path.exists(TEST_FILE),  f"No se encontró {TEST_FILE}"

    print(f"Versión de xgboost: {xgb.__version__}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    train_df = map_target(train_df, TARGET)
    test_df  = map_target(test_df, TARGET)

    cat_cols, _ = detect_cols(train_df, TARGET)
    label_encode_inplace(train_df, test_df, cat_cols)

    X_train_full = train_df.drop(columns=[TARGET])
    y_train_full = train_df[TARGET].astype(int)
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_train_full
    )

    spw = compute_scale_pos_weight(y_tr)
    model = build_xgb(RANDOM_SEED, spw)

    # Early stopping robusto a versiones
    callbacks = []
    try:
        from xgboost.callback import EarlyStopping
        callbacks = [EarlyStopping(rounds=ES_ROUNDS, save_best=True)]
    except Exception:
        callbacks = []

    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=callbacks
        )
    except TypeError:
        # Para versiones sin soporte de callbacks: entrenar sin early stopping
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # Métricas
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    ap  = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan

    print("\nMétricas XGBoost:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}" if not np.isnan(auc) else "ROC AUC:   N/A")
    print(f"PR  AUC:   {ap:.4f}"  if not np.isnan(ap)  else "PR  AUC:   N/A")

    pd.DataFrame([{
        "Modelo": "XGBoost",
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1,
        "ROC_AUC": None if np.isnan(auc) else auc,
        "PR_AUC":  None if np.isnan(ap)  else ap
    }]).to_csv("model_metrics_xgb.csv", index=False)

    # Importancias
    try:
        fi = pd.DataFrame({
            "Variable": X_tr.columns,
            "Importancia": model.feature_importances_
        }).sort_values("Importancia", ascending=False)
        fi.head(10).to_csv("top10_feature_importance_xgb.csv", index=False)
        print("Top10 importancias guardado en top10_feature_importance_xgb.csv")
    except Exception:
        print("Aviso: no se pudieron exportar importancias.")

    dump(model, "best_model_xgb.joblib")
    print("Modelo guardado en best_model_xgb.joblib")
    print("Métricas guardadas en model_metrics_xgb.csv")
    print("Listo.")

if __name__ == "__main__":
    main()
