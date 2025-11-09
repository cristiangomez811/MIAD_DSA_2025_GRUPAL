# modeloRandomForest.py
# =============================================
# Proyecto: Airline Passenger Satisfaction
# Modelo: Random Forest 
# =============================================

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump

train_file = "train.csv"
test_file = "test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

#Mapear satisfacción como binario con 1 y 0
train_df['satisfaction'] = train_df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
test_df['satisfaction'] = test_df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

#Seleccionar columnas categóricas
cat_cols = [c for c in train_df.select_dtypes(exclude='number').columns if c != 'satisfaction']
le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

#Columnas numéricas
num_cols = [c for c in train_df.select_dtypes(include='number').columns if c != 'satisfaction']
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

#separar variable de interés de las predictoras
X_train, y_train = train_df.drop('satisfaction', axis=1), train_df['satisfaction']
X_test, y_test = test_df.drop('satisfaction', axis=1), test_df['satisfaction']

#Entrenar el modelo 
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMétricas del modelo Random Forest con {n_estimators} estimadores:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

#Métricas para evaluar
metrics_df = pd.DataFrame({
    "Modelo": ["Random Forest"],
    "Accuracy": [acc],
    "Precision": [prec],
    "Recall": [rec],
    "F1-Score": [f1]
})
metrics_df.to_csv("model_metrics.csv", index=False)

#Inclur el top 5 de variables más relevantes según RF
importances = rf.feature_importances_
feature_names = X_train.columns
feat_imp_df = pd.DataFrame({
    "Variable": feature_names,
    "Importancia": importances
})
feat_imp_df = feat_imp_df.sort_values(by="Importancia", ascending=False)

print("\n Top 5 variables más importantes según Random Forest:")
print(feat_imp_df.head(5))

# Guardar top 5 en CSV
feat_imp_df.head(5).to_csv("top5_feature_importance.csv", index=False)
print("Top 5 de variables relevantes guardado en top5_feature_importance.csv")

dump(rf, "best_model_random_forest.joblib")
print("\n Modelo guardado como best_model_random_forest.joblib")
print("Métricas guardadas en model_metrics.csv")
print("Proceso completo.")



