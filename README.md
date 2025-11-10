# MIAD_DSA_2025_GRUPAL —


## Contenido principal
- `app.py`: aplicación Dash con formulario (pasajero/viaje), métricas y gráficos (pastel y factores).
- `requirements.txt`: dependencias (Dash, Plotly, pandas, numpy, dash-bootstrap-components).
- `train.csv` y `data/train.csv`: datos usados por el dashboard
- `ModeloXGBoost.py`, `modeloRandomForest.py`: scripts de modelos.
- `best_model_xgb.joblib`: modelo entrenado
- `Modelo_RL.ipynb`, `src/EDA.ipynb`: notebooks de análisis.

## Requisitos
- Python 3.9+ (recomendado 3.10/3.11)

## Instalación
```
pip install -r requirements.txt
```

## Ejecución
```
python app.py
```
- Abrir: http://localhost:8050
- Puerto opcional: `PORT=8051 python app.py`
- El CSV debe estar en `train.csv` o `data/train.csv`.

## Personalización rápida
- Tema Bootstrap: cambiar `dbc.themes.FLATLY` en `app.py` por otro (p. ej., `dbc.themes.CYBORG`).
- Colores del gráfico de pastel fijos en `app.py` (azul para “Satisfecho”, rojo para “Neutral/Insatisfecho”).

## Notas
- Si falta alguna librería, reinstalar con `pip install -r requirements.txt`.
- Si el puerto está en uso, cambie `PORT` o cierre procesos en ese puerto.
