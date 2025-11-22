import os
from typing import List, Optional

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import requests

px.defaults.template = "plotly_white"

# ======================
# CONFIGURACIÓN API
# ======================
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")


# ======================
# CARGAR DATA
# ======================
def load_data() -> pd.DataFrame:
    """Load train.csv from root or data/ subfolder."""
    candidate_paths = [
        os.path.join(os.getcwd(), "train.csv"),
        os.path.join(os.getcwd(), "data", "train.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            "No train.csv found. Expected at ./train.csv or ./data/train.csv"
        )

    # Basic numeric coercion
    for col in [
        "Arrival Delay in Minutes",
        "Departure Delay in Minutes",
        "Flight Distance",
        "Age",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df = load_data()

# ======================
# COLUMNAS DE SERVICIO
# ======================
SERVICE_COLS: List[str] = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]
SERVICE_COLS = [c for c in SERVICE_COLS if c in df.columns]


# ======================
# DROPDOWN OPTIONS
# ======================
opt = (
    lambda s: [
        {"label": str(v), "value": v} for v in sorted(df[s].dropna().unique())
    ]
    if s in df.columns
    else []
)


# ======================
# FILTRO DE COHORTES
# ======================
def cohort_filter(
    data: pd.DataFrame,
    gender: Optional[str],
    customer: Optional[str],
    travel: Optional[str],
    cls: Optional[str],
    age: Optional[float],
    distance: Optional[float],
) -> pd.DataFrame:
    dff = data.copy()
    if gender and "Gender" in dff.columns:
        dff = dff[dff["Gender"] == gender]
    if customer and "Customer Type" in dff.columns:
        dff = dff[dff["Customer Type"] == customer]
    if travel and "Type of Travel" in dff.columns:
        dff = dff[dff["Type of Travel"] == travel]
    if cls and "Class" in dff.columns:
        dff = dff[dff["Class"] == cls]
    # Tolerance windows to avoid empty sets
    if (age is not None) and ("Age" in dff.columns):
        dff = dff[dff["Age"].between(age - 3, age + 3)]
    if (distance is not None) and ("Flight Distance" in dff.columns):
        dff = dff[dff["Flight Distance"].between(distance - 150, distance + 150)]
    return dff


# ======================
# UTILIDADES
# ======================
def label_map(s: str) -> str:
    return "Satisfecho" if s and s.lower().strip() == "satisfied" else "Neutral/Insatisfecho"


def factor_importance(data: pd.DataFrame) -> pd.DataFrame:
    if "satisfaction" not in data.columns or not SERVICE_COLS:
        return pd.DataFrame({"Servicio": [], "Impacto": []})
    tmp = data.copy()
    tmp["is_sat"] = (tmp["satisfaction"].str.lower() == "satisfied").astype(int)
    res = []
    for c in SERVICE_COLS:
        ser = pd.to_numeric(tmp[c], errors="coerce")
        if ser.notna().sum() == 0:
            continue
        m_sat = ser[tmp["is_sat"] == 1].mean()
        m_not = ser[tmp["is_sat"] == 0].mean()
        res.append((c, m_sat - m_not))
    return pd.DataFrame(res, columns=["Servicio", "Impacto"]).sort_values("Impacto", ascending=False)


# ======================
# BUILD PAYLOAD PARA API
# ======================
def build_payload(df, gender, customer, travel, cls, age, distance):
    """Construye un payload COMPLETO basado en df (medianas/modas) y los inputs del usuario."""
    row = df.drop(columns=["satisfaction"], errors="ignore").iloc[0].to_dict()

    # Numericas: medianas
    for col in df.select_dtypes(include="number").columns:
        if col in row:
            row[col] = float(df[col].median())

    # Categoricas: modas
    for col in df.select_dtypes(exclude="number").columns:
        if col in row and col != "satisfaction":
            row[col] = df[col].mode().iloc[0]

    # Sobrescribir con inputs
    if gender is not None: row["Gender"] = gender
    if customer is not None: row["Customer Type"] = customer
    if travel is not None: row["Type of Travel"] = travel
    if cls is not None: row["Class"] = cls
    if age is not None: row["Age"] = float(age)
    if distance is not None: row["Flight Distance"] = float(distance)

    return row


# ======================
# DASHBOARD
# ======================
app = Dash(
    __name__,
    title="Prediccion de Satisfaccion de Pasajeros",
    external_stylesheets=[dbc.themes.FLATLY],
)
server = app.server


# ======================
# LAYOUT
# ======================
app.layout = dbc.Container(
    [
        html.H3(
            "Dashboard: Prediccion de Satisfaccion de Pasajeros",
            className="mt-3 mb-4 text-center fw-semibold",
        ),

        dbc.Row(
            [
                # Left column: form and metrics
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "Ingrese los datos del pasajero y el viaje",
                                    className="mb-3 text-center fw-bold",
                                ),
                                dbc.Form(
                                    dbc.Row(
                                        [
                                            dbc.Col([
                                                dbc.Label("Genero"),
                                                dcc.Dropdown(id="inp-gender", options=opt("Gender"), placeholder="Selecciona"),
                                            ], md=6),
                                            dbc.Col([
                                                dbc.Label("Tipo de cliente"),
                                                dcc.Dropdown(id="inp-customer", options=opt("Customer Type"), placeholder="Selecciona"),
                                            ], md=6),
                                            dbc.Col([
                                                dbc.Label("Tipo de viaje"),
                                                dcc.Dropdown(id="inp-travel", options=opt("Type of Travel"), placeholder="Selecciona"),
                                            ], md=6),
                                            dbc.Col([
                                                dbc.Label("Clase"),
                                                dcc.Dropdown(id="inp-class", options=opt("Class"), placeholder="Selecciona"),
                                            ], md=6),
                                            dbc.Col([
                                                dbc.Label("Edad"),
                                                dbc.Input(id="inp-age", type="number", placeholder="Edad"),
                                            ], md=6),
                                            dbc.Col([
                                                dbc.Label("Distancia del vuelo (en millas)"),
                                                dbc.Input(id="inp-distance", type="number", placeholder="Distancia"),
                                            ], md=6),
                                        ],
                                        className="g-2",
                                    ),
                                ),

                                html.Div(
                                    dbc.Button(
                                        "Predecir",
                                        id="btn-predict",
                                        n_clicks=0,
                                        color="primary",
                                        size="lg",
                                        style={
                                            "width": "50%",
                                            "backgroundColor": "#93c5fd",
                                            "borderColor": "#93c5fd",
                                            "color": "#0b2e59",
                                        },
                                        className="mt-2 py-2 fw-semibold",
                                    ),
                                ),

                                # NUEVO: espacio para la predicción del modelo
                                html.H4(
                                    id="pred-output",
                                    className="mt-3 text-center fw-bold",
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.Div("Tasa de satisfaccion", className="text-muted"),
                                                    html.H2(id="metric-sat", children="-", className="mb-0"),
                                                ])
                                            ),
                                            md=6,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.Div("Tasa de clientes insatisfechos", className="text-muted"),
                                                    html.H2(id="metric-dissat", children="-", className="mb-0"),
                                                ])
                                            ),
                                            md=6,
                                        ),
                                    ],
                                    className="g-2 mt-2",
                                ),
                            ]
                        )
                    ),
                    md=5,
                ),

                # Right column: charts
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "Distribucion de satisfaccion",
                                    className="mb-2 text-center fw-bold",
                                ),
                                dcc.Graph(id="pie-sat"),
                                html.H4(
                                    "Principales factores de satisfaccion",
                                    className="mb-2 mt-2 text-center fw-bold",
                                ),
                                dcc.Graph(id="bar-factors"),
                            ]
                        )
                    ),
                    md=7,
                ),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)


# ======================
# CALLBACK
# ======================
@app.callback(
    [
        Output("pred-output", "children"),     # << NUEVO
        Output("metric-sat", "children"),
        Output("metric-dissat", "children"),
        Output("pie-sat", "figure"),
        Output("bar-factors", "figure"),
    ],
    [Input("btn-predict", "n_clicks")],
    [
        State("inp-gender", "value"),
        State("inp-customer", "value"),
        State("inp-travel", "value"),
        State("inp-class", "value"),
        State("inp-age", "value"),
        State("inp-distance", "value"),
    ],
)
def update_dashboard(n_clicks, gender, customer, travel, cls, age, distance):

    # ======================
    # 1. Construir payload
    # ======================
    payload = build_payload(df, gender, customer, travel, cls, age, distance)

    # ======================
    # 2. Llamar API
    # ======================
    try:
        r = requests.post(API_URL, json={"data": payload}, timeout=5)
        res = r.json()
        pred = res.get("prediction", 0)
        proba = res.get("prob_satisfied", 0)
        pred_text = f"Predicción del modelo: {'Satisfecho' if pred==1 else 'No Satisfecho'} ({proba:.2%})"
    except Exception as e:
        pred_text = "Error llamando la API. Verifique el servidor."

    # ======================
    # 3. Charts históricos
    # ======================
    dff = cohort_filter(df, gender, customer, travel, cls, age, distance)
    if len(dff) == 0:
        dff = df

    if "satisfaction" in dff.columns:
        sat_counts = dff["satisfaction"].value_counts()
        sat = int(sat_counts.get("satisfied", 0))
        not_sat = int(sat_counts.sum() - sat)
        total = sat + not_sat if (sat + not_sat) > 0 else 1
        sat_rate = round(100 * sat / total)
        dissat_rate = 100 - sat_rate
    else:
        sat_rate = dissat_rate = 0

    # Pie figure
    color_map = {"Satisfecho": "#3b82f6", "Neutral/Insatisfecho": "#ef4444"}
    if "satisfaction" in dff.columns:
        pie_df = dff["satisfaction"].map(label_map).value_counts().reset_index()
        pie_df.columns = ["Estado", "Cuenta"]
        pie_fig = px.pie(
            pie_df,
            names="Estado",
            values="Cuenta",
            color="Estado",
            color_discrete_map=color_map,
        )
    else:
        pie_fig = px.pie(
            pd.DataFrame({"Estado": ["Satisfecho", "Neutral/Insatisfecho"], "Cuenta": [1, 1]}),
            names="Estado",
            values="Cuenta",
            color="Estado",
            color_discrete_map=color_map,
        )

    # Factors figure
    imp = factor_importance(dff)
    if imp.empty:
        bar_fig = px.bar(pd.DataFrame({"Servicio": [], "Impacto": []}), x="Impacto", y="Servicio")
    else:
        bar_fig = px.bar(imp.head(7), x="Impacto", y="Servicio", orientation="h")
        bar_fig.update_layout(yaxis={"categoryorder": "total ascending"})

    return pred_text, f"{sat_rate}%", f"{dissat_rate}%", pie_fig, bar_fig


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=True)
