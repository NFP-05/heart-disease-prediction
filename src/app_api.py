import os
import time
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "best_LRmodel.pkl")
ENCODERS_PATH = os.path.join(PROJECT_ROOT, "outputs", "encoders.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "outputs", "scaler.pkl")
TRAIN_COLUMNS_PATH = os.path.join(PROJECT_ROOT, "outputs", "train_columns.pkl")

DB_PATH = os.path.join(PROJECT_ROOT, "data", "monitoring.db")

# ---------------------------------------------------------------------------
# Global model artefacts (loaded once at startup)
# ---------------------------------------------------------------------------
model = None
encoders = None
scaler = None
train_columns = None

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PatientData(BaseModel):
    Age: int = Field(..., ge=20, le=80, description="Age in years")
    Sex: Literal["Male", "Female"] = Field(..., description="Gender")
    ChestPainType: Literal["ATA", "NAP", "TA", "ASY"] = Field(
        ..., description="Chest pain type"
    )
    RestingBP: int = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    Cholesterol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    FastingBS: Literal["No", "Yes"] = Field(
        ..., description="Fasting blood sugar > 120 mg/dl"
    )
    MaxHR: int = Field(..., ge=60, le=210, description="Maximum heart rate achieved")
    ExerciseAngina: Literal["N", "Y"] = Field(
        ..., description="Exercise-induced angina"
    )
    Oldpeak: float = Field(
        ..., ge=-2.0, le=6.0, description="ST depression induced by exercise relative to rest"
    )
    RestingECG: Literal["Normal", "ST", "LVH"] = Field(
        ..., description="Resting electrocardiographic results"
    )
    ST_Slope: Literal["Up", "Flat", "Down"] = Field(
        ..., description="Slope of the peak exercise ST segment"
    )

    @field_validator("Oldpeak")
    @classmethod
    def round_oldpeak(cls, v: float) -> float:
        return round(v, 2)


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    probability: float
    threshold: float
    response_time_ms: float
    timestamp: str


class MonitoringLog(BaseModel):
    id: int
    timestamp: str
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    RestingECG: str
    ST_Slope: str
    probability: float
    risk_label: str
    response_time_ms: float


class MonitoringSummary(BaseModel):
    total_predictions: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    avg_probability: float
    high_risk_pct: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database: str

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def get_db_connection() -> sqlite3.Connection:
    """Return a connection with WAL mode enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create the predictions table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            Age             INTEGER NOT NULL,
            Sex             TEXT    NOT NULL,
            ChestPainType   TEXT    NOT NULL,
            RestingBP       INTEGER NOT NULL,
            Cholesterol     INTEGER NOT NULL,
            FastingBS       TEXT    NOT NULL,
            MaxHR           INTEGER NOT NULL,
            ExerciseAngina  TEXT    NOT NULL,
            Oldpeak         REAL    NOT NULL,
            RestingECG      TEXT    NOT NULL,
            ST_Slope        TEXT    NOT NULL,
            probability     REAL    NOT NULL,
            risk_label      TEXT    NOT NULL,
            response_time_ms REAL   NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def log_prediction(data: PatientData, probability: float,
                   risk_label: str, response_time_ms: float) -> None:
    """Insert a prediction record into SQLite."""
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO predictions
            (timestamp, Age, Sex, ChestPainType, RestingBP, Cholesterol,
             FastingBS, MaxHR, ExerciseAngina, Oldpeak, RestingECG, ST_Slope,
             probability, risk_label, response_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            data.Age,
            data.Sex,
            data.ChestPainType,
            data.RestingBP,
            data.Cholesterol,
            data.FastingBS,
            data.MaxHR,
            data.ExerciseAngina,
            data.Oldpeak,
            data.RestingECG,
            data.ST_Slope,
            round(probability, 4),
            risk_label,
            round(response_time_ms, 2),
        ),
    )
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Preprocessing helper (mirrors dash.py logic)
# ---------------------------------------------------------------------------

THRESHOLD = 0.5


def preprocess_input(data: PatientData) -> pd.DataFrame:
    """Convert PatientData → model-ready DataFrame with the same pipeline as
    the original dashboard."""

    sex = "M" if data.Sex == "Male" else "F"
    fasting_bs = 1 if data.FastingBS == "Yes" else 0

    raw = pd.DataFrame(
        [[
            data.Age, sex, data.RestingBP, data.Cholesterol, fasting_bs,
            data.MaxHR, data.ExerciseAngina, data.Oldpeak,
            data.ChestPainType, data.RestingECG, data.ST_Slope,
        ]],
        columns=[
            "Age", "Sex", "RestingBP", "Cholesterol", "FastingBS",
            "MaxHR", "ExerciseAngina", "Oldpeak",
            "ChestPainType", "RestingECG", "ST_Slope",
        ],
    )

    # Feature engineering
    raw["HR_Ratio"] = raw["MaxHR"] / (220 - raw["Age"])
    raw.drop(columns=["MaxHR"], inplace=True)

    # Label encoding
    raw["Sex"] = encoders["label_encoders"]["Sex"].transform(raw["Sex"])
    raw["ExerciseAngina"] = encoders["label_encoders"]["ExerciseAngina"].transform(
        raw["ExerciseAngina"]
    )

    # One-hot encoding
    raw = pd.get_dummies(
        raw, columns=["ChestPainType", "RestingECG", "ST_Slope"], drop_first=False
    )

    # Scaling
    numeric_cols = ["Age", "RestingBP", "Cholesterol", "Oldpeak", "HR_Ratio"]
    raw[numeric_cols] = scaler.transform(raw[numeric_cols])

    # Align to training columns
    raw = raw.reindex(columns=train_columns, fill_value=0)

    return raw


def predict_risk(data: PatientData) -> tuple[int, float, str]:
    """Run inference and return (prediction, probability, risk_label)."""
    X = preprocess_input(data)
    prob = model.predict_proba(X)[0][1]
    pred = 1 if prob > THRESHOLD else 0

    if prob > 0.8:
        label = "High Risk"
    elif prob > THRESHOLD:
        label = "Moderate Risk"
    else:
        label = "Low Risk"

    return pred, prob, label


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    global model, encoders, scaler, train_columns

    # --- Load artefacts ---
    for path, name in [
        (MODEL_PATH, "model"),
        (ENCODERS_PATH, "encoders"),
        (SCALER_PATH, "scaler"),
        (TRAIN_COLUMNS_PATH, "train_columns"),
    ]:
        if not os.path.exists(path):
            raise RuntimeError(f"{name} not found at {path}")

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    train_columns = joblib.load(TRAIN_COLUMNS_PATH)

    # --- Initialise database ---
    init_database()

    yield

    # --- Cleanup (if needed) ---
    # (nothing special to clean up)


app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI backend for heart disease risk prediction and monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow the Streamlit dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Basic health-check endpoint."""
    db_ok = os.path.exists(DB_PATH)
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        database="connected" if db_ok else "missing",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: PatientData):
    """
    Run heart-disease prediction for a single patient.
    The request is logged to the monitoring database automatically.
    """
    start = time.perf_counter()

    try:
        pred, prob, label = predict_risk(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    elapsed = (time.perf_counter() - start) * 1000  # ms

    # Async-style logging (synchronous here — fast enough for SQLite WAL)
    try:
        log_prediction(data, prob, label, elapsed)
    except Exception as exc:
        # Logging failure should not break the prediction response
        print(f"[WARN] Failed to log prediction: {exc}")

    return PredictionResponse(
        prediction=pred,
        risk_label=label,
        probability=round(prob, 4),
        threshold=THRESHOLD,
        response_time_ms=round(elapsed, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/monitoring/logs", response_model=list[MonitoringLog], tags=["Monitoring"])
def get_logs(
    limit: int = Query(100, ge=1, le=5000, description="Number of recent logs"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """Retrieve recent prediction logs."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/monitoring/summary", response_model=MonitoringSummary, tags=["Monitoring"])
def get_summary():
    """Aggregated monitoring summary."""
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                                    AS total,
            SUM(CASE WHEN risk_label = 'High Risk'      THEN 1 ELSE 0 END) AS high,
            SUM(CASE WHEN risk_label = 'Moderate Risk'   THEN 1 ELSE 0 END) AS moderate,
            SUM(CASE WHEN risk_label = 'Low Risk'        THEN 1 ELSE 0 END) AS low,
            AVG(probability)                                             AS avg_prob
        FROM predictions
        """
    ).fetchone()
    conn.close()

    total = row["total"] or 0
    return MonitoringSummary(
        total_predictions=total,
        high_risk_count=row["high"] or 0,
        moderate_risk_count=row["moderate"] or 0,
        low_risk_count=row["low"] or 0,
        avg_probability=round(row["avg_prob"] or 0.0, 4),
        high_risk_pct=round((row["high"] or 0) / total * 100, 2) if total else 0.0,
    )


@app.delete("/monitoring/logs", tags=["Monitoring"])
def clear_logs():
    """Delete all prediction logs (for testing / maintenance)."""
    conn = get_db_connection()
    conn.execute("DELETE FROM predictions")
    conn.execute("VACUUM")
    conn.commit()
    conn.close()
    return {"message": "All logs cleared."}


