# app/main.py
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model/scaler: {e}")
    yield


app = FastAPI(title="Credit Card Fraud Detection API", lifespan=lifespan)


class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool


@app.get("/")
def root():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")

    df = pd.DataFrame([transaction.model_dump()])
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    prob = float(model.predict_proba(df)[0][1])  # joblib model needs predict_proba, not mlflow's .predict
    return PredictionResponse(
        fraud_probability=prob,
        is_fraud=prob > 0.5,
    )