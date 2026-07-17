# Credit Card Fraud Detection — MLOps

An end-to-end MLOps pipeline for detecting fraudulent credit card transactions: data
versioning with DVC, experiment tracking with MLflow, a FastAPI serving layer, and
CI via GitHub Actions.

## Overview

The [dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) contains 284,807 European
credit card transactions from September 2013, of which only 492 (0.173%) are fraudulent.
Features `V1`–`V28` are PCA components (original data is anonymized); `Time` and `Amount`
are the only raw, unscaled features.

Given the severe class imbalance, the project favors precision/recall/PR-AUC over
accuracy when evaluating models — see [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
for the full exploratory analysis.

## Project structure

```
├── app/                    # FastAPI inference service
│   └── main.py
├── src/                    # Training pipeline
│   ├── data_loader.py      # Load + train/test split
│   └── train.py            # Train RandomForest, log to MLflow, save artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── tests/
│   └── test_api.py         # API tests (pytest)
├── data/                   # Dataset (DVC-tracked, not committed)
├── models/                 # model.pkl, scaler.pkl (DVC-tracked, not committed)
├── Dockerfile
├── requirements.txt        # Full dev/training environment
├── requirements-api.txt    # Minimal runtime deps for the API/Docker image
└── .github/workflows/ci.yml
```

## Setup

```bash
git clone https://github.com/Rakshithraj14/credit-card-fraud-mlops.git
cd credit-card-fraud-mlops
python -m venv .venv
.venv\Scripts\activate      
pip install -r requirements.txt
```

Data and model artifacts are tracked with [DVC](https://dvc.org/) rather than committed
to git:

```bash
dvc pull
```

## Training

```bash
python -m src.train
```

This trains a `RandomForestClassifier` (with `class_weight="balanced"` to account for
the imbalance), logs parameters/metrics/the model to MLflow, and writes
`models/model.pkl` and `models/scaler.pkl` for the API to serve.

View experiment runs with:

```bash
mlflow ui
```

## Serving the API

Locally:

```bash
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

With Docker:

```bash
docker build -t fraud-detector-api .
docker run -p 8000:8000 fraud-detector-api
```

### Endpoints

| Method | Path       | Description                                            |
|--------|------------|--------------------------------------------------------|
| GET    | `/`        | Service status                                         |
| GET    | `/health`  | Health check — reports whether model/scaler are loaded |
| POST   | `/predict` | Predict fraud probability for a transaction            |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 100.0, "V1": -1.36, "V2": -0.07, ..., "V28": -0.02, "Amount": 149.62}'
```

Response:

```json
{
  "fraud_probability": 0.02,
  "is_fraud": false
}
```

## Testing

```bash
pytest tests/ -v
```

## CI

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every
push/PR to `dev` and `main`:

- **test** — installs dependencies, trains a small synthetic placeholder model (so the
  pipeline doesn't depend on pulling the real model from a private DVC remote in CI),
  and runs the pytest suite.
- **docker-build** — generates the same placeholder model and verifies the Docker image
  builds successfully.

## Tech stack

- **Model**: scikit-learn (RandomForest)
- **Experiment tracking**: MLflow
- **Data/model versioning**: DVC
- **Serving**: FastAPI + Uvicorn
- **Containerization**: Docker
- **CI**: GitHub Actions
