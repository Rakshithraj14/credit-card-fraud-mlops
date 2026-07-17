# src/train.py
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)

from src.data_loader import load_data, split_data

def train(n_estimators: int = 100, max_depth: int | None = None):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Only Time and Amount need scaling — V1-V28 are already PCA components
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    mlflow.set_experiment("credit-card-fraud")
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
        })

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1, class_weight="balanced",
        )
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "pr_auc": average_precision_score(y_test, probs),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", registered_model_name="fraud-detector")

        # Also save locally to models/ for quick local testing / Docker fallback
        joblib.dump(model, "models/model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        print("Metrics:", metrics)
        return model, scaler, metrics

if __name__ == "__main__":
    train(n_estimators=200, max_depth=10)