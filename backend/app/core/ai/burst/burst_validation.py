# burst_validation.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("burst_training_data.csv")
MODEL_PATH = Path("pipe_burst_model.pkl")

FEATURES = [
    "age",
    "mean_pressure",
    "std_pressure",
    "mean_velocity",
    "std_velocity",
    "mean_flow",
    "std_flow",
]
TARGET = "burst"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Load model and scaler
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]

    X_scaled = scaler.transform(X)

    # Predict
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    print("Accuracy:", acc)
    print("ROC-AUC:", auc)
    print("Confusion matrix:\n", cm)

    # Plot ROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Pipe Burst Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("burst_roc.png", dpi=200)
    print("ROC curve saved as burst_roc.png")

if __name__ == "__main__":
    main()
