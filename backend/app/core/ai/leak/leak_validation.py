# leak_validation.py  (FIXED)

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    roc_curve
)
import matplotlib.pyplot as plt

DATA_PATH = Path("leak_training_data.csv")
CLASSIFIER_PATH = Path("leak_v2_classifier_v2data.pkl")
REGRESSOR_PATH = Path("leak_v2_distance_v2data.pkl")
SCALER_PATH = Path("leak_v2_scaler_v2data.pkl")

FEATURES = [
    "delta_mean_flow", "delta_std_flow",
    "delta_mean_velocity", "delta_std_velocity",
    "delta_mean_pressure", "delta_std_pressure",
    "neigh_delta_flow", "neigh_delta_pressure",
    "pipe_length", "pipe_age",
    "type_main", "type_branch",
    "pipe_x", "pipe_y",
    "graph_distance", "hydraulic_distance",
]

def main():
    df = pd.read_csv(DATA_PATH)

    # Remove split pipes
    df = df[df["pipe_id"] < 1e8].copy()

    df["is_leak_pipe"] = (df["pipe_id"] == df["leak_pipe_id"]).astype(int)

    # --- Extract features ---
    X = df[FEATURES].astype(float)

    # --- HARDEN: Remove all NaN & Inf ---
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Debug: check if anything is still NaN
    if X.isna().any().any():
        print("ERROR: NaN remained in X after cleaning!")
        print(X.isna().sum())
        return

    y_class = df["is_leak_pipe"]
    y_dist = df["distance_log"].fillna(0.0)

    # --- Load models ---
    clf = pickle.load(open(CLASSIFIER_PATH, "rb"))
    reg = pickle.load(open(REGRESSOR_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))

    X_scaled = scaler.transform(X)

    # --- CLASSIFIER VALIDATION ---
    proba = clf.predict_proba(X_scaled)[:, 1]
    preds = clf.predict(X_scaled)

    auc = roc_auc_score(y_class, proba)
    acc = accuracy_score(y_class, preds)

    print("Classifier AUC:", auc)
    print("Classifier Accuracy:", acc)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_class, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Leak Classifier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("leak_classifier_roc.png", dpi=200)

    # --- REGRESSOR VALIDATION ---
    pred_dist = reg.predict(X_scaled)

    rmse = np.sqrt(mean_squared_error(y_dist, pred_dist))
    print("Distance RMSE:", rmse)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_dist, pred_dist, alpha=0.3, s=8)
    plt.xlabel("True distance (log)")
    plt.ylabel("Predicted distance (log)")
    plt.title("Leak Distance Regressor – Predicted vs True")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("leak_distance_scatter.png", dpi=200)

if __name__ == "__main__":
    main()
