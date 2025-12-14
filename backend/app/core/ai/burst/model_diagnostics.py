"""
model_diagnostics.py
--------------------
Interactive diagnostics and visualization for the burst-risk model.
You can run it standalone to inspect the model’s predictions,
feature importances, and calibration quality.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipe_burst_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "burst_training_data.csv")
SIMULATION_CACHE = os.path.join(
    os.path.dirname(__file__), "../data/simulation_cache.json"
)

# ----------------------------
# Feature configuration
# ----------------------------
FEATURES = [
    "age",
    "mean_pressure",
    "std_pressure",
    "mean_velocity",
    "std_velocity",
    "mean_flow",
    "std_flow",
]

# ----------------------------
# Utilities
# ----------------------------
def load_model(path=MODEL_PATH):
    """Load model + scaler from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

    data = joblib.load(path)
    # backward compatibility if file only contains the model
    if isinstance(data, dict):
        model = data["model"]
        scaler = data.get("scaler", None)
    else:
        model = data
        scaler = None

    print(f"Loaded model from {path}")
    return model, scaler


def analyze_distribution(df, probs):
    plt.figure(figsize=(7, 4))
    plt.hist(probs, bins=20, color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.title("Distribution of Predicted Burst Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Pipe Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def analyze_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(7, 4))
        sorted_idx = np.argsort(importances)
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color="teal")
        plt.title("Feature Importance")
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
    else:
        print("Model has no feature_importances_ attribute.")


def analyze_calibration(df, probs):
    if "burst" not in df.columns:
        print("No ground-truth 'burst' column found, skipping calibration.")
        return

    y_true = df["burst"].values
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    auc = roc_auc_score(y_true, probs)
    print(f"ROC-AUC Score: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def main():
    """Run the diagnostic interactively."""
    model, scaler = load_model()

    # --- Load either CSV or JSON simulation ---
    if os.path.exists(DATA_PATH):
        print(f"Using training dataset at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    elif os.path.exists(SIMULATION_CACHE):
        print(f"Using cached simulation results at {SIMULATION_CACHE}")
        with open(SIMULATION_CACHE, "r") as f:
            sim_data = json.load(f)

        results = sim_data.get("details", sim_data)
        pipe_params = results.get("pipe_parameters", {})
        pipe_vel = results.get("pipe_velocities", {})
        pipe_flow = results.get("pipe_flows", {})

        rows = []
        for pid, params in pipe_params.items():
            mean_p = float(params.get("mean_pressure_bar", 0.0))
            std_p = float(params.get("std_pressure_bar", 0.0))
            age = float(params.get("age", 0.0))
            vel = np.array(pipe_vel.get(f"v_mean_pipe_{pid}", [0]))
            flow = np.array(pipe_flow.get(f"flow_pipe_{pid}", [0]))
            mean_v = float(np.mean(vel))
            std_v = float(np.std(vel))
            mean_f = float(np.mean(flow))
            std_f = float(np.std(flow))
            rows.append({
                "pipe_id": pid,
                "age": age,
                "mean_pressure": mean_p,
                "std_pressure": std_p,
                "mean_velocity": mean_v,
                "std_velocity": std_v,
                "mean_flow": mean_f,
                "std_flow": std_f,
            })

        df = pd.DataFrame(rows)
        print(f"Loaded {len(df)} pipe rows from cached simulation")
    else:
        raise FileNotFoundError("No valid dataset found (CSV or JSON).")

    df = df.dropna(subset=FEATURES)
    X = df[FEATURES]
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    print("Predicting burst probabilities...")
    probs = model.predict_proba(X_scaled)[:, 1]
    df["burst_risk"] = probs

    print("Stats:")
    print(df["burst_risk"].describe())

    analyze_distribution(df, probs)
    analyze_feature_importance(model, FEATURES)
    analyze_calibration(df, probs)
    print("All plots generated. Close the figures to exit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
