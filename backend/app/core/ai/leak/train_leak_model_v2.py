# ============================================================
#   LEAK MODEL V2 — CLASSIFIER + DISTANCE REGRESSOR ENSEMBLE
# ============================================================

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
)
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


DATA_PATH = Path("leak_training_data_v2.csv")

MODEL_CLASSIFIER = Path("leak_v2_classifier_v2data.pkl")
MODEL_DISTANCE = Path("leak_v2_distance_v2data.pkl")
SCALER_PATH = Path("leak_v2_scaler_v2data.pkl")
META_PATH = Path("leak_v2_meta_v2data.pkl")


# ============================================================
#              FEATURES USED BY BOTH MODELS
# ============================================================

FEATURE_COLS = [
    # Δ features
    "delta_mean_flow", "delta_std_flow",
    "delta_mean_velocity", "delta_std_velocity",
    "delta_mean_pressure", "delta_std_pressure",

    # Neighbour features
    "neigh_delta_flow", "neigh_delta_pressure",

    # Pipe metadata
    "pipe_length", "pipe_age",
    "type_main", "type_branch",

    # Coordinates
    "pipe_x", "pipe_y",

    # Graph metrics
    "graph_distance", "hydraulic_distance",
]


# ============================================================
#                      LOAD DATASET
# ============================================================
def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # Drop split-pipes (id > 1e8)
    df = df[df["pipe_id"] < 1e8].copy()

    # Create classification label
    df["is_leak_pipe"] = (df["pipe_id"] == df["leak_pipe_id"]).astype(int)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


# ============================================================
#                  TRAIN CLASSIFIER
# ============================================================
def train_classifier(df, X_train, X_test, y_train, y_test):

    print("🔎 Training classifier...")

    clf = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print(f"   ✔ Accuracy = {acc:.3f}")
    print(f"   ✔ ROC AUC = {auc:.3f}")

    return clf


# ============================================================
#               TRAIN DISTANCE REGRESSOR (log)
# ============================================================
def train_distance_regressor(X_train, X_test, y_train, y_test):

    print("📏 Training distance regressor...")

    model = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.1,
        objective="reg:squarederror",
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=60,
    )

    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"   ✔ Distance RMSE = {rmse:.3f}")

    return model


# ============================================================
#                       MAIN TRAIN
# ============================================================
def main():

    print(f"📥 Loading dataset: {DATA_PATH}")
    df = load_dataset()

    # ------------------------------
    # Select features + targets
    # ------------------------------
    X = df[FEATURE_COLS].astype(float)
    y_class = df["is_leak_pipe"].astype(int)
    y_dist = df["distance_log"].astype(float)

    # ------------------------------
    # Scaling
    # ------------------------------
    X_train, X_test, yC_train, yC_test, yD_train, yD_test = train_test_split(
        X, y_class, y_dist, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------
    # Train models
    # ------------------------------
    clf = train_classifier(df, X_train_scaled, X_test_scaled, yC_train, yC_test)
    dist_model = train_distance_regressor(X_train_scaled, X_test_scaled, yD_train, yD_test)

    # ------------------------------
    # Save everything
    # ------------------------------
    print("\n💾 Saving V2 ensemble models...")
    with open(MODEL_CLASSIFIER, "wb") as f:
        pickle.dump(clf, f)

    with open(MODEL_DISTANCE, "wb") as f:
        pickle.dump(dist_model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    with open(META_PATH, "wb") as f:
        pickle.dump({
            "features": FEATURE_COLS,
            "distance_target": "distance_log",
        }, f)

    print("   ✔ Done!")
    print("================================================")


if __name__ == "__main__":
    main()
