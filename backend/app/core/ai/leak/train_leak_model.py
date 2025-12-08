import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle

DATA_PATH = Path("leak_training_data_v2.csv")

MODEL_PATH = Path("leak_distance_xgb_v2data.pkl")
SCALER_PATH = Path("leak_distance_scaler_v2data.pkl")
META_PATH = Path("leak_distance_meta_v2data.pkl")


# ========================================================
#        TRAIN MODEL FOR ONE TARGET TRANSFORMATION
# ========================================================
def train_single_target(df, target_col, feature_cols):
    print(f"\n📌 Training target: {target_col}")

    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)

    # Clean NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        objective="reg:squarederror",
        tree_method="hist",
    )

    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
        early_stopping_rounds=80,
    )

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "target": target_col,
    }


# ========================================================
#                MAIN TRAINING FUNCTION
# ========================================================
def train_distance_model():
    print("📥 Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # ----------------------------------------------------
    # Feature columns — ALL engineered features
    # ----------------------------------------------------
    feature_cols = [
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

        # Graph distances
        "graph_distance", "hydraulic_distance",
    ]

    # Ensure missing columns are zero-filled
    for col in feature_cols:
        if col not in df:
            print(f"⚠ Missing column in dataset: {col} — filling with zeros")
            df[col] = 0.0

    print(f"📊 Using {len(feature_cols)} input features")

    # ----------------------------------------------------
    # Try different target formulations
    # ----------------------------------------------------
    target_candidates = {
        "distance_to_leak": lambda x: x,
        "distance_log": np.expm1,     # inverse transform
        "distance_sqrt": lambda x: x**2,
    }

    results = {}

    for target_col in target_candidates.keys():
        if target_col not in df.columns:
            print(f"⚠ Target missing: {target_col}, skipping.")
            continue

        results[target_col] = train_single_target(df, target_col, feature_cols)

    # ----------------------------------------------------
    # Pick the best model (lowest RMSE)
    # ----------------------------------------------------
    best_key = min(results.keys(), key=lambda k: results[k]["rmse"])
    best = results[best_key]

    print("\n================ BEST MODEL SELECTED ================")
    print(f"🎯 Best target: {best['target']}")
    print(f"MAE  = {best['mae']:.4f}")
    print(f"RMSE = {best['rmse']:.4f}")
    print(f"R²   = {best['r2']:.4f}")
    print("=====================================================\n")

    # ----------------------------------------------------
    # Save model, scaler, metadata
    # ----------------------------------------------------
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best["model"], f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(best["scaler"], f)

    with open(META_PATH, "wb") as f:
        pickle.dump({
            "target": best["target"],
            "feature_cols": feature_cols,
            "inverse_transform": target_candidates[best["target"]],
        }, f)

    print(f"💾 Model saved to: {MODEL_PATH}")
    print(f"💾 Scaler saved to: {SCALER_PATH}")
    print(f"💾 Metadata saved to: {META_PATH}")


# ========================================================
#                       MAIN
# ========================================================
def main():
    train_distance_model()


if __name__ == "__main__":
    main()
