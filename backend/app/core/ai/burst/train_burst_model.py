"""
Train and validate pipe burst prediction model.
Uses Random Forest on synthetic burst dataset with SMOTE balancing.
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = Path("burst_training_data.csv")
MODEL_PATH = Path("pipe_burst_model.pkl")

# ----------------------------
# Feature setup
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
TARGET = "burst"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run generate_burst_data.py first."
        )

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # --- Basic sanity checks ---
    print(f"Total rows: {len(df)}, burst ratio: {df[TARGET].mean():.3f}")

    # Drop any rows missing core features
    df = df.dropna(subset=FEATURES + [TARGET])

    # --- Feature / target split ---
    X = df[FEATURES]
    y = df[TARGET]

    # --- Split data (stratified for balanced validation) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # --- Scale features (important for SMOTE) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Apply SMOTE to handle imbalance ---
    print("Balancing dataset with SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print(f"Before SMOTE: {sum(y_train)} bursts / {len(y_train)} samples")
    print(f"After SMOTE:  {sum(y_train_res)} bursts / {len(y_train_res)} samples")

    # --- Train model ---
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,                # let it find deeper relations
        min_samples_leaf=10,           # smoother decision boundaries
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    model.fit(X_train_res, y_train_res)

    # --- Validation ---
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {auc:.3f}")

    # --- Feature importances ---
    print("\nFeature Importances:")
    for name, imp in zip(FEATURES, model.feature_importances_):
        print(f"  {name:20s}: {imp:.3f}")

    # --- Save model + scaler ---
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"\nModel + scaler saved to {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
