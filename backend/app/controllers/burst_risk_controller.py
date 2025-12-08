# app/controllers/burst_risk_controller.py
from flask_restx import Namespace, Resource
from flask import jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from app import db
from app.models import Town, Simulation
from app.core.simulation.runner import run_simulation

ns = Namespace('burst-risk', description='Predict pipe burst risk using trained ML model')

# -----------------------------
# 1️⃣ Load trained model + scaler
# -----------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "core" / "ai" / "burst" / "pipe_burst_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Missing ML model file at {MODEL_PATH}")

data = joblib.load(MODEL_PATH)
if isinstance(data, dict):
    model = data["model"]
    scaler = data.get("scaler", None)
else:
    model = data
    scaler = None

print(f"✅ Loaded burst model from {MODEL_PATH.resolve()}")
if scaler:
    print("📏 Using trained scaler for normalization")

# -----------------------------
# 2️⃣ Features used for prediction
# -----------------------------
FEATURES = [
    "age",
    "mean_pressure",
    "std_pressure",
    "mean_velocity",
    "std_velocity",
    "mean_flow",
    "std_flow",
]

# -----------------------------
# 3️⃣ Endpoint definition
# -----------------------------
@ns.route('/<int:town_id>')
@ns.param('town_id', 'The town identifier')
class BurstRiskPredict(Resource):
    @ns.doc('predict_burst_risk')
    def get(self, town_id):
        """
        Predict burst probabilities for each pipe in the latest simulation of a town.
        Uses existing simulation data if available, otherwise runs one.
        """
        # Ensure town exists
        town = Town.query.get_or_404(town_id, description="Town not found.")

        # Try to find the latest simulation
        sim = Simulation.query.filter_by(town_id=town_id).order_by(Simulation.id.desc()).first()
        if sim and sim.details:
            print(f"📊 Using existing simulation for town {town_id} (Simulation ID {sim.id})")
            results = sim.details
        else:
            print(f"⚙️ No simulation found for town {town_id} — running a fresh one...")
            try:
                results = run_simulation(town.data)
                sim = Simulation(
                    title=f"Auto-generated for burst-risk {town_id}",
                    town_id=town_id,
                    details=results,
                )
                db.session.add(sim)
                db.session.commit()
            except Exception as e:
                ns.abort(500, f"Simulation failed: {str(e)}")

        # Extract simulation results
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

        if not rows:
            ns.abort(404, f"No pipe data found for town {town_id}")

        # Prepare for prediction
        df = pd.DataFrame(rows)
        df = df.dropna(subset=FEATURES)

        X = df[FEATURES]
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Predict burst probabilities
        probs = model.predict_proba(X_scaled)[:, 1]
        df["burst_risk"] = probs

        # --- Compute young high-risk anomalies ---
        YOUNG_AGE_THRESHOLD = 15  # consider pipes younger than this
        HIGH_RISK_THRESHOLD = 0.35  # absolute risk threshold
        AGE_BIN_SIZE = 5  # for contextual comparison

        # Create age bins and compute average risk per bin
        bins = pd.IntervalIndex.from_breaks(np.arange(0, 55, AGE_BIN_SIZE))
        df["age_bin"] = pd.cut(df["age"], bins)
        age_bins = df.groupby("age_bin", observed=False)["burst_risk"].mean().to_dict()

        def baseline_for_age(age: float) -> float:
            for interval, mean_val in age_bins.items():
                if age in interval:
                    return mean_val
            return np.mean(list(age_bins.values())) if age_bins else 0.0

        # 1️⃣ Young pipes with absolute high risk
        young_high_risk = (df["age"] <= YOUNG_AGE_THRESHOLD) & (df["burst_risk"] >= HIGH_RISK_THRESHOLD)

        # 2️⃣ Young pipes that are significantly above their group average
        young_relative_high = df.apply(
            lambda r: (r["age"] <= YOUNG_AGE_THRESHOLD)
                      and (r["burst_risk"] > baseline_for_age(r["age"]) + 0.15),
            axis=1
        )

        # Combine both conditions
        concerning_mask = young_high_risk | young_relative_high

        concerning = df[concerning_mask].sort_values("burst_risk", ascending=False)
        if "age_bin" in df.columns:
            df = df.drop(columns=["age_bin"])
        if "age_bin" in concerning.columns:
            concerning = concerning.drop(columns=["age_bin"])

        concerning_list = concerning.to_dict(orient="records")

        # Sort and summarize
        ranked = df.sort_values("burst_risk", ascending=False).to_dict(orient="records")

        return jsonify({
            "town_id": town_id,
            "simulation_id": sim.id if sim else None,
            "pipe_risk_summary": ranked[:50],
            "concerning_young_pipes": concerning_list,
            "average_risk": float(df["burst_risk"].mean()),
            "total_pipes": len(df),
        })
