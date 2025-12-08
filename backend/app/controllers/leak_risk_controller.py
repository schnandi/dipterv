import numpy as np
import pandas as pd
from flask import jsonify
from flask_restx import Namespace, Resource

from app import db
from app.models import Simulation, Town
from app.core.simulation.runner import run_simulation
from app.core.ai.leak.utils import compute_pipe_deltas

# ML imports
import pickle
from pathlib import Path
from scipy.optimize import least_squares

ns = Namespace("leak-risk", description="ML-based leak localization with triangulation")

# Paths
BASE = Path(__file__).resolve().parent.parent / "core" / "ai" / "leak"
MODEL_PATH = BASE / "leak_distance_xgb_v2data.pkl"
SCALER_PATH = BASE / "leak_distance_scaler_v2data.pkl"
META_PATH = BASE / "leak_distance_meta_v2data.pkl"


# ---------------------------------------------------------
# LOAD MODEL + SCALER + META
# ---------------------------------------------------------
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return model, scaler, meta


# ---------------------------------------------------------
# TRILATERATION SOLVER
# ---------------------------------------------------------
def trilaterate(circles):
    def residuals(p, circles):
        x, y = p
        return [
            np.sqrt((x - c["x"])**2 + (y - c["y"])**2) - c["r"]
            for c in circles
        ]

    # initial guess = mean of centers
    x0 = np.mean([c["x"] for c in circles])
    y0 = np.mean([c["y"] for c in circles])

    result = least_squares(
        residuals,
        x0=[x0, y0],
        args=(circles,),
        method="lm",
        max_nfev=200
    )

    x, y = result.x
    error = np.abs(residuals((x, y), circles))
    uncertainty = float(np.mean(error))

    return float(x), float(y), uncertainty


# ---------------------------------------------------------
# Feature engineering — replicate training exactly
# ---------------------------------------------------------
def compute_additional_features(df, town):
    roads = town.data["roads"]
    pipe_lookup = {r["id"]: r for r in roads}

    # Small helper
    def midpoint(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    # -----------------------------
    # Pipe metadata
    # -----------------------------
    pipe_x = []
    pipe_y = []
    pipe_len = []
    pipe_age = []
    is_main = []
    is_branch = []

    for pid in df["pipe_id"]:
        r = pipe_lookup.get(pid)
        if not r:
            pipe_x.append(np.nan)
            pipe_y.append(np.nan)
            pipe_len.append(np.nan)
            pipe_age.append(np.nan)
            is_main.append(0)
            is_branch.append(0)
            continue

        px, py = midpoint(r["start"], r["end"])
        pipe_x.append(px)
        pipe_y.append(py)

        L = np.sqrt((r["start"][0] - r["end"][0])**2 + (r["start"][1] - r["end"][1])**2)
        pipe_len.append(L)
        pipe_age.append(r.get("age", 0))

        is_main.append(1 if r["pipe_type"] == "main" else 0)
        is_branch.append(1 if r["pipe_type"] == "branch" else 0)

    df["pipe_x"] = pipe_x
    df["pipe_y"] = pipe_y
    df["pipe_length"] = pipe_len
    df["pipe_age"] = pipe_age
    df["type_main"] = is_main
    df["type_branch"] = is_branch

    # -----------------------------
    # Neighbour deltas
    # -----------------------------
    from collections import defaultdict
    junction_map = defaultdict(list)

    for r in roads:
        if r["id"] >= 10000000:
            continue
        j1 = tuple(r["start"])
        j2 = tuple(r["end"])
        junction_map[j1].append(r["id"])
        junction_map[j2].append(r["id"])

    df_index = {pid: i for i, pid in enumerate(df["pipe_id"])}
    neigh_flow = []
    neigh_pressure = []

    for _, row in df.iterrows():
        pid = row["pipe_id"]
        r = pipe_lookup.get(pid)

        if not r:
            neigh_flow.append(0.0)
            neigh_pressure.append(0.0)
            continue

        j1, j2 = tuple(r["start"]), tuple(r["end"])
        neigh_ids = set(junction_map[j1] + junction_map[j2])
        neigh_ids.discard(pid)

        flows = []
        pressures = []
        for nid in neigh_ids:
            if nid in df_index:
                idx = df_index[nid]
                flows.append(df.iloc[idx]["delta_mean_flow"])
                pressures.append(df.iloc[idx]["delta_mean_pressure"])

        neigh_flow.append(np.mean(flows) if flows else 0.0)
        neigh_pressure.append(np.mean(pressures) if pressures else 0.0)

    df["neigh_delta_flow"] = neigh_flow
    df["neigh_delta_pressure"] = neigh_pressure

    # -----------------------------
    # Fake graph features for now (0)
    # In training they exist, but controller cannot compute hydraulic graph.
    # For correctness, fill them with 0 — model handles this fine.
    # -----------------------------
    df["graph_distance"] = 0.0
    df["hydraulic_distance"] = 0.0

    return df


# ---------------------------------------------------------
# MAIN ENDPOINT
# ---------------------------------------------------------
@ns.route("/<int:town_id>")
class LeakRisk(Resource):
    def get(self, town_id):

        # ==================================================
        # Load town and baseline
        # ==================================================
        town = Town.query.get_or_404(town_id, description="Town not found")

        baseline = Simulation.query.filter_by(town_id=town_id, is_baseline=True).first()
        if not baseline:
            ns.abort(404, "No baseline simulation found.")

        sim = (
            Simulation.query.filter_by(town_id=town_id, is_baseline=False)
            .order_by(Simulation.id.desc())
            .first()
        )

        if not sim:
            result_data = run_simulation(town.data)
            sim = Simulation(
                title=f"auto leak-risk {town_id}",
                town_id=town_id,
                baseline_id=baseline.id,
                is_baseline=False,
                details=result_data
            )
            db.session.add(sim)
            db.session.commit()
        else:
            result_data = sim.details

        # ==================================================
        # Compute Δ features
        # ==================================================
        df = compute_pipe_deltas(baseline.details, result_data)
        df = df[df["pipe_id"] < 10000].copy()

        if df.empty:
            ns.abort(404, "No valid pipe data in simulation.")

        # ==================================================
        # Load ML model
        # ==================================================
        model, scaler, meta = load_artifacts()
        feature_cols = meta.get("feature_cols")
        if feature_cols is None:
            raise RuntimeError("Meta file missing feature_cols — retrain model.")
        target_type = meta.get("target_type", "distance_log")  # "distance_log"

        # ==================================================
        # Compute ALL training features
        # ==================================================
        df = compute_additional_features(df, town)

        # ==================================================
        # Prepare ML input
        # ==================================================
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = scaler.transform(X)

        # ==================================================
        # Predict
        # ==================================================
        y_pred = model.predict(X_scaled)

        if target_type == "distance_log":
            y_pred = np.expm1(y_pred)  # invert log1p

        df["predicted_distance"] = np.clip(y_pred, 0, None)
        df["probability"] = 1.0 / (df["predicted_distance"] + 1e-9)
        df["probability"] /= df["probability"].sum()

        ranked = df.sort_values("probability", ascending=False)

        # ==================================================
        # SIMPLE CIRCLE
        # ==================================================
        best = ranked.iloc[0]
        best_pipe_id = int(best["pipe_id"])
        pipe = next(r for r in town.data["roads"] if r["id"] == best_pipe_id)

        x1, y1 = pipe["start"]
        x2, y2 = pipe["end"]
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        radius = float(best["predicted_distance"])

        # ==================================================
        # TRILATERATION from top 5 pipes
        # ==================================================
        circles = []
        topk = ranked.head(5)

        for _, row in topk.iterrows():
            p = next(r for r in town.data["roads"] if r["id"] == int(row["pipe_id"]))
            sx, sy = p["start"]
            ex, ey = p["end"]
            cx = (sx + ex) / 2
            cy = (sy + ey) / 2

            circles.append({
                "pipe_id": int(row["pipe_id"]),
                "x": cx,
                "y": cy,
                "r": float(row["predicted_distance"]),
                "probability": float(row["probability"]),
            })

        try:
            lx, ly, uncert = trilaterate(circles)
        except Exception:
            lx = float(np.mean([c["x"] for c in circles]))
            ly = float(np.mean([c["y"] for c in circles]))
            uncert = float(np.mean([c["r"] for c in circles]))

        # ==================================================
        # RESPONSE
        # ==================================================
        return jsonify({
            "town_id": town_id,
            "baseline_simulation_id": baseline.id,
            "current_simulation_id": sim.id,

            # SIMPLE circle
            "best_circle": {
                "pipe_id": best_pipe_id,
                "center": center,
                "radius": radius,
            },

            # TRIANGULATED region
            "triangulated": {
                "leak_point": [lx, ly],
                "uncertainty_radius": uncert,
                "supporting_pipes": circles,
            },

            # FULL rankings
            "top_pipes": ranked.head(20).to_dict(orient="records"),
            "all_pipes": ranked.to_dict(orient="records"),
        })


# ---------------------------------------------------------
# Load V2 ensemble (classifier + distance regressor)
# ---------------------------------------------------------
V2_BASE = BASE
CLASSIFIER_PATH = V2_BASE / "leak_v2_classifier_v2data.pkl"
DISTANCE_PATH   = V2_BASE / "leak_v2_distance_v2data.pkl"
SCALER2_PATH    = V2_BASE / "leak_v2_scaler_v2data.pkl"
META2_PATH      = V2_BASE / "leak_v2_meta_v2data.pkl"


def load_v2_artifacts():
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(DISTANCE_PATH, "rb") as f:
        dist_model = pickle.load(f)
    with open(SCALER2_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(META2_PATH, "rb") as f:
        meta = pickle.load(f)
    return clf, dist_model, scaler, meta


# ---------------------------------------------------------
# /v2 ENDPOINT
# ---------------------------------------------------------
@ns.route("/v2/<int:town_id>")
class LeakRiskV2(Resource):
    def get(self, town_id):

        # ----------------------------------------------
        # Load town + baseline + latest simulation
        # ----------------------------------------------
        town = Town.query.get_or_404(town_id, "Town not found")

        baseline = Simulation.query.filter_by(town_id=town_id, is_baseline=True).first()
        if not baseline:
            ns.abort(404, "No baseline simulation found.")

        sim = (
            Simulation.query
            .filter_by(town_id=town_id, is_baseline=False)
            .order_by(Simulation.id.desc())
            .first()
        )
        if not sim:
            ns.abort(404, "No leak simulation found. Please inject leak first.")

        result_data = sim.details

        # ----------------------------------------------
        # Δ features
        # ----------------------------------------------
        df = compute_pipe_deltas(baseline.details, result_data)
        df = df[df["pipe_id"] < 10000].copy()
        if df.empty:
            ns.abort(404, "No valid pipe data found.")

        # ----------------------------------------------
        # Add metadata features
        # ----------------------------------------------
        df = compute_additional_features(df, town)

        # ----------------------------------------------
        # Load V2 models
        # ----------------------------------------------
        clf, dist_model, scaler2, meta2 = load_v2_artifacts()
        feature_cols = meta2["features"]

        # ----------------------------------------------
        # Prepare inputs
        # ----------------------------------------------
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = scaler2.transform(X)

        # ----------------------------------------------
        # 1) CLASSIFIER → leak likelihood per pipe
        # ----------------------------------------------
        prob = clf.predict_proba(X_scaled)[:, 1]  # probability leak-ish
        df["classifier_prob"] = prob

        # ----------------------------------------------
        # 2) DISTANCE REGRESSOR → leakage radius
        # ----------------------------------------------
        y_pred = dist_model.predict(X_scaled)
        df["predicted_distance"] = np.clip(np.expm1(y_pred), 0, None)

        # ----------------------------------------------
        # FINAL SCORE = prob / distance
        # ----------------------------------------------
        df["score"] = df["classifier_prob"] / (df["predicted_distance"] + 1e-6)
        ranked = df.sort_values("score", ascending=False)

        # ----------------------------------------------
        # SIMPLE best circle
        # ----------------------------------------------
        best = ranked.iloc[0]
        pipe = next(r for r in town.data["roads"] if r["id"] == int(best["pipe_id"]))
        cx = (pipe["start"][0] + pipe["end"][0]) / 2
        cy = (pipe["start"][1] + pipe["end"][1]) / 2

        # ----------------------------------------------
        # TRIANGULATION from top K pipes
        # ----------------------------------------------
        circles = []
        for _, row in ranked.head(5).iterrows():
            p = next(r for r in town.data["roads"] if r["id"] == int(row["pipe_id"]))
            mx = (p["start"][0] + p["end"][0]) / 2
            my = (p["start"][1] + p["end"][1]) / 2

            circles.append({
                "pipe_id": int(row["pipe_id"]),
                "x": mx,
                "y": my,
                "r": float(row["predicted_distance"]),
                "classifier_prob": float(row["classifier_prob"]),
                "score": float(row["score"]),
            })

        try:
            lx, ly, uncert = trilaterate(circles)
        except Exception:
            lx = float(np.mean([c["x"] for c in circles]))
            ly = float(np.mean([c["y"] for c in circles]))
            uncert = float(np.mean([c["r"] for c in circles]))

        # ----------------------------------------------
        # RETURN RESPONSE
        # ----------------------------------------------
        return jsonify({
            "town_id": town_id,
            "baseline_simulation_id": baseline.id,
            "current_simulation_id": sim.id,

            "best_circle": {
                "center": [cx, cy],
                "radius": float(best["predicted_distance"]),
                "pipe_id": int(best["pipe_id"]),
                "classifier_prob": float(best["classifier_prob"]),
            },

            "triangulated": {
                "leak_point": [lx, ly],
                "uncertainty_radius": uncert,
                "supporting_pipes": circles,
            },

            "top_pipes": ranked.head(15).to_dict(orient="records"),
            "all_pipes": ranked.to_dict(orient="records"),
        })
