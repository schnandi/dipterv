import numpy as np
import pandas as pd


def extract_pipe_features(results):
    """Extract per-pipe mean/std features and metadata from a simulation result dict."""
    pipe_params = results.get("pipe_parameters", {})
    pipe_vel = results.get("pipe_velocities", {})
    pipe_flow = results.get("pipe_flows", {})

    data = {}
    for pid_str, params in pipe_params.items():
        pid = int(pid_str)
        mean_p = float(params.get("mean_pressure_bar", 0.0))
        std_p = float(params.get("std_pressure_bar", 0.0))
        pipe_type = params.get("pipe_type", "unknown")

        vel = np.array(pipe_vel.get(f"v_mean_pipe_{pid}", [0.0]))
        flow = np.array(pipe_flow.get(f"flow_pipe_{pid}", [0.0]))

        data[pid] = {
            "pipe_type": pipe_type,
            "mean_pressure": mean_p,
            "std_pressure": std_p,
            "mean_velocity": float(np.mean(vel)),
            "std_velocity": float(np.std(vel)),
            "mean_flow": float(np.mean(flow)),
            "std_flow": float(np.std(flow)),
        }
    return data


def compute_pipe_deltas(baseline_results, leak_results):
    """
    Compare two simulation result dicts (baseline vs leak) and compute relative Δ features.
    Returns a DataFrame with one row per pipe_id.
    """
    base = extract_pipe_features(baseline_results)
    leak = extract_pipe_features(leak_results)

    rows = []
    all_pipe_ids = set(base.keys()) | set(leak.keys())

    for pid in sorted({int(p) for p in all_pipe_ids if str(p).isdigit()}):
        b = base.get(pid, {})
        l = leak.get(pid, {})
        if not b and not l:
            continue

        # Compute relative deltas
        def rel_delta(key):
            base_val = abs(b.get(key, 0)) + 1e-6
            return (l.get(key, 0) - b.get(key, 0)) / base_val

        row = {
            "pipe_id": pid,
            "pipe_type": b.get("pipe_type", l.get("pipe_type", "unknown")),
            "delta_mean_flow": rel_delta("mean_flow"),
            "delta_std_flow": rel_delta("std_flow"),
            "delta_mean_velocity": rel_delta("mean_velocity"),
            "delta_std_velocity": rel_delta("std_velocity"),
            "delta_mean_pressure": rel_delta("mean_pressure"),
            "delta_std_pressure": rel_delta("std_pressure"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_leak_scores(df):
    """Compute heuristic leak scores with normalization and filtering."""
    # 1️⃣ Filter out artificial and building-connection pipes
    df = df[df["pipe_id"] < 10000].copy()
    if "pipe_type" in df.columns:
        df = df[df["pipe_type"] != "building connection"].copy()

    # 2️⃣ Compute a combined leak score (weighted sum of relative deltas)
    df["leak_score"] = (
        df["delta_mean_flow"].abs() * 2.0 +
        df["delta_std_pressure"].abs() +
        df["delta_mean_pressure"].abs() * 0.5
    )

    # 3️⃣ Normalize to [0, 1]
    max_score = df["leak_score"].abs().max()
    df["leak_score"] = df["leak_score"] / max(max_score, 1e-6)

    # 4️⃣ Sort and return
    return df.sort_values("leak_score", ascending=False).reset_index(drop=True)
