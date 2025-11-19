import numpy as np
import pandas as pd
from pathlib import Path
from core.simulation.runner import run_simulation

# -----------------------------
# 1️⃣ Heuristic Burst Function
# -----------------------------
def burst_probability(pipe_row):
    """
    Calculates the synthetic burst probability for a pipe based on its physical & dynamic parameters.
    Adds stochastic 'random event' bursts for realism.

    pipe_row: dict with keys
      - age, mean_pressure, std_pressure, mean_velocity, std_velocity, mean_flow, std_flow, diameter_m
    Returns:
      Probability [0, 1]
    """
    # --- Normalized feature factors ---
    age_factor = np.clip(pipe_row.get("age", 0) / 100, 0, 1)
    stress_factor = np.clip(pipe_row.get("std_pressure", 0) / 0.5, 0, 2)      # pressure instability
    velocity_factor = np.clip(pipe_row.get("mean_velocity", 0) / 2.0, 0, 1)   # high velocity risk
    flow_factor = np.clip(pipe_row.get("std_flow", 0) / 0.2, 0, 2)            # unsteady flow
    diameter_factor = 0.2 if pipe_row.get("diameter_m", 0.1) < 0.06 else 0.0  # thinner pipes more fragile
    random_noise = np.random.normal(0, 0.1)

    # --- Nonlinear risk composition ---
    risk_score = (
        3.0 * age_factor**2 +
        1.8 * stress_factor +
        1.2 * velocity_factor +
        0.8 * flow_factor +
        diameter_factor +
        random_noise
    )

    # --- Add rare "sudden failure" events ---
    # Simulate unpredictable bursts caused by transient hydraulic shocks or material defects
    if pipe_row.get("std_pressure", 0) > 1 or pipe_row.get("age", 0) > 42:
        risk_score += 0.8   # strong influence from extreme conditions
    elif np.random.rand() < 0.005:  # 0.5% chance of random manufacturing defect
        risk_score += np.random.uniform(0.5, 1.5)

    # --- Convert to probability ---
    # The threshold (3.3–3.6) roughly tunes burst ratio to ~7–10%
    p = 1 / (1 + np.exp(-(risk_score - 3.4)))

    return float(np.clip(p, 0, 1))


# -----------------------------
# 2️⃣ Single Simulation → Pipe Stats
# -----------------------------
def analyze_simulation(city_data, town_id):
    results = run_simulation(city_data)

    # Extract relevant arrays
    pipe_vel = results["pipe_velocities"]
    pipe_flow = results["pipe_flows"]
    pipe_params = results["pipe_parameters"]
    pressures = results["junction_pressures"]

    pipe_ids = list(pipe_params.keys())
    rows = []

    for pid in pipe_ids:
        # Defensive fallback: empty lists if missing
        vel = np.array(pipe_vel.get(f"v_mean_pipe_{pid}", [0]))
        flow = np.array(pipe_flow.get(f"flow_pipe_{pid}", [0]))

        mean_v = float(np.mean(vel))
        std_v = float(np.std(vel))
        mean_f = float(np.mean(flow))
        std_f = float(np.std(flow))

        # Approximate pipe pressure as avg of its endpoints
        p_from = pressures.get(f"p_bar_junction_{pipe_params[pid]['from_junction']}", [0])
        p_to = pressures.get(f"p_bar_junction_{pipe_params[pid]['to_junction']}", [0])
        mean_p = float((np.mean(p_from) + np.mean(p_to)) / 2)
        std_p = float((np.std(p_from) + np.std(p_to)) / 2)

        age = pipe_params[pid].get("age", 50) if "age" in pipe_params[pid] else np.random.randint(1, 80)
        p_burst = burst_probability({
            "age": age,
            "mean_pressure": mean_p,
            "std_pressure": std_p,
            "mean_velocity": mean_v,
            "std_velocity": std_v,
            "mean_flow": mean_f,
            "std_flow": std_f,
            "diameter_m": pipe_params[pid].get("diameter_m", 0.05)
        })
        burst = np.random.rand() < p_burst

        rows.append({
            "town_id": town_id,
            "pipe_id": pid,
            "age": age,
            "mean_pressure": mean_p,
            "std_pressure": std_p,
            "mean_velocity": mean_v,
            "std_velocity": std_v,
            "mean_flow": mean_f,
            "std_flow": std_f,
            "burst": int(burst)
        })

    return pd.DataFrame(rows)


# -----------------------------
# 3️⃣ Multi-town Dataset Builder
# -----------------------------
def generate_burst_dataset(city_generator_func, n_towns=20, out_path="burst_training_data.csv"):
    """
    Generate synthetic burst dataset using multiple towns.
    city_generator_func: function returning a fresh city_data dict each call
    """
    all_dfs = []
    for i in range(n_towns):
        print(f"🏙️ Simulating town {i+1}/{n_towns}...")
        city_data = city_generator_func()
        df = analyze_simulation(city_data, i)
        all_dfs.append(df)

    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset.to_csv(out_path, index=False)
    print(f"✅ Dataset saved to {Path(out_path).resolve()} ({len(dataset)} rows)")
    return dataset
