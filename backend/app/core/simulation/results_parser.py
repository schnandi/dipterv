import numpy as np
import pandas as pd


def _collect(df, prefix):
    if prefix:
        return {f"{prefix}_{col}": df[col].tolist() for col in df.columns}
    else:
        return {col: df[col].tolist() for col in df.columns}


def format_results(net, ow, city_data=None):
    timestamps = pd.date_range(
        "2025-01-01", periods=len(ow.time_steps), freq="15min"
    ).strftime("%Y-%m-%d %H:%M:%S").tolist()

    roads_by_id = {r["id"]: r for r in (city_data.get("roads", []) if city_data else [])}

    # --- identical param info as old ---
    pipe_params = {
        int(i): {
            "name": row["name"],
            "from_junction": int(row["from_junction"]),
            "to_junction": int(row["to_junction"]),
            "length_m": float(row["length_km"] * 1e3),
            "diameter_m": float(row["diameter_m"]),
            "k_mm": float(row["k_mm"]),
            "age": float(roads_by_id.get(i, {}).get("age", 0.0)),
            "pipe_type": roads_by_id.get(i, {}).get("pipe_type", None),
        }
        for i, row in net.pipe.iterrows()
    }

    # --- Build DataFrames with correct column names ---
    junction_pressures = pd.DataFrame(
        ow.np_results["res_junction.p_bar"],
        columns=[f"junction_{i}" for i in net.junction.index],
    )
    sink_flows = pd.DataFrame(
        ow.np_results["res_sink.mdot_kg_per_s"],
        columns=[f"sink_{i}" for i in net.sink.index],
    )
    pipe_velocities = pd.DataFrame(
        ow.np_results["res_pipe.v_mean_m_per_s"],
        columns=[f"v_mean_pipe_{i}" for i in net.pipe.index],
    )
    pipe_flows = pd.DataFrame(
        abs(ow.np_results["res_pipe.mdot_from_kg_per_s"]),
        columns=[f"flow_pipe_{i}" for i in net.pipe.index],
    )

    # --- Compute mean & std pressure per pipe ---
    mean_pressures = {}
    std_pressures = {}

    for pid, pinfo in pipe_params.items():
        j_from = pinfo["from_junction"]
        j_to = pinfo["to_junction"]

        col_from = f"junction_{j_from}"
        col_to = f"junction_{j_to}"

        if col_from in junction_pressures.columns and col_to in junction_pressures.columns:
            p_avg = (junction_pressures[col_from] + junction_pressures[col_to]) / 2
            mean_pressures[pid] = float(p_avg.mean())
            std_pressures[pid] = float(p_avg.std())
        else:
            mean_pressures[pid] = 0.0
            std_pressures[pid] = 0.0

    # Attach them into pipe_params
    for pid in pipe_params:
        pipe_params[pid]["mean_pressure_bar"] = mean_pressures[pid]
        pipe_params[pid]["std_pressure_bar"] = std_pressures[pid]

    geo = {int(i): (row.x, row.y) for i, row in net.junction_geodata.iterrows()}
    ext = {
        "junction": int(net.ext_grid.junction.iloc[0]),
        "coord": geo[int(net.ext_grid.junction.iloc[0])],
    }

    # --- Merge split leak pipes back into single entries ---
    if city_data:
        for road in city_data.get("roads", []):
            if "leak" not in road:
                continue

            rid = road["id"]
            up_name = f"{rid}_upstream"
            down_name = f"{rid}_downstream"

            # Find internal pipe indices by name
            ids = [i for i, n in enumerate(net.pipe.name) if n in (up_name, down_name)]
            if not ids:
                continue

            # Average the two segment results for display
            v_mean_cols = [f"v_mean_pipe_{i}" for i in ids if f"v_mean_pipe_{i}" in pipe_velocities.columns]
            flow_cols = [f"flow_pipe_{i}" for i in ids if f"flow_pipe_{i}" in pipe_flows.columns]

            if v_mean_cols:
                pipe_velocities[f"v_mean_pipe_{rid}"] = pipe_velocities[v_mean_cols].mean(axis=1)
            if flow_cols:
                pipe_flows[f"flow_pipe_{rid}"] = pipe_flows[flow_cols].mean(axis=1)

            # --- collect structural data from sub-pipes ---
            sub_params = [pipe_params.get(i, {}) for i in ids if i in pipe_params]
            if not sub_params:
                continue

            # From/to junctions: take upstream start and downstream end
            from_junction = sub_params[0].get("from_junction")
            to_junction = sub_params[-1].get("to_junction")

            # Sum lengths and take representative geometry
            total_length = sum(sp.get("length_m", 0.0) for sp in sub_params)
            diameter = np.mean([sp.get("diameter_m", 0.0) for sp in sub_params])
            k_mm = np.mean([sp.get("k_mm", 0.0) for sp in sub_params])

            # Mean & std pressure from both segments
            mean_pressure = np.mean([
                sp.get("mean_pressure_bar", 0.0) for sp in sub_params
            ])
            std_pressure = np.mean([
                sp.get("std_pressure_bar", 0.0) for sp in sub_params
            ])

            new_id = max(pipe_params.keys()) + 1  # next available index
            pipe_velocities[f"v_mean_pipe_{new_id}"] = pipe_velocities[v_mean_cols].mean(axis=1)
            pipe_flows[f"flow_pipe_{new_id}"] = pipe_flows[flow_cols].mean(axis=1)

            # --- Construct merged pipe entry ---
            pipe_params[new_id] = {
                "from_junction": from_junction,
                "to_junction": to_junction,
                "length_m": total_length,
                "diameter_m": diameter,
                "k_mm": k_mm,
                "age": road.get("age"),
                "pipe_type": road.get("pipe_type"),
                "leak": road["leak"],
                "mean_pressure_bar": mean_pressure,
                "std_pressure_bar": std_pressure,
                "display_id": rid,  # <-- keep original road ID for frontend
            }

            # Cleanup: drop the internal segment IDs
            for i in ids:
                pipe_params.pop(i, None)

    return {
        "timestamps": timestamps,
        "junction_pressures": _collect(junction_pressures, "p_bar"),
        "sink_flows": _collect(sink_flows, "mdot"),
        "pipe_velocities": _collect(pipe_velocities, ""),
        "pipe_flows": _collect(pipe_flows, ""),
        "pipe_parameters": pipe_params,
        "external_grid": ext,
        "pumps": [],
    }
