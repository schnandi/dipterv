import pandas as pd


def _collect(df, prefix):
    if prefix:
        return {f"{prefix}_{col}": df[col].tolist() for col in df.columns}
    else:
        return {col: df[col].tolist() for col in df.columns}


def format_results(net, ow):
    timestamps = pd.date_range(
        "2025-01-01", periods=len(ow.time_steps), freq="15min"
    ).strftime("%Y-%m-%d %H:%M:%S").tolist()

    # --- identical param info as old ---
    pipe_params = {
        int(i): {
            "name": row["name"],
            "from_junction": int(row["from_junction"]),
            "to_junction": int(row["to_junction"]),
            "length_m": float(row["length_km"] * 1e3),
            "diameter_m": float(row["diameter_m"]),
            "k_mm": float(row["k_mm"]),
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

    geo = {int(i): (row.x, row.y) for i, row in net.junction_geodata.iterrows()}
    ext = {
        "junction": int(net.ext_grid.junction.iloc[0]),
        "coord": geo[int(net.ext_grid.junction.iloc[0])],
    }

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
