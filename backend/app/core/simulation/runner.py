import numpy as np
import pandas as pd
import pandapipes as pp
import pandapower.control as control
from pandapower.timeseries import DFData, OutputWriter
from pandapipes.timeseries import run_timeseries

from .consumption_profiles import get_profile
from .model_builder import build_network
from .results_parser import format_results


def run_simulation(city_data, n_steps=96):
    net, sink_info = build_network(city_data)
    # --- Generate dynamic consumption profiles ---
    t_daily = np.linspace(0, 24, n_steps)

    # 1. Identify building sinks only (exclude leak sinks)
    building_sink_indices = []
    profiles = {}

    for info in sink_info:
        sink_idx = info["sink_index"]
        btype = info["building_type"]
        nominal = info["nominal_kg_s"]

        # Generate base consumption profile
        prof = get_profile(btype, t_daily)

        # Scale to nominal mass flow
        scale = nominal / np.mean(prof)
        profiles[str(sink_idx)] = prof * scale

        building_sink_indices.append(sink_idx)

    df_profiles = pd.DataFrame(profiles)
    ds = DFData(df_profiles)

    print("Actual sink indices used:", building_sink_indices)
    print("df_profiles.columns:", df_profiles.columns.tolist())

    # --- Dynamic sink control ---
    control.ConstControl(
        net,
        element="sink",
        variable="mdot_kg_per_s",
        element_index=building_sink_indices,      # REAL sink row indices
        data_source=ds,
        profile_name=df_profiles.columns,
    )

    # --- Output Writer (same as before) ---
    ow = OutputWriter(
        net,
        time_steps=range(n_steps),
        output_path=None,
        log_variables=[
            ("res_junction", "p_bar"),
            ("res_pipe", "v_mean_m_per_s"),
            ("res_pipe", "mdot_from_kg_per_s"),
            ("res_sink", "mdot_kg_per_s"),
            ("res_ext_grid", "mdot_kg_per_s"),
        ],
    )

    # --- Run time series with pandapipes' solver ---
    run_timeseries(net, range(n_steps), iter=200)

    return format_results(net, ow, city_data)
