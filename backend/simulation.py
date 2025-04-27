import numpy as np
import pandas as pd
import pandapipes as pp
import pandapower.control as control
from pandapower.timeseries import DFData, OutputWriter
from pandapipes.timeseries import run_timeseries

# ------------------ Define Consumption Functions ------------------
def single_family_consumption(t):
    baseline = 0.2
    morning_peak = 1.0 * np.exp(-((t - 8) / 1.5) ** 2)
    evening_peak = 1.2 * np.exp(-((t - 19) / 1.5) ** 2)
    return baseline + morning_peak + evening_peak

def apartment_consumption(t):
    baseline = 0.3
    morning_peak = 0.8 * np.exp(-((t - 8) / 1.2) ** 2)
    evening_peak = 0.9 * np.exp(-((t - 19) / 1.2) ** 2)
    return baseline + morning_peak + evening_peak

def restaurant_consumption(t):
    baseline = 0.1
    lunch_peak = 1.5 * np.exp(-((t - 12) / 0.8) ** 2)
    dinner_peak = 2.0 * np.exp(-((t - 20) / 1.0) ** 2)
    return baseline + lunch_peak + dinner_peak

def office_consumption(t):
    ramp_up = 1 / (1 + np.exp(-2 * (t - 8)))
    ramp_down = 1 / (1 + np.exp(2 * (t - 17)))
    return 0.2 + 0.8 * ramp_up * ramp_down

def hospital_consumption(t):
    return 1.0 + 0.1 * np.sin(2 * np.pi * (t - 6) / 24)

def library_consumption(t):
    baseline = 0.2
    peak = 1.0 * np.exp(-((t - 14) / 1.0) ** 2)
    return baseline + peak

def school_consumption(t):
    baseline = 0.2
    morning_peak = 1.2 * np.exp(-((t - 8) / 1.0) ** 2)
    afternoon_peak = 1.0 * np.exp(-((t - 15) / 1.0) ** 2)
    return baseline + morning_peak + afternoon_peak

def factory_consumption(t):
    base = 1.8
    midday_hump = 0.2 * np.exp(-((t - 12) / 2)**2)
    return base + midday_hump

def warehouse_consumption(t):
    key_times = np.array([0, 8, 18, 24])
    key_values = np.array([0.5, 1.5, 1.5, 0.5])
    return np.interp(t, key_times, key_values)

def processing_plant_consumption(t):
    base = 1.2
    midday_dip = -0.2 * np.exp(-((t - 13) / 1)**2)
    return base + midday_dip

consumption_functions = {
    "single_family": single_family_consumption,
    "apartment": apartment_consumption,
    "restaurant": restaurant_consumption,
    "office": office_consumption,
    "hospital": hospital_consumption,
    "library": library_consumption,
    "school": school_consumption,
    "factory": factory_consumption,
    "warehouse": warehouse_consumption,
    "processing_plant": processing_plant_consumption
}

def simulate_water_network(city_data):
    height_scale = 1.0
    new_ext_grid_pressure = 5.0
    baseline_daily = 150.0
    multipliers = {
        "single_family": 1.0,
        "apartment": 7.0,
        "restaurant": 2.0,
        "office": 4.0,
        "hospital": 10.0,
        "library": 2.0,
        "school": 10.0,
        "factory": 20.0,
        "warehouse": 1.5,
        "processing_plant": 2.5
    }

    junction_coords = {}
    junction_list = []
    junction_heights = {}
    for road in city_data["roads"]:
        for point_type in ["start", "end"]:
            pt = tuple(road[point_type])
            height = road.get(f"{point_type}_height", 0)
            if pt not in junction_coords:
                junction_coords[pt] = len(junction_list)
                junction_list.append(pt)
                junction_heights[pt] = [height]
            else:
                junction_heights[pt].append(height)
    avg_junction_heights = {pt: np.mean(heights) * height_scale for pt, heights in junction_heights.items()}

    net = pp.create_empty_network(fluid="water")
    for coord in junction_list:
        scaled_height = avg_junction_heights.get(coord, 0)
        pp.create_junction(
            net,
            index=junction_coords[coord],
            pn_bar=1.0,
            tfluid_k=293.15,
            height_m=scaled_height,
            name=f"Junction {coord}",
            geodata={"x": coord[0], "y": coord[1]}
        )

    main_pipe_coords = set()
    for road in city_data["roads"]:
        if road.get("pipe_type", "side") == "main":
            main_pipe_coords.add(tuple(road["start"]))
            main_pipe_coords.add(tuple(road["end"]))
    if main_pipe_coords:
        supply_junction_coord = max(main_pipe_coords, key=lambda pt: avg_junction_heights.get(pt, 0))
    else:
        supply_junction_coord = max(avg_junction_heights, key=avg_junction_heights.get)
    supply_junction_index = junction_coords[supply_junction_coord]
    pp.create_ext_grid(
        net,
        junction=supply_junction_index,
        p_bar=new_ext_grid_pressure,
        t_k=293.15,
        name="External Grid",
        index=city_data.get("external_grid_id", 0)
    )

    pipe_type_diameters = {
        "main": 0.6,
        "side": 0.4,
        "building connection": 0.2
    }
    for road in city_data["roads"]:
        pt_start = tuple(road["start"])
        pt_end = tuple(road["end"])
        j_from = junction_coords[pt_start]
        j_to = junction_coords[pt_end]
        pipe_type = road.get("pipe_type", "side")
        diameter = pipe_type_diameters.get(pipe_type, pipe_type_diameters["side"])
        length = np.linalg.norm(np.array(pt_end) - np.array(pt_start))
        length_km = length / 5000.0
        age = road.get("age", 50)
        k_mm = 0.0015 + (0.3 - 0.0015) * (age / 100.0)
        pp.create_pipe_from_parameters(
            net,
            from_junction=j_from,
            to_junction=j_to,
            length_km=length_km,
            diameter_m=diameter,
            k_mm=k_mm,
            name=f"Pipe {pt_start} -> {pt_end}",
            index=road["id"]
        )

    sink_info = []
    ext_grid_junction_index = net.ext_grid["junction"].iloc[0]
    for building in city_data.get("buildings", []):
        center = tuple(building["center"])
        building_type = building.get("building_type", "single_family")
        if center in junction_coords and junction_coords[center] == ext_grid_junction_index:
            continue
        if center in junction_coords:
            j_building = junction_coords[center]
        else:
            j_building = pp.create_junction(
                net,
                pn_bar=1.0,
                tfluid_k=293.15,
                height_m=building.get("terrain_height", 0) * height_scale,
                name=f"Junction {center}",
                geodata={"x": center[0], "y": center[1]}
            )
            junction_coords[center] = j_building
        noise = np.random.uniform(0.9, 1.1)
        nominal_daily_L = baseline_daily * multipliers.get(building_type, 1.0) * noise
        nominal_kg_s = nominal_daily_L / 86400.0
        pp.create_sink(
            net,
            junction=j_building,
            mdot_kg_per_s=nominal_kg_s,
            name=f"Sink {building['id']}",
            index=building["id"]
        )
        sink_info.append((building["id"], building_type, nominal_kg_s))

    t_daily = np.linspace(0, 24, 96)
    dynamic_profiles = []
    for sink_id, btype, nominal in sink_info:
        func = consumption_functions.get(btype, lambda t: np.ones_like(t))
        raw_profile = func(t_daily)
        scale_factor = nominal / np.mean(raw_profile)
        scaled_profile = raw_profile * scale_factor
        noise_factor = np.random.uniform(0.9, 1.1, size=scaled_profile.shape)
        final_profile = scaled_profile * noise_factor
        dynamic_profiles.append(final_profile)

    dynamic_profiles_array = np.array(dynamic_profiles).T
    sink_indices = net.sink.index.values.astype(str)
    profiles_df_dyn = pd.DataFrame(data=dynamic_profiles_array, index=range(96), columns=sink_indices)
    ds_dyn = DFData(profiles_df_dyn)
    control.ConstControl(
        net,
        element='sink',
        variable='mdot_kg_per_s',
        element_index=net.sink.index.values,
        data_source=ds_dyn,
        profile_name=sink_indices
    )

    ow_dyn = OutputWriter(net, time_steps=range(96), output_path=None, log_variables=[
        ('res_junction', 'p_bar'),
        ('res_pipe', 'v_mean_m_per_s'),
        ('res_sink', 'mdot_kg_per_s'),
        ('res_ext_grid', 'mdot_kg_per_s')
    ])

    run_timeseries(net, range(96), iter=200)

    def collect_results(df, prefix):
        return {
            f"{prefix}_{col}": df[col].tolist()
            for col in df.columns
        }

    timestamps = pd.date_range("2025-01-01", periods=96, freq="15min").strftime("%Y-%m-%d %H:%M:%S").tolist()

    return {
        "timestamps": timestamps,
        "junction_pressures": collect_results(pd.DataFrame(
            ow_dyn.np_results["res_junction.p_bar"], columns=[f"junction_{i}" for i in net.junction.index]), "p_bar"),
        "sink_flows": collect_results(pd.DataFrame(
            ow_dyn.np_results["res_sink.mdot_kg_per_s"], columns=[f"sink_{i}" for i in net.sink.index]), "mdot"),
        "pipe_velocities": collect_results(pd.DataFrame(
            ow_dyn.np_results["res_pipe.v_mean_m_per_s"], columns=[f"pipe_{i}" for i in net.pipe.index]), "v_mean"),
        "external_grid_flows": collect_results(pd.DataFrame(
            ow_dyn.np_results["res_ext_grid.mdot_kg_per_s"], columns=[f"ext_grid_{i}" for i in net.ext_grid.index]), "mdot")
    }
