import json
import numpy as np
import pandapipes as pp
import pandas as pd
import matplotlib.pyplot as plt

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

# Civic buildings:
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
    # Base consumption is high during operating hours.
    # A modest Gaussian hump is added around noon (t = 12) to simulate increased activity.
    base = 1.8
    midday_hump = 0.2 * np.exp(-((t - 12) / 2)**2)
    return base + midday_hump

# Warehouse: Use piecewise linear interpolation to simulate a clear shift in usage.
def warehouse_consumption(t):
    # Define key times and corresponding consumption levels:
    # Very low before 8:00, a plateau from 8:00 to 18:00, and then low again.
    key_times = np.array([0, 8, 18, 24])
    # These values create a trapezoidal profile.
    key_values = np.array([0.5, 1.5, 1.5, 0.5])
    return np.interp(t, key_times, key_values)

# Processing Plant: Mostly constant consumption with a short dip mid-day.
def processing_plant_consumption(t):
    # Base consumption remains nearly constant.
    # A small dip is introduced around 13:00 (1 PM) to simulate a brief slowdown (e.g., a scheduled break).
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

# ------------------ Define Pipe Roughness Function ------------------
def get_pipe_roughness():
    age = np.random.uniform(0, 100)
    return 0.0015 + (0.3 - 0.0015) * (age / 100.0)

# ------------------ Parameters ------------------
height_scale = 1.0
new_ext_grid_pressure = 5.0  # bar

# Baseline daily consumption (L/day) for a single-family home.
baseline_daily = 150.0  # L/day

# Multipliers for other building types relative to a single-family home.
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

# ------------------ Step 1: Load JSON Data ------------------
with open("city_data.json", "r") as f:
    city_data = json.load(f)

# ------------------ Step 2: Compute Unique Junctions and Their Heights ------------------
junction_coords = {}      # Maps coordinate tuple to junction index.
junction_list = []        # List of coordinate tuples.
junction_heights = {}     # Maps coordinate tuple to list of height values.

for road in city_data["roads"]:
    pt_start = tuple(road["start"])
    height_start = road.get("start_height", 0)
    if pt_start not in junction_coords:
        junction_coords[pt_start] = len(junction_list)
        junction_list.append(pt_start)
        junction_heights[pt_start] = [height_start]
    else:
        junction_heights[pt_start].append(height_start)
    pt_end = tuple(road["end"])
    height_end = road.get("end_height", 0)
    if pt_end not in junction_coords:
        junction_coords[pt_end] = len(junction_list)
        junction_list.append(pt_end)
        junction_heights[pt_end] = [height_end]
    else:
        junction_heights[pt_end].append(height_end)

avg_junction_heights = {pt: np.mean(heights) * height_scale for pt, heights in junction_heights.items()}

# ------------------ Step 3: Create a pandapipes Network ------------------
net = pp.create_empty_network(fluid="water")
for coord in junction_list:
    scaled_height = avg_junction_heights.get(coord, 0)
    pp.create_junction(
        net,
        pn_bar=1.0,
        tfluid_k=293.15,
        height_m=scaled_height,
        name=f"Junction {coord}",
        geodata={"x": coord[0], "y": coord[1]}
    )

# ------------------ External Grid: Connect to Highest Junction on a Main Pipe ------------------
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
    name="External Grid"
)

# ------------------ Step 4: Create Pipes ------------------
default_pipe_type = "side"
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
    
    pipe_type = road.get("pipe_type", default_pipe_type)
    diameter = pipe_type_diameters.get(pipe_type, pipe_type_diameters["side"])
    # For convergence, we use a fixed k_mm (the previous working value).
    k_mm = pipe_type_diameters["side"]
    
    length = np.linalg.norm(np.array(pt_end) - np.array(pt_start))
    # Use the conversion factor that worked in your non-time-series model.
    length_km = length / 5000.0
    
    pp.create_pipe_from_parameters(
        net,
        from_junction=j_from,
        to_junction=j_to,
        length_km=length_km,
        diameter_m=diameter,
        k_mm=k_mm,
        name=f"Pipe {pt_start} -> {pt_end}"
    )

# ------------------ Step 5: Add Buildings as Sinks with New Consumption ------------------
# Instead of using building area, we use baseline_daily and multipliers.
sink_info = []  # Will store tuples: (sink_id, building_type, nominal_kg_s)
ext_grid_junction_index = net.ext_grid["junction"].iloc[0]

for idx, building in enumerate(city_data.get("buildings", [])):
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
    nominal_kg_s = nominal_daily_L / 86400.0  # Convert L/day to kg/s (for water, 1 L ≈ 1 kg)
    pp.create_sink(
        net,
        junction=j_building,
        mdot_kg_per_s=nominal_kg_s,
        name=f"Sink {idx+1}"
    )
    sink_info.append((len(sink_info), building_type, nominal_kg_s))

print("Network built. Number of sinks:", len(net.sink))
# (For a single_family, 150 L/day should yield ~0.001736 kg/s)

# ------------------ Step 6: Run a Single-Step Pipeflow Simulation ------------------
# Run a static pipeflow simulation (without time series) to check convergence.
try:
    pp.pipeflow(net, friction_model="nikuradse", stop_condition="tol", iter=100, check_connectivity=True)
except Exception as e:
    print("Static pipeflow simulation did not converge:")
    print(e)
    raise

if net.converged:
    print("Static pipeflow converged.")
    # Print some results:
    print("Junction pressures (bar):")
    print(net.res_junction[["p_bar"]])
    print("Sink flows (kg/s):")
    print(net.res_sink[["mdot_kg_per_s"]])
else:
    print("Static pipeflow did NOT converge.")

# (Optionally, plot a simple diagram of the network.)
# pp.plotting.simple_plot(net, plot_sinks=True)

# End of static simulation code.



import pandas as pd
import numpy as np
import pandapower.control as control
from pandapower.timeseries import DFData, OutputWriter
from pandapipes.timeseries import run_timeseries

# ===================== Dynamic Sink Profile Time Series Simulation =====================
# At this point, your network (net) and the sink_info list (each element is a tuple:
# (sink_id, building_type, nominal_kg_s)) are already available from Step 5.

# Generate a time vector for a 24‐hour day with 96 time steps (15-min intervals)
t_daily = np.linspace(0, 24, 96)
print("t_daily shape:", t_daily.shape)  # Should be (96,)

# Generate a dynamic consumption profile for each sink using its consumption function.
# The raw function returns a pattern (e.g. peaks in the morning and evening for houses),
# but its mean is not yet at the desired level.
dynamic_profiles = []
for sink_id, btype, nominal in sink_info:
    # Look up the consumption function for this building type.
    # (If not found, a flat profile of ones is used.)
    func = consumption_functions.get(btype, lambda t: np.ones_like(t))
    raw_profile = func(t_daily)  # raw_profile is an array of shape (96,)
    # Debug print (optional):
    print(f"Sink {sink_id} ({btype}): raw profile: min={np.min(raw_profile):.4e}, max={np.max(raw_profile):.4e}, mean={np.mean(raw_profile):.4e}")
    
    # Scale the raw profile so its mean matches the nominal sink flow (kg/s).
    scale_factor = nominal / np.mean(raw_profile)
    scaled_profile = raw_profile * scale_factor
    
    # Optionally add multiplicative noise (here ±10%)
    noise_factor = np.random.uniform(0.9, 1.1, size=scaled_profile.shape)
    final_profile = scaled_profile * noise_factor
    dynamic_profiles.append(final_profile)

# Convert the list of profiles into a 2D array with shape (96, n_sinks)
dynamic_profiles_array = np.array(dynamic_profiles).T
print("Dynamic profiles array shape:", dynamic_profiles_array.shape)

# Create a DataFrame for the dynamic profiles.
# Rows represent time steps; columns represent sink indices.
sink_indices = net.sink.index.values.astype(str)
profiles_df_dyn = pd.DataFrame(data=dynamic_profiles_array, index=range(96), columns=sink_indices, dtype=np.float64)
print("Dynamic profiles_df head:")
print(profiles_df_dyn.head())

# Create a DFData object from the dynamic profiles DataFrame.
ds_dyn = DFData(profiles_df_dyn)

# Update sink control to use the dynamic profiles.
control.ConstControl(
    net,
    element='sink',
    variable='mdot_kg_per_s',
    element_index=net.sink.index.values,
    data_source=ds_dyn,
    profile_name=sink_indices.astype(str)
)

# Set up logging for key network variables.
log_variables = [
    ('res_junction', 'p_bar'),      # Junction pressures
    ('res_pipe', 'v_mean_m_per_s'),   # Pipe velocities
    ('res_sink', 'mdot_kg_per_s'),    # Sink flows
    ('res_ext_grid', 'mdot_kg_per_s') # External grid flows
]

# Create a new OutputWriter for the dynamic simulation over 96 time steps.
ow_dyn = OutputWriter(
    net,
    time_steps=range(96),
    output_path=None,
    log_variables=log_variables
)

# Run the time series simulation (using iter=200 for improved convergence).
try:
    run_timeseries(net, range(96), iter=200)
    print("Time series simulation (dynamic profile) converged.")
except Exception as e:
    print("Time series simulation (dynamic profile) did not converge:")
    print(e)
    raise

# ===================== Export Results =====================
timestamps = pd.date_range("2025-01-01", periods=96, freq="15min")

# Export junction pressures.
df_junction_pressures = pd.DataFrame(
    ow_dyn.np_results["res_junction.p_bar"],
    index=timestamps,
    columns=[f"junction_{j_idx}_pressure_bar" for j_idx in net.junction.index]
)
df_junction_pressures.to_csv("junction_pressures_dyn.csv", index_label="timestamp")

# Export sink flows.
df_sink_flows = pd.DataFrame(
    ow_dyn.np_results["res_sink.mdot_kg_per_s"],
    index=timestamps,
    columns=[f"sink_{s_idx}_massflow_kgs" for s_idx in net.sink.index]
)
df_sink_flows.to_csv("sink_massflows_dyn.csv", index_label="timestamp")

# Export external grid flows.
df_ext_grid = pd.DataFrame(
    ow_dyn.np_results["res_ext_grid.mdot_kg_per_s"],
    index=timestamps,
    columns=[f"ext_grid_{e_idx}_massflow_kgs" for e_idx in net.ext_grid.index]
)
df_ext_grid.to_csv("external_grid_massflows_dyn.csv", index_label="timestamp")

# Export pipe velocities.
df_pipe_velocities = pd.DataFrame(
    ow_dyn.np_results["res_pipe.v_mean_m_per_s"],
    index=timestamps,
    columns=[f"pipe_{p_idx}_velocity_mps" for p_idx in net.pipe.index]
)
df_pipe_velocities.to_csv("pipe_velocities_dyn.csv", index_label="timestamp")

# Combine all sensor data into one CSV.
df_all_sensors = pd.concat([
    df_junction_pressures,
    df_sink_flows,
    df_ext_grid,
    df_pipe_velocities
], axis=1)
df_all_sensors.to_csv("all_sensor_data_dyn.csv", index_label="timestamp")

print("Dynamic time series CSV files created successfully.")

# ===================== Create CSV of Sink Consumption Characteristics =====================
# Compute per-sink statistics (average, minimum, maximum) over the 96 time steps.
avg_kg_per_s = np.nanmean(dynamic_profiles_array, axis=0)
min_kg_per_s = np.nanmin(dynamic_profiles_array, axis=0)
max_kg_per_s = np.nanmax(dynamic_profiles_array, axis=0)

# Convert from kg/s to kg/hour (multiply by 3600).
avg_kg_per_hour = avg_kg_per_s * 3600
min_kg_per_hour = min_kg_per_s * 3600
max_kg_per_hour = max_kg_per_s * 3600

# Build a list of dictionaries for each sink including its building type and computed flows.
sink_consumption_data = []
for sink_id, btype, nominal in sink_info:
    idx = int(sink_id)  # Use sink_id as column index.
    sink_consumption_data.append({
         "sink_id": sink_id,
         "building_type": btype,
         "avg_kg_per_hour": avg_kg_per_hour[idx],
         "min_kg_per_hour": min_kg_per_hour[idx],
         "max_kg_per_hour": max_kg_per_hour[idx]
    })

df_sink_consumption = pd.DataFrame(sink_consumption_data)
df_sink_consumption.to_csv("sink_consumption_dynamic.csv", index=False)

print("Sink consumption CSV created successfully.")

