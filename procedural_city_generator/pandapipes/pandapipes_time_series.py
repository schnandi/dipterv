import json
import numpy as np
import pandapipes as pp
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# --- Parameters for Convergence ---
height_scale = 1    # Scale down computed junction heights (e.g. 1% of original)
new_ext_grid_pressure = 5  # External grid pressure in bar

# --- Step 1: Load JSON Data ---
with open("city_data.json", "r") as f:
    city_data = json.load(f)

# --- Step 2: Compute Unique Junctions and Their Heights ---
junction_coords = {}      # Maps coordinate tuple to junction index.
junction_list = []        # List of coordinate tuples.
junction_heights = {}     # Maps coordinate tuple to list of height values from roads.

for road in city_data["roads"]:
    # Process start point.
    pt_start = tuple(road["start"])
    height_start = road.get("start_height", 0)
    if pt_start not in junction_coords:
        junction_coords[pt_start] = len(junction_list)
        junction_list.append(pt_start)
        junction_heights[pt_start] = [height_start]
    else:
        junction_heights[pt_start].append(height_start)
    # Process end point.
    pt_end = tuple(road["end"])
    height_end = road.get("end_height", 0)
    if pt_end not in junction_coords:
        junction_coords[pt_end] = len(junction_list)
        junction_list.append(pt_end)
        junction_heights[pt_end] = [height_end]
    else:
        junction_heights[pt_end].append(height_end)

# Compute average height for each junction and scale it down.
avg_junction_heights = {}
for pt, heights in junction_heights.items():
    avg_junction_heights[pt] = np.mean(heights) * height_scale

# --- Step 3: Create a pandapipes Network ---
net = pp.create_empty_network(fluid="water")

# Create junctions with scaled heights.
for coord in junction_list:
    scaled_height = avg_junction_heights.get(coord, 0)
    pp.create_junction(
        net,
        pn_bar=1.0,           # Initial pressure (bar); adjust as needed.
        tfluid_k=293.15,      # Fluid temperature (K).
        height_m=scaled_height,  # Scaled height above sea level.
        name=f"Junction {coord}",
        geodata={"x": coord[0], "y": coord[1]}
    )

# --- Supply the Network: Add an External Grid ---
# Choose the junction with the highest scaled height as the supply node.
supply_junction_coord = max(avg_junction_heights, key=avg_junction_heights.get)
supply_junction_index = junction_coords[supply_junction_coord]
pp.create_ext_grid(
    net,
    junction=supply_junction_index,
    p_bar=new_ext_grid_pressure,  # Increased supply pressure (bar).
    t_k=293.15,
    name="External Grid"
)

# --- Step 4: Define and Create Pipes ---
pipe_type_diameters = {
    "main": 0.6,               # Main road pipe: larger.
    "side": 0.4,               # Side road pipe: medium.
    "building connection": 0.2 # Building connection pipe: smaller.
}
pipe_type_k_mm = {
    "main": 0.2,
    "side": 0.2,
    "building connection": 0.2
}
default_pipe_type = "side"

for road in city_data["roads"]:
    pt_start = tuple(road["start"])
    pt_end = tuple(road["end"])
    j_from = junction_coords[pt_start]
    j_to = junction_coords[pt_end]
    
    pipe_type = road.get("pipe_type", default_pipe_type)
    diameter = pipe_type_diameters.get(pipe_type, pipe_type_diameters["side"])
    k_mm = pipe_type_k_mm.get(pipe_type, 0.2)
    
    length = np.linalg.norm(np.array(pt_end) - np.array(pt_start))
    length_km = length / 100.0
    
    pp.create_pipe_from_parameters(
        net,
        from_junction=j_from,
        to_junction=j_to,
        length_km=length_km,
        diameter_m=diameter,
        k_mm=k_mm,
        name=f"Pipe {pt_start} -> {pt_end}"
    )

# --- Step 5: Add Buildings as Sinks ---
# Further reduce consumption by lowering the scaling factor.
sink_scaling_factor = 100  # [kg/s per square meter].
# Get the junction index where the external grid is connected:
ext_grid_junction_index = net.ext_grid["junction"].iloc[0]
for idx, building in enumerate(city_data.get("buildings", [])):
    center = tuple(building["center"])
    width = building.get("width", 50)
    height = building.get("height", 50)
    consumption = width * height * sink_scaling_factor  # kg/s consumption.

    # If the building's junction is the same as the ext grid junction, skip creating a sink.
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
    
    pp.create_sink(
        net,
        junction=j_building,
        mdot_kg_per_s=consumption,
        name=f"Sink {idx+1}"
    )

print("Network built. Number of sinks:", len(net.sink))

#pp.pipeflow(net, friction_model="nikuradse")

# if net.converged:
#     print("Single-step pipeflow converged.")
#     #print(net.res_junction)
# else:
#     print("Single-step pipeflow did NOT converge.")

import pandas as pd
import numpy as np
import pandapower.control as control
from pandapower.timeseries import DFData, OutputWriter
from pandapipes.timeseries import run_timeseries

# --- Daily Simulation Setup: 96 intervals (15 min each) ---
time_steps = range(96)
timestamps = range(96)

# Sink indices
sink_indices = net.sink.index.values
n_sinks = len(sink_indices)

# --- Generate realistic daily consumption patterns ---
daily_pattern = np.array(
    [0.2]*24 +                      # 00:00 - 06:00 (6 hours): low usage
    [0.4, 0.6, 0.8, 1.0]*3 +        # 06:00 - 09:00 (morning peak, 3 hours)
    [0.7]*20 +                      # 09:00 - 14:00 (daytime moderate usage)
    [0.8]*16 +                      # 14:00 - 18:00 (afternoon increase)
    [0.9, 1.0, 1.0, 0.8]*3 +        # 18:00 - 21:00 (evening peak, 3 hours)
    [0.6]*12                        # 21:00 - 24:00 (late evening drop)
)
daily_pattern = daily_pattern[:96]

# Random variation per sink around the daily pattern (fixed)
random_profiles = np.array([
    daily_pattern * np.random.uniform(0.8, 1.2)
    for _ in range(n_sinks)
]).T * 0.0001  # Scale to realistic [kg/s] per building sink

profiles_df = pd.DataFrame(
    data=random_profiles,
    index=timestamps,
    columns=sink_indices.astype(str)
)

# --- Control setup ---
ds = DFData(profiles_df)

control.ConstControl(
    net,
    element='sink',
    variable='mdot_kg_per_s',
    element_index=sink_indices,
    data_source=ds,
    profile_name=sink_indices.astype(str)
)

# --- Variables to log (sensor data) ---
log_variables = [
    ('res_junction', 'p_bar'),            # Junction pressures
    ('res_pipe', 'v_mean_m_per_s'),       # Pipe velocities
    ('res_sink', 'mdot_kg_per_s'),        # Actual sink flows
    ('res_ext_grid', 'mdot_kg_per_s'),    # External grid mass flows
]

ow = OutputWriter(
    net,
    time_steps=time_steps,
    output_path=None,
    log_variables=log_variables
)

# --- Run Simulation ---
run_timeseries(net, time_steps)

# --- Export data to CSV ---
timestamps = pd.date_range("2025-01-01", periods=96, freq="15min")

# Junction pressures
df_junction_pressures = pd.DataFrame(
    ow.np_results["res_junction.p_bar"],
    index=timestamps,
    columns=[f"junction_{j_idx}_pressure_bar" for j_idx in net.junction.index]
)
df_junction_pressures.to_csv("junction_pressures.csv", index_label="timestamp")

# Sink mass flows
df_sink_flows = pd.DataFrame(
    ow.np_results["res_sink.mdot_kg_per_s"],
    index=timestamps,
    columns=[f"sink_{s_idx}_massflow_kgs" for s_idx in net.sink.index]
)
df_sink_flows.to_csv("sink_massflows.csv", index_label="timestamp")

# External grid mass flows
df_ext_grid = pd.DataFrame(
    ow.np_results["res_ext_grid.mdot_kg_per_s"],
    index=timestamps,
    columns=[f"ext_grid_{e_idx}_massflow_kgs" for e_idx in net.ext_grid.index]
)
df_ext_grid.to_csv("external_grid_massflows.csv", index_label="timestamp")

# Pipe velocities (flow sensors)
df_pipe_velocities = pd.DataFrame(
    ow.np_results["res_pipe.v_mean_m_per_s"],
    index=timestamps,
    columns=[f"pipe_{p_idx}_velocity_mps" for p_idx in net.pipe.index]
)
df_pipe_velocities.to_csv("pipe_velocities.csv", index_label="timestamp")

# --- Combine all sensor data into one CSV (optional) ---
df_all_sensors = pd.concat([
    df_junction_pressures,
    df_sink_flows,
    df_ext_grid,
    df_pipe_velocities
], axis=1)
df_all_sensors.to_csv("all_sensor_data.csv", index_label="timestamp")

# Quick check plot (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(timestamps, df_ext_grid.iloc[:, 0], 'r-o', label='External Grid Flow [kg/s]')
plt.xlabel("Time")
plt.ylabel("Mass Flow [kg/s]")
plt.title("Daily External Grid Mass Flow Profile")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Time series CSV files created successfully.")


