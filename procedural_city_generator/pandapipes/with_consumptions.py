import json
import numpy as np
import pandapipes as pp
import pandapipes.plotting as plot
import matplotlib.pyplot as plt
import pandas as pd
import pandapower.control as control
from pandapower.timeseries import DFData, OutputWriter
from pandapipes.timeseries import run_timeseries
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

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

def base_consumption(t):
    return np.ones_like(t)

# consumption_functions = {
#     "single_family": single_family_consumption,
#     "apartment": apartment_consumption,
#     "restaurant": restaurant_consumption,
#     "office": office_consumption,
#     "hospital": hospital_consumption,
#     "library": library_consumption,
#     "school": school_consumption,
#     "factory": factory_consumption,
#     "warehouse": warehouse_consumption,
#     "processing_plant": processing_plant_consumption
# }

consumption_functions = {
    "single_family": base_consumption,
    "apartment": base_consumption,
    "restaurant": base_consumption,
    "office": base_consumption,
    "hospital": base_consumption,
    "library": base_consumption,
    "school": base_consumption,
    "factory": base_consumption,
    "warehouse": base_consumption,
    "processing_plant": base_consumption
}

# ------------------ Define Pipe Roughness Function ------------------
def get_pipe_roughness():
    age = np.random.uniform(0, 100)
    roughness = 0.0015 + (0.3 - 0.0015) * (age / 100.0)
    return roughness

# ------------------ Parameters ------------------
height_scale = 1
new_ext_grid_pressure = 5

# Baseline daily consumption for a single-family home, now set to 150 L/day.
baseline_daily = 150  # L/day

# Multipliers for other building types relative to a single-family home.
multipliers = {
    "single_family": 1.0,
    "apartment": 7.0,
    "restaurant": 2.0,
    "office": 2.5,
    "hospital": 4.0,
    "library": 1.0,
    "school": 2.0,
    "factory": 3.0,
    "warehouse": 1.5,
    "processing_plant": 2.5
}

# ------------------ Step 1: Load JSON Data ------------------
with open("city_data.json", "r") as f:
    city_data = json.load(f)

# ------------------ Step 2: Compute Unique Junctions and Their Heights ------------------
junction_coords = {}
junction_list = []
junction_heights = {}

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

# ------------------ Step 4: Create Pipes with Random Roughness ------------------
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
    k_mm = get_pipe_roughness()
    
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

# ------------------ Step 5: Add Buildings as Sinks with Consumption Based on Building Type ------------------
# Now, the nominal daily consumption (L/day) is defined based solely on the building type.
sink_info = []
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
    nominal_kg_s = nominal_daily_L / 86400.0
    
    pp.create_sink(
        net,
        junction=j_building,
        mdot_kg_per_s=nominal_kg_s,
        name=f"Sink {idx+1}"
    )
    sink_info.append((len(sink_info), building_type, nominal_kg_s))

print("Network built. Number of sinks:", len(net.sink))


# --- Step 6: Run the Pipeflow Simulation ---
# Increase max iterations and tighten tolerances to improve convergence chances.
try:
    pp.pipeflow(
    net,
    friction_model="nikuradse",
    stop_condition="tol",
    iter=100,
    check_connectivity=True
)
except:
    print('exception')

if net.converged:
    print("Single-step pipeflow converged.")
    print("Network built. Number of sinks:", len(net.sink))
    #print(net.res_junction)
else:
    print("Single-step pipeflow did NOT converge.")

import networkx as nx

G = nx.Graph()

# 1) Add nodes for each junction
for j_idx in net.junction.index:
    G.add_node(j_idx)

# 2) Add edges for each pipe
for p_idx in net.pipe.index:
    j_from = net.pipe.at[p_idx, "from_junction"]
    j_to   = net.pipe.at[p_idx, "to_junction"]
    # Add an undirected edge
    G.add_edge(j_from, j_to)

# 3) Identify the external grid junction
ext_grid_j = net.ext_grid.at[net.ext_grid.index[0], "junction"]

# 4) Get all reachable junctions from ext_grid_j
reachable = nx.algorithms.traversal.breadth_first_search.bfs_tree(G, source=ext_grid_j).nodes()

print("External grid junction:", ext_grid_j)
print(f"Number of reachable junctions: {len(reachable)} out of {len(net.junction.index)} total.")



# --- Print Pipe Lengths and Junction Pressures in Descending Order ---
print("\nPipes sorted by descending length with junction pressures:")

# # Gather pipe info into a list
# pipe_info = []
# for pipe_idx in net.pipe.index:
#     pipe = net.pipe.loc[pipe_idx]

#     length_m = pipe["length_km"] * 1000
#     from_junction = pipe["from_junction"]
#     to_junction = pipe["to_junction"]

#     p_from = net.res_junction.at[from_junction, "p_bar"]
#     p_to = net.res_junction.at[to_junction, "p_bar"]

#     pipe_info.append((pipe_idx, length_m, from_junction, p_from, to_junction, p_to))

# # Sort the pipes by descending length
# pipe_info.sort(key=lambda x: x[1], reverse=True)

# # Print sorted information
# for pipe_idx, length_m, from_junction, p_from, to_junction, p_to in pipe_info:
#     print(
#         f"Pipe {pipe_idx}: Length = {length_m:.2f} m | "
#         f"Pressure from_junction ({from_junction}) = {p_from:.4f} bar | "
#         f"to_junction ({to_junction}) = {p_to:.4f} bar"
#     )




# Create default collections with a custom junction size.
simple_collections = plot.create_simple_collections(
    net,
    junction_size=0.2,  # Adjusted relative size of junctions.
    ext_grid_size=0.0,
    pipe_width=1.0,
    as_dict=False
)

ext_grid_pc, ext_grid_lc = plot.create_ext_grid_collection(
    net,
    size=40,  # Increase this value to get larger ext_grid patches.
    orientation=0,  # 0 means oriented upwards.
    # Optionally, you can also pass a color:
    color="blue"
)

# (Optional) Create a collection for highlighting main pipes.
main_pipes = [i for i, road in enumerate(city_data["roads"]) if road.get("pipe_type", "side") == "main"]
pipe_collection = plot.create_pipe_collection(
    net,
    pipes=main_pipes,
    linewidths=2.5,
    color="green",
    zorder=5
)

# Identify junctions with sinks.
sink_junctions = net.sink["junction"].unique()

# Create a collection for sink junctions using an orange rectangle.
sink_collection = plot.create_junction_collection(
    net,
    junctions=sink_junctions,
    patch_type="circle",  # Options: "circle", "rect", "triangle"
    size=18,
    color="orange",
    zorder=10
)



# Append custom collections to the default collections.
simple_collections.append(ext_grid_pc)
simple_collections.append(ext_grid_lc)
simple_collections.append(pipe_collection)
#simple_collections.append(sink_collection)

# --- Create a custom building zone collection ---
# Define district colors.
district_colors = {
    "residential": "lightblue",
    "industrial": "lightcoral",
    "undefined": "gray"
}

building_patches = []
for building in city_data.get("buildings", []):
    if "corners" in building:
        corners = building["corners"]
        district = building.get("district", "undefined")
        # Ensure only the two valid zones are used; otherwise fallback.
        if district not in ["residential", "industrial"]:
            district = "undefined"
        face_color = district_colors.get(district, "gray")
        patch = MplPolygon(corners, closed=True, facecolor=face_color, edgecolor="black", alpha=0.7)
        building_patches.append(patch)

# Create a PatchCollection for the building zones.
building_collection = PatchCollection(building_patches, match_original=True, zorder=20)

# Define legend patches for only the two zones.
res_patch = mpatches.Patch(color='lightblue', label='Residential')
ind_patch = mpatches.Patch(color='lightcoral', label='Industrial')



# Draw all pandapipes collections.
ax = plot.draw_collections(simple_collections)

# Get junctions that have sinks (buildings)
sink_junctions = set(net.sink["junction"].unique())

# Add Junction Heights (to the left) and Pressures (to the right) on the plot
for j_idx in net.junction.index:
    if j_idx in sink_junctions:
        continue  # Skip junctions with sinks (buildings)

    x = net.junction_geodata.at[j_idx, "x"]
    y = net.junction_geodata.at[j_idx, "y"]
    pressure = net.res_junction.at[j_idx, "p_bar"]
    height = net.junction.at[j_idx, "height_m"]

    # Pressure to the right
# Check if pressure is NaN
    if np.isnan(pressure):
        # Draw red text indicating NaN pressure with larger fontsize
        ax.text(
            x, y, "NaN", fontsize=5, ha='left', va='center',
            color='red',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2),
            zorder=30
        )
        # Also draw a larger red marker at the junction location
        #ax.scatter(x, y, color='red', s=1, marker='o', zorder=35)
    else:
        # Draw pressure normally (black text, smaller marker)
        ax.text(
            x + 20, y, f"{pressure:.2f} bar", fontsize=5, ha='left', va='center',
            color='black',
            bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.2),
            zorder=30
        )
        #ax.scatter(x, y, color='black', s=1, marker='o', zorder=30)

    # Height to the left
    ax.text(
        x + 20, y + 20, f"{height:.1f} m",
        fontsize=5,
        ha='left',
        va='center',
        color='black',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2),
        zorder=30
    )


# Add the building zone collection on top.
ax.add_collection(building_collection)

# Add the legend to the axes.
ax.legend(handles=[res_patch, ind_patch], loc='upper right')


from noise import pnoise2

# --- Configuration (use same as ZoneRenderer) ---
HEIGHT_MAP_SCALE = 1000.0
HEIGHT_MAP_OCTAVES = 2
HEIGHT_MAP_AMPLITUDE = 100
HEIGHT_MAP_RES = 300

# Compute plot bounds from junction coordinates
all_x = net.junction_geodata['x'].values
all_y = net.junction_geodata['y'].values
padding = 200
min_x, max_x = all_x.min() - padding, all_x.max() + padding
min_y, max_y = all_y.min() - padding, all_y.max() + padding

# Generate the height map
terrain = np.zeros((HEIGHT_MAP_RES, HEIGHT_MAP_RES))
for i in range(HEIGHT_MAP_RES):
    for j in range(HEIGHT_MAP_RES):
        x = min_x + (i / (HEIGHT_MAP_RES - 1)) * (max_x - min_x)
        y = min_y + (j / (HEIGHT_MAP_RES - 1)) * (max_y - min_y)
        val = pnoise2(x / HEIGHT_MAP_SCALE, y / HEIGHT_MAP_SCALE, octaves=HEIGHT_MAP_OCTAVES, base=city_data["seed"])
        terrain[j, i] = ((val + 1) / 2) * HEIGHT_MAP_AMPLITUDE

# Plot the terrain map below the existing pandapipes plot
ax.imshow(
    terrain,
    origin='lower',
    extent=[min_x, max_x, min_y, max_y],
    cmap='terrain',
    alpha=0.5,
    interpolation='bilinear',
    zorder=-1
)

# Ensure the plot stays square and neat
ax.set_aspect('equal', adjustable='box')


plt.show()



# --- Step 7: Render the Network ---
#pp.plotting.simple_plot(net, plot_sinks=True)
