import random
import pandas as pd
from pathlib import Path
import numpy as np
from copy import deepcopy
from math import sqrt
from collections import defaultdict
import networkx as nx   # ✔ graph distance

from app.core.generator.city_generator import CityGenerator
from app.core.simulation.runner import run_simulation
from app.core.ai.leak.utils import compute_pipe_deltas

OUTPUT_PATH = Path("leak_training_data.csv")


# ============================================================
# Helper: generate towns
# ============================================================
def generate_town(seed=None):
    if seed is None:
        seed = random.randint(0, 2**16 - 1)

    gen = CityGenerator(map_size=(2000, 2000), seed=seed)
    return gen.generate(), seed


# ============================================================
# Helper: leak injection (with bigger range)
# ============================================================
def inject_random_leak(city_data, leak_rate_range=(0.05, 0.20)):
    roads = city_data["roads"]

    candidates = [
        r for r in roads
        if r["pipe_type"] != "building connection" and r["id"] < 1e8
    ]
    selected = random.choice(candidates)

    frac = random.uniform(0.05, 0.95)
    rate = random.uniform(*leak_rate_range)

    leaky = deepcopy(city_data)

    leak_x = leak_y = None

    for r in leaky["roads"]:
        if r["id"] == selected["id"]:
            x1, y1 = r["start"]
            x2, y2 = r["end"]
            leak_x = x1 + (x2 - x1) * frac
            leak_y = y1 + (y2 - y1) * frac

            r["leak"] = {
                "coord": [leak_x, leak_y],
                "rate_kg_per_s": rate,
                "fraction": frac,
            }

    return leaky, selected["id"], (leak_x, leak_y)


# ============================================================
# Helper: midpoint
# ============================================================
def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ============================================================
# Build graph for computing topological distances
# ============================================================
def build_graph(roads):
    G = nx.Graph()
    for r in roads:
        if r["id"] >= 1e8:
            continue

        start = tuple(r["start"])
        end = tuple(r["end"])

        length = sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        diameter = r.get("diameter", 0.3)  # fallback

        # hydraulic cost approx: length / diameter
        hydraulic_cost = length / max(diameter, 0.05)

        G.add_edge(start, end,
                   pipe_id=r["id"],
                   length=length,
                   hydraulic=hydraulic_cost)
    return G


def shortest_graph_distance(G, start_pt, end_pt, weight="length"):
    if start_pt not in G or end_pt not in G:
        return np.nan
    try:
        dist = nx.shortest_path_length(G, start_pt, end_pt, weight=weight)
        return float(dist)
    except Exception:
        return np.nan


# ============================================================
# Compute neighbour deltas
# ============================================================
def compute_neighbour_stats(df, city_data):
    junction_map = defaultdict(list)

    for r in city_data["roads"]:
        if r["id"] >= 1e8:
            continue
        j1 = tuple(r["start"])
        j2 = tuple(r["end"])
        junction_map[j1].append(r["id"])
        junction_map[j2].append(r["id"])

    neigh_mean_flow = []
    neigh_mean_pressure = []

    df_index = {pid: i for i, pid in enumerate(df["pipe_id"])}

    for _, row in df.iterrows():
        pid = row["pipe_id"]

        # find junction neighbours
        r = next((p for p in city_data["roads"] if p["id"] == pid), None)
        if r is None:
            neigh_mean_flow.append(np.nan)
            neigh_mean_pressure.append(np.nan)
            continue

        j1, j2 = tuple(r["start"]), tuple(r["end"])
        neighbours = set(junction_map[j1] + junction_map[j2])
        neighbours.discard(pid)

        if not neighbours:
            neigh_mean_flow.append(0.0)
            neigh_mean_pressure.append(0.0)
            continue

        flows, pressures = [], []
        for nid in neighbours:
            if nid in df_index:
                idx = df_index[nid]
                flows.append(df.iloc[idx]["delta_mean_flow"])
                pressures.append(df.iloc[idx]["delta_mean_pressure"])

        neigh_mean_flow.append(np.mean(flows) if flows else 0.0)
        neigh_mean_pressure.append(np.mean(pressures) if pressures else 0.0)

    df["neigh_delta_flow"] = neigh_mean_flow
    df["neigh_delta_pressure"] = neigh_mean_pressure

    return df


# ============================================================
# Generate data for one town
# ============================================================
def generate_for_one_town(city_data, town_seed, n_leaks=3):
    rows = []

    print(f"▶ Baseline simulation for town {town_seed}")
    baseline = run_simulation(city_data)

    G = build_graph(city_data["roads"])

    for li in range(n_leaks):
        print(f"   → Leak {li+1}/{n_leaks}")

        leaky_data, leak_pid, (lx, ly) = inject_random_leak(city_data)
        leaky = run_simulation(leaky_data)

        df = compute_pipe_deltas(baseline, leaky)
        df = df[df["pipe_id"] < 1e8].copy()

        # ----------------------------------------
        # Compute spatial distance
        # ----------------------------------------
        pipe_lookup = {r["id"]: r for r in city_data["roads"]}

        px_list, py_list, dist_list = [], [], []

        for pid in df["pipe_id"]:
            r = pipe_lookup.get(pid)

            if r is None:
                px_list.append(np.nan)
                py_list.append(np.nan)
                dist_list.append(np.nan)
                continue

            px, py = midpoint(r["start"], r["end"])
            px_list.append(px)
            py_list.append(py)

            dist_list.append(sqrt((px - lx)**2 + (py - ly)**2))

        df["pipe_x"] = px_list
        df["pipe_y"] = py_list
        df["distance_to_leak"] = dist_list
        df["distance_log"] = np.log1p(df["distance_to_leak"])

        # ----------------------------------------
        # Graph/hydraulic distance
        # ----------------------------------------
        graph_dists = []
        hydraulic_dists = []

        leak_pt = (lx, ly)

        # find closest junction to leak
        leak_node = min(G.nodes, key=lambda n: (n[0]-lx)**2 + (n[1]-ly)**2)

        for pid in df["pipe_id"]:
            r = pipe_lookup.get(pid)
            if not r:
                graph_dists.append(np.nan)
                hydraulic_dists.append(np.nan)
                continue

            mid = midpoint(r["start"], r["end"])
            node = min(G.nodes, key=lambda n: (n[0]-mid[0])**2 + (n[1]-mid[1])**2)

            graph_dists.append(shortest_graph_distance(G, leak_node, node, weight="length"))
            hydraulic_dists.append(shortest_graph_distance(G, leak_node, node, weight="hydraulic"))

        df["graph_distance"] = graph_dists
        df["hydraulic_distance"] = hydraulic_dists

        # ----------------------------------------
        # Pipe metadata
        # ----------------------------------------
        lengths = []
        ages = []
        main = []
        branch = []

        for pid in df["pipe_id"]:
            r = pipe_lookup.get(pid)
            if not r:
                lengths.append(np.nan)
                ages.append(np.nan)
                main.append(0)
                branch.append(0)
                continue

            L = sqrt((r["start"][0]-r["end"][0])**2 + (r["start"][1]-r["end"][1])**2)
            lengths.append(L)
            ages.append(r.get("age", 0))
            main.append(1 if r["pipe_type"] == "main" else 0)
            branch.append(1 if r["pipe_type"] == "branch" else 0)

        df["pipe_length"] = lengths
        df["pipe_age"] = ages
        df["type_main"] = main
        df["type_branch"] = branch

        # ----------------------------------------
        # Neighbour features
        # ----------------------------------------
        df = compute_neighbour_stats(df, city_data)

        # Metadata
        df["leak_x"] = lx
        df["leak_y"] = ly
        df["leak_pipe_id"] = leak_pid
        df["town_seed"] = town_seed
        df["scenario"] = f"{town_seed}_{li}"

        rows.append(df)

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Main
# ============================================================
def main(n_towns=200, n_leaks_per_town=3):
    all_rows = []

    for i in range(n_towns):
        print(f"\n🏙 Generating town {i+1}/{n_towns}")
        city_data, seed = generate_town()

        df_town = generate_for_one_town(city_data, seed, n_leaks_per_town)
        all_rows.append(df_town)

    dataset = pd.concat(all_rows, ignore_index=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print("\n====================================")
    print(f"✅ Saved leak dataset to {OUTPUT_PATH.resolve()}")
    print(f"📊 Total rows: {len(dataset)}")
    print("====================================")


if __name__ == "__main__":
    main()
