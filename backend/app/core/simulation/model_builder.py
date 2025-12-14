from random import random

import numpy as np
import pandapipes as pp


def build_network(city_data, height_scale=1.0, base_pressure=10.0, baseline_daily=100.0):
    """
    Build a pandapipes network from the procedural city_data dictionary.
    Adds leaks as sinks connected through short branch pipes instead of splitting the main one.
    This way, the number of pipes and their IDs remain stable.
    Returns (net, sink_info)
    """

    multipliers = {
        "single_family": 4.0,
        "apartment": 40.0,
        "restaurant": 50.0,
        "office": 10.0,
        "hospital": 50.0,
        "library": 10.0,
        "school": 50.0,
        "factory": 50.0,
        "warehouse": 3.0,
        "processing_plant": 10.0
    }

    junction_coords = {}
    junction_heights = {}
    junction_list = []

    # Collect junctions
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

    avg_heights = {pt: np.mean(h) * height_scale for pt, h in junction_heights.items()}

    net = pp.create_empty_network(fluid="water")

    # Create junctions
    for coord in junction_list:
        pp.create_junction(
            net,
            index=junction_coords[coord],
            pn_bar=1.0,
            tfluid_k=293.15,
            height_m=avg_heights.get(coord, 0.0),
            name=f"Junction {coord}",
            geodata={"x": coord[0], "y": coord[1]},
        )

    # Determine ext_grid (highest main junction)
    main_pipe_coords = [tuple(r["start"]) for r in city_data["roads"] if r.get("pipe_type") == "main"] + \
                       [tuple(r["end"]) for r in city_data["roads"] if r.get("pipe_type") == "main"]
    if main_pipe_coords:
        supply_pt = max(main_pipe_coords, key=lambda pt: avg_heights.get(pt, 0))
    else:
        supply_pt = max(avg_heights, key=avg_heights.get)
    supply_idx = junction_coords[supply_pt]

    pp.create_ext_grid(
        net,
        junction=supply_idx,
        p_bar=base_pressure,
        t_k=293.15,
        name="External Grid"
    )

    # Create pipes
    pipe_diams = {"main": 0.1, "side": 0.06, "building connection": 0.03}
    for road in city_data["roads"]:
        start, end = tuple(road["start"]), tuple(road["end"])
        j_from, j_to = junction_coords[start], junction_coords[end]
        ptype = road.get("pipe_type", "side")
        diameter = pipe_diams.get(ptype, 0.05)
        length = np.linalg.norm(np.array(end) - np.array(start)) / 3000.0
        age = road.get("age", 50)
        k_mm = 0.01 + (0.5 - 0.01) * (age / 100.0)

        pp.create_pipe_from_parameters(
            net,
            index=road["id"],
            from_junction=j_from,
            to_junction=j_to,
            length_km=length,
            diameter_m=diameter,
            k_mm=k_mm,
            name=f"Pipe {road['id']}",
        )

        # Add leak as side-branch (short pipe + sink), not as split
        if "leak" in road:
            leak = road["leak"]
            lx, ly = leak["coord"]
            rate = leak.get("rate_kg_per_s", 0.01)
            frac = leak.get("fraction", 0.5)

            # interpolate along pipe to find attachment
            attach_x = start[0] + frac * (end[0] - start[0])
            attach_y = start[1] + frac * (end[1] - start[1])
            h_from = avg_heights.get(start, 0.0)
            h_to = avg_heights.get(end, 0.0)
            h_leak = h_from * (1 - frac) + h_to * frac

            j_attach = pp.create_junction(
                net,
                pn_bar=1.0,
                tfluid_k=293.15,
                height_m=h_leak,
                name=f"LeakJunction_{road['id']}",
                geodata={"x": attach_x, "y": attach_y},
            )


            # tiny branch pipe (0.001 km) from mid-point to leak junction
            branch_idx = pp.create_pipe_from_parameters(
                net,
                index=int(1e8 + np.random.randint(0, 1e7)),
                from_junction=j_attach,
                to_junction=j_to,
                length_km=0.001,
                diameter_m=0.02,
                k_mm=0.1,
                name=f"LeakBranch_{road['id']}",
            )
            # attach leak sink
            pp.create_sink(
                net,
                junction=j_attach,
                mdot_kg_per_s=rate,
                index=int(1e8 + np.random.randint(0, 1e7)),
                name=f"LeakSink_{road['id']}",
            )

    # Add building sinks
    sink_info = []
    for b in city_data.get("buildings", []):
        center = tuple(b["center"])
        btype = b.get("building_type", "single_family")
        if center not in junction_coords:
            j_idx = pp.create_junction(
                net,
                pn_bar=1.0,
                tfluid_k=293.15,
                height_m=b.get("terrain_height", 0.0),
                name=f"Building {b['id']}",
                geodata={"x": center[0], "y": center[1]},
            )
            junction_coords[center] = j_idx
        else:
            j_idx = junction_coords[center]

        nominal_daily_L = baseline_daily * multipliers.get(btype, 1.0)
        nominal_kg_s = nominal_daily_L / 86400.0
        sink_idx = pp.create_sink(net, junction=j_idx, mdot_kg_per_s=nominal_kg_s,
                       name=f"Sink {b['id']}", index=b['id'])
        sink_info.append({
            "building_id": b["id"],
            "building_type": btype,
            "nominal_kg_s": nominal_kg_s,
            "sink_index": sink_idx,
        })

    return net, sink_info
