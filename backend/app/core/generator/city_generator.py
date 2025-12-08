import numpy as np
import heapq
import random
from shapely.geometry import LineString
from .config import CONFIG
from .districts import DistrictAssigner
from .roads import RoadSegment, RoadNetwork
from .buildings import BuildingPlanner, Building
from .terrain import TerrainGenerator


class CityGenerator:
    def __init__(self, map_size=(2000, 2000), seed=None):
        Building._id_counter = 0
        RoadSegment._id_counter = 0
        self.map_size = map_size
        self.seed = seed or random.randint(0, 2**12 - 1)
        np.random.seed(self.seed)
        self.network = RoadNetwork()
        self.terrain = TerrainGenerator(self.seed)
        self.buildings = []

    # --- Main road generation ---
    def _start_point_angle(self, d, w, h):
        per = 2 * (w + h)
        if d < w: return (d, 0), 90
        elif d < w + h: return (w, d - w), 180
        elif d < 2 * w + h: return (w - (d - (w + h)), h), -90
        else: return (0, h - (d - (2 * w + h))), 0

    def _generate_main_roads(self):
        w, h = self.map_size
        per = 2 * (w + h)
        for i in range(CONFIG["num_highways"]):
            d = (i + 1) * per / (CONFIG["num_highways"] + 1)
            start, base = self._start_point_angle(d, w, h)
            var = np.random.randint(CONFIG["highway_branch_angle_range"][0],
                                    CONFIG["highway_branch_angle_range"][1] + 1)
            angle = base + var
            dir_vec = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            length = np.random.randint(CONFIG["highway_min_length"], CONFIG["highway_max_length"])
            end = tuple(np.array(start) + dir_vec * length)
            seg = RoadSegment(start, end, 0, {"pipe_type": "main"})
            self.network.add(seg)
            heapq.heappush(self.network.priority_queue, (0, seg))

    # --- Branch generation identical to old ---
    def _angle_between(self, v1, v2):
        dot = np.dot(v1, v2)
        mag = np.linalg.norm(v1) * np.linalg.norm(v2)
        if mag < 1e-8:
            return 180.0
        return np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0)))

    def _is_too_close(self, origin, new_dir, threshold=20):
        for r in self.network.roads:
            if r.start == origin or r.end == origin:
                existing = np.array(r.end) - np.array(r.start)
                angle = self._angle_between(existing, new_dir)
                if abs(angle) < threshold:
                    return True
        return False

    def _generate_new_branches(self, segment):
        new_segments = []
        direction = np.array(segment.end) - np.array(segment.start)
        if np.linalg.norm(direction) == 0:
            return new_segments
        direction /= np.linalg.norm(direction)
        max_roads = 2 if segment.metadata.get("pipe_type") == "main" else 4
        if self.network.intersections.get(segment.end, 0) >= max_roads:
            return new_segments

        # Extend main
        if segment.metadata.get("pipe_type") == "main":
            angle = np.random.randint(*CONFIG["highway_branch_angle_range"])
            rot = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                            [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
            new_dir = rot @ direction
            L = np.random.randint(CONFIG["highway_min_length"], CONFIG["highway_max_length"])
            end = tuple(np.array(segment.end) + new_dir * L)
            new_segments.append(RoadSegment(segment.end, end, segment.time_delay + 1, {"pipe_type": "main"}))

        # Perpendicular side roads
        if np.random.random() < CONFIG["branch_probability"]:
            for base_angle in [90, -90]:
                var = np.random.randint(*CONFIG["side_road_branch_variation"])
                angle = base_angle + var
                rot = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
                side_dir = rot @ direction
                if not self._is_too_close(segment.end, side_dir, 20):
                    L = np.random.randint(CONFIG["side_road_min_length"], CONFIG["side_road_max_length"])
                    end = tuple(np.array(segment.end) + side_dir * L)
                    new_segments.append(RoadSegment(segment.end, end, segment.time_delay + 1, {"pipe_type": "side"}))

        return new_segments

    # --- Highway connection identical to old ---
    def connect_highways(self):
        from shapely.geometry import LineString
        mains = [r for r in self.network.roads if r.metadata.get("pipe_type") == "main"]
        points = set()
        for r in mains:
            points.add(r.start)
            points.add(r.end)
        points = list(points)
        if len(points) < 2:
            return

        parent = {p: p for p in points}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            parent[find(x)] = find(y)

        for r in mains:
            if r.start in parent and r.end in parent:
                union(r.start, r.end)

        edges = [(np.linalg.norm(np.array(p1) - np.array(p2)), p1, p2)
                 for i, p1 in enumerate(points) for p2 in points[i + 1:]]
        edges.sort(key=lambda x: x[0])

        new_mains = []
        for dist, p1, p2 in edges:
            if find(p1) != find(p2):
                union(p1, p2)
                seg = RoadSegment(p1, p2, 0, {"pipe_type": "main"})
                self.network.add(seg)
                new_mains.append(seg)

        for main in new_mains:
            line = LineString([main.start, main.end])
            for r in self.network.roads:
                if r.metadata.get("pipe_type") != "side":
                    continue
                sline = LineString([r.start, r.end])
                if sline.crosses(line):
                    inter = sline.intersection(line)
                    if inter.geom_type == "Point":
                        new_end = (inter.x, inter.y)
                        r.end = new_end
                        self.network.intersections[new_end] = self.network.intersections.get(new_end, 0) + 1

    # --- Main entrypoint ---
    def generate(self):
        self._generate_main_roads()
        for _ in range(CONFIG["max_iterations"]):
            if not self.network.priority_queue:
                break
            _, seg = heapq.heappop(self.network.priority_queue)
            self.network.add(seg)
            for new in self._generate_new_branches(seg):
                if self.network.apply_constraints(new):
                    self.network.add(new)
                    heapq.heappush(self.network.priority_queue, (new.time_delay, new))

        self.connect_highways()
        planner = BuildingPlanner(self.network)
        self.buildings = planner.generate()

        # --- assign districts after generation ---
        DistrictAssigner(self.map_size).assign(self.buildings)

        # --- compute terrain map ---
        bounds = self._calc_bounds()
        height_map = self.terrain.generate_height_map(bounds)

        # --- build final JSON ---
        output_data = {
            "seed": self.seed,
            "height_map_bounds": list(bounds),
            "height_map_resolution": CONFIG["height_map_resolution"],
            "height_map": height_map,
            "roads": [r.to_dict() for r in self.network.roads],
            "buildings": []
        }

        # add back terrain height + district for each building
        for b in self.buildings:
            bx, by = b.center
            bdict = b.to_dict()
            bdict["terrain_height"] = self.terrain.get_height(bx, by)
            bdict["district"] = getattr(b, "district", "undefined")
            output_data["buildings"].append(bdict)

        return _convert_to_native(output_data)

    def _calc_bounds(self, margin=0.1):
        xs, ys = [], []
        for r in self.network.roads:
            xs += [r.start[0], r.end[0]]
            ys += [r.start[1], r.end[1]]
        for b in self.buildings:
            for (x, y) in b.corners:
                xs.append(x)
                ys.append(y)
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        dx, dy = (maxx - minx) * margin, (maxy - miny) * margin
        return minx - dx, miny - dy, maxx + dx, maxy + dy

def _convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, np.generic):  # np.int32, np.float64, etc.
        return obj.item()
    else:
        return obj