import numpy as np
import math
import random
from shapely.geometry import Polygon, LineString, Point
from .config import CONFIG
from .roads import RoadSegment


class Building:
    _id_counter = 0

    def __init__(self, center, width, height, angle, corners, btype="residential"):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.corners = corners
        self.building_type = btype
        self.id = Building._id_counter
        Building._id_counter += 1

    def to_dict(self):
        return {
            "id": self.id,
            "center": self.center,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
            "corners": self.corners,
            "building_type": self.building_type
        }


class BuildingPlanner:
    def __init__(self, road_network):
        self.network = road_network
        self.buildings = []

    def generate(self):
        existing_buildings = []

        for road in list(self.network.roads):  # Copy because we modify the network
            # Only generate buildings along side roads
            if road.metadata.get("pipe_type") != "side":
                continue

            vec = np.array(road.end) - np.array(road.start)
            length = np.linalg.norm(vec)
            if length == 0:
                continue
            dir_vec = vec / length
            perp = np.array([-dir_vec[1], dir_vec[0]])

            for _ in range(np.random.randint(2, 5)):
                offset = np.random.randint(10, 55)
                w, h = np.random.randint(25, 120, 2)
                if np.random.random() < 0.7:
                    w, h = max(w, h), min(w, h)

                side = 1 if np.random.random() < 0.5 else -1
                pos_factor = np.random.uniform(0.2, 0.8)
                base = np.array(road.start) + vec * pos_factor
                center = base + perp * offset * side

                angle = math.degrees(math.atan2(vec[1], vec[0]))
                hw, hh = w / 2, h / 2
                rot = np.array([[math.cos(np.radians(angle)), -math.sin(np.radians(angle))],
                                [math.sin(np.radians(angle)), math.cos(np.radians(angle))]])
                rel = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
                corners = [tuple(center + rot @ r) for r in rel]
                new_poly = Polygon(corners)

                # 🔒 collision checks
                if any(new_poly.intersects(Polygon(b.corners)) for b in existing_buildings):
                    continue

                if any(new_poly.intersects(LineString([r.start, r.end])) for r in self.network.roads):
                    continue

                # ✅ Safe to place
                b = Building(tuple(center), w, h, angle, corners)
                self.buildings.append(b)
                existing_buildings.append(b)

                # --- add building connection pipe ---
                road_line = LineString([road.start, road.end])
                conn_pt = road_line.interpolate(road_line.project(Point(center)))
                conn_coords = (conn_pt.x, conn_pt.y)

                # avoid microscopic segments
                min_sep = CONFIG["min_intersection_separation"]
                if (Point(conn_coords).distance(Point(road.start)) > min_sep and
                    Point(conn_coords).distance(Point(road.end)) > min_sep):
                    self.network.split(road, conn_coords)

                conn_seg = RoadSegment(conn_coords, tuple(center), 0, {"pipe_type": "building connection"})
                self.network.roads.append(conn_seg)

        return self.buildings
