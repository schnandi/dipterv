import numpy as np
import heapq
import random
from shapely.geometry import LineString, Point
from .config import CONFIG


class RoadSegment:
    _id_counter = 0

    def __init__(self, start, end, time_delay=0, metadata=None):
        if metadata is None:
            metadata = {"pipe_type": "side"}
        self.start = start
        self.end = end
        self.time_delay = time_delay
        self.metadata = metadata
        self.id = RoadSegment._id_counter
        RoadSegment._id_counter += 1
        self.age = random.randint(0, 50)

    def length(self):
        return np.linalg.norm(np.array(self.end) - np.array(self.start))

    def to_dict(self):
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "time_delay": self.time_delay,
            "pipe_type": self.metadata.get("pipe_type", "side"),
            "age": self.age
        }

    def __lt__(self, other):
        return self.time_delay < other.time_delay


class RoadNetwork:
    def __init__(self):
        self.roads = []
        self.priority_queue = []
        self.intersections = {}

    def add(self, segment):
        if segment not in self.roads:
            self.roads.append(segment)
            self.intersections[segment.start] = self.intersections.get(segment.start, 0) + 1
            self.intersections[segment.end] = self.intersections.get(segment.end, 0) + 1

    def split(self, road, point):
        """Split existing road at point."""
        meta = road.metadata.copy()
        s1 = RoadSegment(road.start, point, road.time_delay, meta)
        s2 = RoadSegment(point, road.end, road.time_delay, meta)
        if road in self.roads:
            self.roads.remove(road)
        self.roads.extend([s1, s2])
        heapq.heappush(self.priority_queue, (s1.time_delay, s1))
        heapq.heappush(self.priority_queue, (s2.time_delay, s2))

    def apply_constraints(self, segment):
        """Intersection/snap rules identical to old generator."""
        new_line = LineString([segment.start, segment.end])
        min_distance = CONFIG["highway_snap_distance"] if segment.metadata.get("pipe_type") == "main" \
                       else CONFIG["intersection_snap_distance"]

        for road in self.roads:
            road_line = LineString([road.start, road.end])
            if new_line.intersects(road_line):
                inter = new_line.intersection(road_line)
                if inter.geom_type == 'Point':
                    ip = tuple(inter.coords[0])
                    if ip in [road.start, road.end]:
                        continue
                    if any(np.linalg.norm(np.array(ip) - np.array(pt)) < CONFIG["min_intersection_separation"]
                           for pt in self.intersections.keys()):
                        return False

                    rt, st = road.metadata.get("pipe_type"), segment.metadata.get("pipe_type")
                    if rt == "main" and st == "main":
                        return False
                    if rt == "main" and st != "main":
                        return False
                    if rt != "main" and st == "main":
                        self.split(road, ip)
                        return False
                    if rt != "main" and st != "main":
                        self.split(road, ip)
                        return False

            dist_to_end = road_line.distance(Point(segment.end))
            if dist_to_end < min_distance:
                segment.end = road.end
                return True
        return True
