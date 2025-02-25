import numpy as np
import heapq  # Priority queue for procedural generation
import json
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, Polygon
import matplotlib.pyplot as plt
from noise import pnoise2
import math

class RoadSegment:
    """Represents a road segment."""
    def __init__(self, start, end, time_delay=0, metadata={"highway": False}):
        self.start = start
        self.end = end
        self.time_delay = time_delay
        self.metadata = metadata if metadata else {}

    def length(self):
        return np.linalg.norm(np.array(self.end) - np.array(self.start))

    def __lt__(self, other):
        return self.time_delay < other.time_delay

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "time_delay": self.time_delay,
            "is_highway": self.metadata.get("highway", False)
        }
    
class Building:
    """Represents a building as a rotated rectangle."""
    def __init__(self, center, width, height, angle, corners, building_type="residential"):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle  # Store rotation angle in degrees
        self.corners = corners  # Store actual corner coordinates
        self.building_type = building_type

    def to_dict(self):
        return {
            "center": self.center,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
            "corners": self.corners,
            "building_type": self.building_type
        }

class CityGenerator:
    """Procedurally generates a city road network based on Parish & Müller (2001) algorithm."""

    def __init__(self, map_size=(2000, 2000), seed=None):
        self.map_size = map_size
        self.seed = seed
        self.road_network = []
        self.buildings = []
        self.priority_queue = []
        self.intersection_count = {}  # Track intersections
        self.population_density = self._generate_population_density()

    def _generate_population_density(self):
        """Generate population density using Perlin noise."""
        width, height = self.map_size
        density = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                value = (
                    pnoise2(i / 500, j / 500, octaves=3) +
                    pnoise2(i / 1000, j / 1000, octaves=2) +
                    pnoise2(i / 2000, j / 2000, octaves=1)
                )
                density[i, j] = (value + 1) / 2  # Normalize
        return density

    def _add_segment(self, segment):
        """Add a road segment to the network, tracking intersections."""
        if segment not in self.road_network:
            self.road_network.append(segment)
            self.intersection_count[segment.start] = self.intersection_count.get(segment.start, 0) + 1
            self.intersection_count[segment.end] = self.intersection_count.get(segment.end, 0) + 1

    def _apply_local_constraints(self, segment):
        """Ensure new roads follow constraints: intersection handling, snapping, and merging."""
        new_line = LineString([segment.start, segment.end])
        min_distance = 60.0  # Minimum snapping distance

        for road in self.road_network:
            road_line = LineString([road.start, road.end])

            # 1️⃣ **Intersection Handling**
            if new_line.intersects(road_line):
                intersection = new_line.intersection(road_line)
                if intersection.geom_type == 'Point':
                    intersection_point = tuple(intersection.coords[0])
                    if intersection_point not in [road.start, road.end]:
                        self._split_road(road, intersection_point)
                        return False  # Retry
    
            # 2️⃣ **Snap to Close Roads**
            if road_line.distance(Point(segment.end)) < min_distance:
                segment.end = road.end
                return True

        return True

    def _split_road(self, road, intersection_point):
        """Splits an existing road into two at an intersection."""
        new_segment1 = RoadSegment(road.start, intersection_point, road.time_delay)
        new_segment2 = RoadSegment(intersection_point, road.end, road.time_delay)

        self.road_network.remove(road)
        self.road_network.append(new_segment1)
        self.road_network.append(new_segment2)

        heapq.heappush(self.priority_queue, (new_segment1.time_delay, new_segment1))
        heapq.heappush(self.priority_queue, (new_segment2.time_delay, new_segment2))

    def _generate_new_branches(self, segment):
        """Generate new road segments from existing ones."""
        new_segments = []
        direction = np.array(segment.end) - np.array(segment.start)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return []  # Skip processing if the direction vector is zero
        direction = direction / norm

        # 🚨 **Prevent Overgrowth from Intersections**
        if self.intersection_count.get(segment.end, 0) >= 3:
            return new_segments

        # 📏 **Extend Straight if Highway**
        if segment.metadata.get("highway", False):
            angle = np.random.randint(-30, 30)
            rotation_matrix = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])
            direction = rotation_matrix @ direction  # Apply turn

            new_end = tuple(np.array(segment.end) + direction * (100 + np.random.randint(0, 100)))
            new_segments.append(RoadSegment(segment.end, new_end, segment.time_delay + 1, {"highway": True}))

        # 🌟 **Perpendicular Branching**
        if np.random.random() < 0.5:  # 50% chance to branch
            for angle in [90, -90]:
                rotation_matrix = np.array([
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
                ])
                new_direction = rotation_matrix @ direction
                new_end = tuple(np.array(segment.end) + new_direction * np.random.randint(250, 400))
                new_segments.append(RoadSegment(segment.end, new_end, segment.time_delay + 1))

        return new_segments

    def _generate_main_roads(self):
        """Create structured highways as the backbone of the city."""
        center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
        main_highway = [
            RoadSegment((center_x, 0), (center_x, 150), 0, {"highway": True}),
            RoadSegment((0, 0), (300, 300), 0, {"highway": True}),
        ]
        for road in main_highway:
            self.road_network.append(road)
            self.intersection_count[road.start] = 1
            self.intersection_count[road.end] = 1
            heapq.heappush(self.priority_queue, (road.time_delay, road))


    def _generate_buildings_along_roads(self):
        """Generate buildings along road segments, ensuring alignment with roads and avoiding intersections."""
        existing_buildings = []  # Store existing buildings to check for intersections

        for road in self.road_network:
            if road.metadata.get("highway", False):  # Skip highways for buildings
                continue

            num_buildings = np.random.randint(2, 5)  # Random number of buildings per segment
            road_vector = np.array(road.end) - np.array(road.start)
            road_length = np.linalg.norm(road_vector)
            if road_length == 0:
                continue  # Skip degenerate road segments

            road_direction = road_vector / road_length  # Normalize direction
            perpendicular_vector = np.array([-road_direction[1], road_direction[0]])  # Perpendicular direction

            for _ in range(num_buildings):
                offset_dist = np.random.randint(0, 50) + 5  # Distance from road
                width = np.random.randint(25, 120)
                height = np.random.randint(25, 120)

                # 🔄 **Ensure longer side is aligned with the road 70% of the time**
                if np.random.random() < 0.7:
                    aligned_width, aligned_height = max(width, height), min(width, height)
                else:
                    aligned_width, aligned_height = min(width, height), max(width, height)

                # Choose left or right side of the road
                side_multiplier = 1 if np.random.random() < 0.5 else -1
                offset = perpendicular_vector * offset_dist * side_multiplier

                # Place the building in the middle of the road segment with offset
                center = np.array(road.start) + (road_vector / 2) + offset

                # Get rotation angle in degrees
                angle = math.degrees(math.atan2(road_vector[1], road_vector[0]))

                # **Compute rotated rectangle corners using rotation matrices**
                half_width = aligned_width / 2
                half_height = aligned_height / 2

                rotation_matrix = np.array([
                    [math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
                    [math.sin(math.radians(angle)), math.cos(math.radians(angle))]
                ])

                relative_corners = np.array([
                    [-half_width, -half_height],  # Bottom-left
                    [half_width, -half_height],   # Bottom-right
                    [half_width, half_height],    # Top-right
                    [-half_width, half_height]    # Top-left
                ])

                # Rotate corners
                rotated_corners = [tuple(center + rotation_matrix @ corner) for corner in relative_corners]

                new_building_poly = Polygon(rotated_corners)

                # 🚫 **Ensure no intersection with existing buildings**
                if any(new_building_poly.intersects(Polygon(b.corners)) for b in existing_buildings):
                    continue  # Skip this building if it overlaps another building

                # 🚫 **Ensure no intersection with any road in the network**
                if any(new_building_poly.intersects(LineString([r.start, r.end])) for r in self.road_network):
                    continue  # Skip if building intersects any road


                # ✅ Successfully placed building
                new_building = Building(tuple(center), aligned_width, aligned_height, angle, rotated_corners)
                self.buildings.append(new_building)
                existing_buildings.append(new_building)  # Store for future intersection checks





    def generate(self):
        """Run procedural road generation using priority queue."""
        self._generate_main_roads()
        max_iterations = 50  # Control growth

        for _ in range(max_iterations):
            if not self.priority_queue:
                break

            _, segment = heapq.heappop(self.priority_queue)
            self._add_segment(segment)
            new_roads = self._generate_new_branches(segment)

            for new_segment in new_roads:
                if self._apply_local_constraints(new_segment):
                    self._add_segment(new_segment)
                    heapq.heappush(self.priority_queue, (new_segment.time_delay, new_segment))

        self._generate_buildings_along_roads()

        return self.road_network

# Run the generator
generator = CityGenerator(map_size=(2000, 2000), seed=2353)
roads = generator.generate()

output_data = {
    "roads": [road.to_dict() for road in roads],
    "buildings": [building.to_dict() for building in generator.buildings]
}

# Save to JSON
with open("city_data.json", "w") as f:
    json.dump(output_data, f, indent=4)
