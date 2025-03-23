import numpy as np
import heapq  # Priority queue for procedural generation
import json
from shapely.geometry import LineString, Point, Polygon
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from noise import pnoise2
import math
import random  # for generating a random seed if none provided
import sys

# -------------------------- CONFIGURATION --------------------------
CONFIG = {
    "num_highways": 5,                    # Number of main highways to generate
    "highway_min_length": 150,            # Minimum length for a highway segment (main highways)
    "highway_max_length": 350,            # Maximum length for a highway segment (main highways)
    "side_road_min_length": 100,          # Minimum length for a side road segment
    "side_road_max_length": 150,          # Maximum length for a side road segment
    "highway_branch_angle_range": (-30, 30),  # Range (in degrees) for highway extension rotation
    "side_road_branch_variation": (-5, 5),    # Variation (in degrees) added to perpendicular branches
    "branch_probability": 0.7,            # Chance that a branch (side road) is created
    "max_iterations": 200,                # Maximum iterations in the generation loop
    "intersection_snap_distance": 100,    # Snap distance for non-highway roads
    "highway_snap_distance": 1,           # Snap distance for highways

    # Terrain / Height Map parameters:
    "height_noise_scale": 1000,           # Larger scale => smoother, broader height changes
    "height_octaves": 2,                  # Number of octaves for Perlin noise
    "height_amplitude": 100,              # Overall vertical range of heights
    "min_intersection_separation": 5.0,    # Minimum distance between intersection points
}
# --------------------------------------------------------------------

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
        """Convert to dictionary. Heights will be added later in final output step."""
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
        """Convert to dictionary. Heights will be added later in final output step."""
        return {
            "center": self.center,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
            "corners": self.corners,
            "building_type": self.building_type
        }

class CityGenerator:
    """Procedurally generates a city road network with terrain heights."""
    def __init__(self, map_size=(2000, 2000), seed=None):
        self.map_size = map_size

        # If user does not provide a seed, generate a random one
        if seed is None:
            seed = random.randint(0, 2**12 - 1)
        self.seed = seed

        # Make sure all np.random calls use this seed
        np.random.seed(self.seed)

        self.road_network = []
        self.buildings = []
        self.priority_queue = []
        self.intersection_count = {}  # Track how many roads connect at each coordinate

    def _get_terrain_height(self, x, y):
        """
        Compute terrain height at (x, y) using Perlin noise.
        Scale and amplitude come from CONFIG.
        """
        scale = CONFIG["height_noise_scale"]
        octaves = CONFIG["height_octaves"]
        amplitude = CONFIG["height_amplitude"]
        # Perlin noise returns ~[-1..1], so shift to [0..1], then multiply by amplitude
        noise_val = pnoise2(x / scale, y / scale, octaves=octaves, base=self.seed)
        return (noise_val + 1) / 2 * amplitude

    def _add_segment(self, segment):
        """Add a road segment to the network, tracking intersections."""
        if segment not in self.road_network:
            self.road_network.append(segment)
            self.intersection_count[segment.start] = self.intersection_count.get(segment.start, 0) + 1
            self.intersection_count[segment.end] = self.intersection_count.get(segment.end, 0) + 1

    def _apply_local_constraints(self, segment):
        """Ensure new roads follow constraints: intersection handling, snapping, and merging."""
        new_line = LineString([segment.start, segment.end])
        min_distance = (
            CONFIG["highway_snap_distance"]
            if segment.metadata.get("highway", False)
            else CONFIG["intersection_snap_distance"]
        )

        for road in self.road_network:
            road_line = LineString([road.start, road.end])

            # 1️⃣ Intersection Handling
            if new_line.intersects(road_line):
                intersection = new_line.intersection(road_line)
                if intersection.geom_type == 'Point':
                    intersection_point = tuple(intersection.coords[0])

                    # Skip if intersection is at the road's existing endpoint
                    if intersection_point in [road.start, road.end]:
                        continue

                    # Check if we already have an intersection super close
                    if any(np.linalg.norm(np.array(intersection_point) - np.array(pt))
                           < CONFIG["min_intersection_separation"]
                           for pt in self.intersection_count.keys()):
                        # If too close to an existing intersection, discard
                        return False

                    # CASE A: Highway vs. Highway => pass over each other, no intersection
                    if road.metadata.get("highway", False) and segment.metadata.get("highway", False):
                        return False

                    # CASE B: Existing is highway, new is not => discard new road
                    if road.metadata.get("highway", False) and not segment.metadata.get("highway", False):
                        return False

                    # CASE C: Existing is normal, new is highway => highway splits the normal road
                    if not road.metadata.get("highway", False) and segment.metadata.get("highway", False):
                        self._split_road(road, intersection_point)
                        return False  # The generator will retry with the newly split segments

                    # CASE D: Both are non-highways => normal intersection
                    if not road.metadata.get("highway", False) and not segment.metadata.get("highway", False):
                        self._split_road(road, intersection_point)
                        return False  # The generator will retry with the newly split segments

            # 2️⃣ Snap to Close Roads
            dist_to_end = road_line.distance(Point(segment.end))
            if dist_to_end < min_distance:
                segment.end = road.end
                return True

        return True

    def _split_road(self, road, intersection_point):
        """Split the existing road into two segments at intersection_point."""
        metadata_copy = road.metadata.copy()

        new_segment1 = RoadSegment(road.start, intersection_point, road.time_delay, metadata_copy)
        new_segment2 = RoadSegment(intersection_point, road.end, road.time_delay, metadata_copy)

        self.road_network.remove(road)
        self.road_network.append(new_segment1)
        self.road_network.append(new_segment2)

        heapq.heappush(self.priority_queue, (new_segment1.time_delay, new_segment1))
        heapq.heappush(self.priority_queue, (new_segment2.time_delay, new_segment2))

    def _angle_between(self, v1, v2):
        dot = np.dot(v1, v2)
        mag = np.linalg.norm(v1) * np.linalg.norm(v2)
        if mag < 1e-8:
            return 180.0
        return np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0)))

    def _is_too_close_in_angle(self, origin, new_direction, angle_threshold=20):
        """Check if new_direction is within angle_threshold of an existing road from 'origin'."""
        for road in self.road_network:
            if road.start == origin or road.end == origin:
                existing_dir = np.array(road.end) - np.array(road.start)
                angle_diff = self._angle_between(existing_dir, new_direction)
                if abs(angle_diff) < angle_threshold:
                    return True
        return False

    def _generate_new_branches(self, segment):
        """Generate new road segments from existing ones."""
        new_segments = []
        direction = np.array(segment.end) - np.array(segment.start)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return new_segments
        direction = direction / norm

        # 🚨 Prevent Overgrowth from Intersections
        max_roads = 2 if segment.metadata.get("highway", False) else 4
        if self.intersection_count.get(segment.end, 0) >= max_roads:
            return new_segments

        # 📏 Extend Straight if Highway
        if segment.metadata.get("highway", False):
            angle = np.random.randint(CONFIG["highway_branch_angle_range"][0],
                                      CONFIG["highway_branch_angle_range"][1] + 1)
            rotation_matrix = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]
            ])
            direction_hw = rotation_matrix @ direction
            length_hw = np.random.randint(CONFIG["highway_min_length"], CONFIG["highway_max_length"] + 1)
            new_end_hw = tuple(np.array(segment.end) + direction_hw * length_hw)
            new_segments.append(RoadSegment(segment.end, new_end_hw, segment.time_delay + 1, {"highway": True}))

        # 🌟 Perpendicular Branching (side roads)
        if np.random.random() < CONFIG["branch_probability"]:
            # Example: create up to 2 side roads, ±90°, with small variation
            for base_angle in [90, -90]:
                variation = np.random.randint(CONFIG["side_road_branch_variation"][0],
                                              CONFIG["side_road_branch_variation"][1] + 1)
                angle = base_angle + variation
                rotation_matrix = np.array([
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                    [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]
                ])
                new_dir = rotation_matrix @ direction
                if not self._is_too_close_in_angle(segment.end, new_dir, angle_threshold=20):
                    length_side = np.random.randint(CONFIG["side_road_min_length"], CONFIG["side_road_max_length"] + 1)
                    new_end_side = tuple(np.array(segment.end) + new_dir * length_side)
                    new_segments.append(RoadSegment(segment.end, new_end_side, segment.time_delay + 1))

        return new_segments

    def _get_start_point_and_base_angle(self, d, width, height):
        """Given a distance d along the perimeter, return the start point and base angle pointing into the map."""
        perimeter = 2 * (width + height)
        if d < width:
            # Bottom edge: start moves from left to right, point upward
            return (d, 0), 90
        elif d < width + height:
            # Right edge: start moves from bottom to top, point leftward
            d_prime = d - width
            return (width, d_prime), 180
        elif d < 2 * width + height:
            # Top edge: start moves from right to left, point downward
            d_prime = d - (width + height)
            return (width - d_prime, height), -90
        else:
            # Left edge: start moves from top to bottom, point rightward
            d_prime = d - (2 * width + height)
            return (0, height - d_prime), 0

    def _generate_main_roads(self):
        """Create structured highways as the backbone of the city."""
        num_highways = CONFIG["num_highways"]
        width, height = self.map_size
        perimeter = 2 * (width + height)
        # Evenly distribute starting positions along the perimeter.
        for i in range(num_highways):
            d = (i + 1) * perimeter / (num_highways + 1)
            start, base_angle = self._get_start_point_and_base_angle(d, width, height)
            # Add a small random variation to the base angle.
            var = np.random.randint(CONFIG["highway_branch_angle_range"][0],
                                    CONFIG["highway_branch_angle_range"][1] + 1)
            angle = base_angle + var
            direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            length = np.random.randint(CONFIG["highway_min_length"], CONFIG["highway_max_length"] + 1)
            end = tuple(np.array(start) + direction * length)
            highway_segment = RoadSegment(start, end, 0, {"highway": True})
            self.road_network.append(highway_segment)
            self.intersection_count[highway_segment.start] = 1
            self.intersection_count[highway_segment.end] = 1
            heapq.heappush(self.priority_queue, (highway_segment.time_delay, highway_segment))

    def _generate_buildings_along_roads(self):
        """Generate buildings along road segments, ensuring alignment with roads and avoiding intersections."""
        existing_buildings = []

        for road in self.road_network:
            # Skip highways for buildings
            if road.metadata.get("highway", False):
                continue

            num_buildings = np.random.randint(2, 5)
            road_vector = np.array(road.end) - np.array(road.start)
            road_length = np.linalg.norm(road_vector)
            if road_length == 0:
                continue

            road_direction = road_vector / road_length
            perpendicular_vector = np.array([-road_direction[1], road_direction[0]])

            for _ in range(num_buildings):
                offset_dist = np.random.randint(0, 50) + 5
                width_build = np.random.randint(25, 120)
                height_build = np.random.randint(25, 120)

                # Ensure longer side is aligned with the road 70% of the time
                if np.random.random() < 0.7:
                    aligned_width, aligned_height = max(width_build, height_build), min(width_build, height_build)
                else:
                    aligned_width, aligned_height = min(width_build, height_build), max(width_build, height_build)

                # Choose left or right side of the road
                side_multiplier = 1 if np.random.random() < 0.5 else -1
                offset = perpendicular_vector * offset_dist * side_multiplier

                # Place the building in the middle of the road segment with offset
                center = np.array(road.start) + (road_vector / 2) + offset

                # Get rotation angle in degrees
                angle = math.degrees(math.atan2(road_vector[1], road_vector[0]))

                # Compute rotated rectangle corners using rotation matrices
                half_width = aligned_width / 2
                half_height = aligned_height / 2

                rotation_matrix = np.array([
                    [math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
                    [math.sin(math.radians(angle)),  math.cos(math.radians(angle))]
                ])

                relative_corners = np.array([
                    [-half_width, -half_height],  # Bottom-left
                    [ half_width, -half_height],  # Bottom-right
                    [ half_width,  half_height],  # Top-right
                    [-half_width,  half_height]   # Top-left
                ])

                # Rotate corners
                rotated_corners = [tuple(center + rotation_matrix @ corner) for corner in relative_corners]
                new_building_poly = Polygon(rotated_corners)

                # Ensure no intersection with existing buildings
                if any(new_building_poly.intersects(Polygon(b.corners)) for b in existing_buildings):
                    continue

                # Ensure no intersection with any road in the network
                if any(new_building_poly.intersects(LineString([r.start, r.end])) for r in self.road_network):
                    continue

                # Successfully placed building
                new_building = Building(tuple(center), aligned_width, aligned_height, angle, rotated_corners)
                self.buildings.append(new_building)
                existing_buildings.append(new_building)

    def connect_highways(self):
        """
        Connect highway endpoints using an MST approach.
        First, incorporate the connectivity of existing highway segments so that
        only disconnected highways are connected by new segments.
        Then, adjust any side roads that cross the newly added highways so that
        their free endpoints snap to the highway (avoiding crossing).
        """
        # 1. Extract unique highway endpoints.
        highway_points = set()
        for road in self.road_network:
            if road.metadata.get("highway", False):
                highway_points.add(road.start)
                highway_points.add(road.end)
        highway_points = list(highway_points)
        if len(highway_points) < 2:
            return

        # 2. Initialize union-find with all highway endpoints.
        parent = {pt: pt for pt in highway_points}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            parent[find(x)] = find(y)

        # 3. Union endpoints for each existing highway segment.
        for road in self.road_network:
            if road.metadata.get("highway", False):
                if road.start in parent and road.end in parent:
                    union(road.start, road.end)

        # 4. Build all possible edges between highway endpoints.
        edges = []
        for i, p1 in enumerate(highway_points):
            for p2 in highway_points[i+1:]:
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                edges.append((distance, p1, p2))
        edges.sort(key=lambda x: x[0])

        # 5. Compute MST edges only connecting disconnected highway groups.
        mst_edges = []
        for dist, p1, p2 in edges:
            if find(p1) != find(p2):
                union(p1, p2)
                mst_edges.append((p1, p2))

        # 6. Add new highway segments from the MST if they don't already exist.
        new_highways = []
        for p1, p2 in mst_edges:
            exists = False
            for road in self.road_network:
                if road.metadata.get("highway", False):
                    if (np.allclose(road.start, p1) and np.allclose(road.end, p2)) or \
                    (np.allclose(road.start, p2) and np.allclose(road.end, p1)):
                        exists = True
                        break
            if not exists:
                new_highway = RoadSegment(p1, p2, time_delay=0, metadata={"highway": True})
                self.road_network.append(new_highway)
                new_highways.append(new_highway)
                self.intersection_count[p1] = self.intersection_count.get(p1, 0) + 1
                self.intersection_count[p2] = self.intersection_count.get(p2, 0) + 1

        # 7. Adjust side roads that cross any of the newly added highways.
        for highway in new_highways:
            highway_line = LineString([highway.start, highway.end])
            for road in self.road_network:
                # Only check non-highway (side) roads.
                if road.metadata.get("highway", False):
                    continue
                side_line = LineString([road.start, road.end])
                if side_line.crosses(highway_line):
                    intersection = side_line.intersection(highway_line)
                    if intersection.geom_type == "Point":
                        new_endpoint = (intersection.x, intersection.y)
                        # Snap the endpoint that is not the origin.
                        if not np.allclose(new_endpoint, road.start):
                            old_endpoint = road.end
                            road.end = new_endpoint
                            # Update intersection counts.
                            if old_endpoint in self.intersection_count:
                                self.intersection_count[old_endpoint] -= 1
                                if self.intersection_count[old_endpoint] <= 0:
                                    del self.intersection_count[old_endpoint]
                            self.intersection_count[new_endpoint] = self.intersection_count.get(new_endpoint, 0) + 1

    def assign_building_districts(self, n_clusters=3):
        """
        Clusters buildings into n_clusters using their center coordinates and assigns
        a district and a building type to each. For demonstration, we assume:
        - Cluster 0: residential (types: house, apartment, small_business)
        - Cluster 1: commercial (types: shop, office, restaurant)
        - Cluster 2: industrial (types: factory, warehouse, distribution_center)
        """
        # Ensure there are buildings to cluster.
        if not self.buildings:
            return

        # Extract building center coordinates.
        coords = np.array([b.center for b in self.buildings])
        
        # Use KMeans from scikit-learn for clustering.
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        labels = kmeans.fit_predict(coords)
        
        # Define a mapping from cluster label to district and building types with probabilities.          
        # (Feel free to adjust the probabilities and type names as desired.)
        district_building_types = {
            0: {"district": "residential", "types": (["house", "apartment", "small_business"], [0.7, 0.2, 0.1])},
            1: {"district": "commercial",  "types": (["shop", "office", "restaurant"], [0.4, 0.4, 0.2])},
            2: {"district": "industrial",  "types": (["factory", "warehouse", "distribution_center"], [0.5, 0.3, 0.2])},
        }
        
        # Assign each building a district and a building type.
        for i, building in enumerate(self.buildings):
            label = labels[i]
            mapping = district_building_types[label]
            building.district = mapping["district"]
            building.building_type = np.random.choice(mapping["types"][0], p=mapping["types"][1])



    def generate(self):
        """Run procedural road generation using priority queue."""
        self._generate_main_roads()
        max_iterations = CONFIG["max_iterations"]

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

        self.connect_highways()

        self._generate_buildings_along_roads()

        self.assign_building_districts()

        return self.road_network

# ------------------------- Run the Generator -------------------------
if __name__ == "__main__":
    # If user provided a seed on the command line, use it
    # Example usage: python city_gen.py 12345
    seed = None
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            pass  # if not an int, we'll ignore
    
    generator = CityGenerator(map_size=(2000, 2000), seed=seed)
    roads = generator.generate()


    # Build final JSON output, adding terrain heights to roads & buildings
    output_data = {
        "seed": generator.seed,
        "roads": [],
        "buildings": [],
    }

    # For each road, add start/end heights
    for road in generator.road_network:
        rdict = road.to_dict()
        start_x, start_y = rdict["start"]
        end_x, end_y = rdict["end"]
        rdict["start_height"] = generator._get_terrain_height(start_x, start_y)
        rdict["end_height"] = generator._get_terrain_height(end_x, end_y)
        output_data["roads"].append(rdict)

    # For each building, add terrain height at building center
    for bldg in generator.buildings:
        bdict = bldg.to_dict()
        cx, cy = bdict["center"]
        bdict["terrain_height"] = generator._get_terrain_height(cx, cy)
        # Optionally include the district information:
        bdict["district"] = getattr(bldg, "district", "undefined")
        output_data["buildings"].append(bdict)


    # Save to JSON
    with open("city_data.json", "w") as f:
        json.dump(output_data, f, indent=4)
    
    # Print the seed so user knows which was used
    print(f"Seed used: {generator.seed}")
