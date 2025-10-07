import numpy as np
import heapq  # Priority queue for procedural generation
from shapely.geometry import LineString, Point, Polygon
from perlin_noise import PerlinNoise
import math
import random  # for generating a random seed if none provided

# -------------------------- CONFIGURATION --------------------------
CONFIG = {"num_highways": 5, "highway_min_length": 150, "highway_max_length": 350, "side_road_min_length": 100,
          "side_road_max_length": 150, "highway_branch_angle_range": (-30, 30), "side_road_branch_variation": (-5, 5),
          "branch_probability": 0.7, "max_iterations": 200, "intersection_snap_distance": 100,
          "highway_snap_distance": 1, "height_noise_scale": 1000, "height_octaves": 6, "height_persistence": 0.5,
          "height_lacunarity": 2.0, "height_amplitude": 100, "min_intersection_separation": 5.0, "height_curve_exponent": 1.5,
          "height_map_resolution": (256, 256)}


# --------------------------------------------------------------------

class RoadSegment:
    """Represents a road segment (pipe) with a type: main, side, or building connection."""
    _id_counter = 0

    def __init__(self, start, end, time_delay=0, metadata=None):
        if metadata is None:
            metadata = {"pipe_type": "side"}  # Default to side road if not specified
        self.start = start
        self.end = end
        self.time_delay = time_delay
        self.metadata = metadata
        self.id = RoadSegment._id_counter
        RoadSegment._id_counter += 1
        self.age = random.randint(0, 50)

    def length(self):
        return np.linalg.norm(np.array(self.end) - np.array(self.start))

    def __lt__(self, other):
        return self.time_delay < other.time_delay

    def to_dict(self):
        """Convert to dictionary. Heights will be added later in final output step."""
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "time_delay": self.time_delay,
            "pipe_type": self.metadata.get("pipe_type", "side"),
            "age": self.age,
        }


class Building:
    """Represents a building as a rotated rectangle."""
    _id_counter = 0

    def __init__(self, center, width, height, angle, corners, building_type="residential"):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle  # Store rotation angle in degrees
        self.corners = corners  # Store actual corner coordinates
        self.building_type = building_type
        self.id = Building._id_counter
        Building._id_counter += 1

    def to_dict(self):
        """Convert to dictionary. Heights will be added later in final output step."""
        return {
            "id": self.id,
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
            seed = random.randint(0, 2 ** 12 - 1)
        self.seed = seed

        # Ensure np.random uses this seed
        np.random.seed(self.seed)

        self.road_network = []
        self.buildings = []
        self.priority_queue = []
        self.intersection_count = {}  # Track how many roads connect at each coordinate
        self.height_noise = PerlinNoise(octaves=CONFIG["height_octaves"], seed=seed)

    def _get_terrain_height(self, x, y):
        """
        Compute height via PerlinNoise (replacement for pnoise2).
        """
        noise_val = self.height_noise([
            x / CONFIG["height_noise_scale"],
            y / CONFIG["height_noise_scale"]
        ])

        # normalize to [0,1]
        h = (noise_val + 1) * 0.5

        # bias towards low-elevation
        exp = CONFIG["height_curve_exponent"]
        h = h ** exp

        return h * CONFIG["height_amplitude"]

    def _add_segment(self, segment):
        """Add a road segment to the network, tracking intersections."""
        if segment not in self.road_network:
            self.road_network.append(segment)
            self.intersection_count[segment.start] = self.intersection_count.get(segment.start, 0) + 1
            self.intersection_count[segment.end] = self.intersection_count.get(segment.end, 0) + 1

    def _apply_local_constraints(self, segment):
        """Ensure new roads follow constraints: intersection handling, snapping, and merging."""
        new_line = LineString([segment.start, segment.end])
        min_distance = CONFIG["highway_snap_distance"] if segment.metadata.get("pipe_type") == "main" else CONFIG[
            "intersection_snap_distance"]

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
                    if any(np.linalg.norm(np.array(intersection_point) - np.array(pt)) < CONFIG[
                        "min_intersection_separation"] for pt in self.intersection_count.keys()):
                        return False

                    # Handle different pipe types:
                    if road.metadata.get("pipe_type") == "main" and segment.metadata.get("pipe_type") == "main":
                        # Main vs. Main: pass over each other, no intersection
                        return False

                    if road.metadata.get("pipe_type") == "main" and segment.metadata.get("pipe_type") != "main":
                        # Existing is main, new is side/building connection => discard new road
                        return False

                    if road.metadata.get("pipe_type") != "main" and segment.metadata.get("pipe_type") == "main":
                        # Existing is side, new is main => split the side road
                        self._split_road(road, intersection_point)
                        return False

                    if road.metadata.get("pipe_type") != "main" and segment.metadata.get("pipe_type") != "main":
                        # Both are side (or building connection) => normal intersection splitting
                        self._split_road(road, intersection_point)
                        return False

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

        if road in self.road_network:
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
        max_roads = 2 if segment.metadata.get("pipe_type") == "main" else 4
        if self.intersection_count.get(segment.end, 0) >= max_roads:
            return new_segments

        # 📏 Extend Straight if Main
        if segment.metadata.get("pipe_type") == "main":
            angle = np.random.randint(CONFIG["highway_branch_angle_range"][0],
                                      CONFIG["highway_branch_angle_range"][1] + 1)
            rotation_matrix = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])
            direction_main = rotation_matrix @ direction
            length_main = np.random.randint(CONFIG["highway_min_length"], CONFIG["highway_max_length"] + 1)
            new_end_main = tuple(np.array(segment.end) + direction_main * length_main)
            new_segments.append(RoadSegment(segment.end, new_end_main, segment.time_delay + 1, {"pipe_type": "main"}))

        # 🌟 Perpendicular Branching (side roads)
        if np.random.random() < CONFIG["branch_probability"]:
            for base_angle in [90, -90]:
                variation = np.random.randint(CONFIG["side_road_branch_variation"][0],
                                              CONFIG["side_road_branch_variation"][1] + 1)
                angle = base_angle + variation
                rotation_matrix = np.array([
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
                ])
                new_dir = rotation_matrix @ direction
                if not self._is_too_close_in_angle(segment.end, new_dir, angle_threshold=20):
                    length_side = np.random.randint(CONFIG["side_road_min_length"], CONFIG["side_road_max_length"] + 1)
                    new_end_side = tuple(np.array(segment.end) + new_dir * length_side)
                    new_segments.append(
                        RoadSegment(segment.end, new_end_side, segment.time_delay + 1, {"pipe_type": "side"}))

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
        """Create structured main roads (previously highways) as the backbone of the city."""
        num_main = CONFIG["num_highways"]
        width, height = self.map_size
        perimeter = 2 * (width + height)
        # Evenly distribute starting positions along the perimeter.
        for i in range(num_main):
            d = (i + 1) * perimeter / (num_main + 1)
            start, base_angle = self._get_start_point_and_base_angle(d, width, height)
            # Add a small random variation to the base angle.
            var = np.random.randint(CONFIG["highway_branch_angle_range"][0],
                                    CONFIG["highway_branch_angle_range"][1] + 1)
            angle = base_angle + var
            direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            length = np.random.randint(CONFIG["highway_min_length"], CONFIG["highway_max_length"] + 1)
            end = tuple(np.array(start) + direction * length)
            main_segment = RoadSegment(start, end, 0, {"pipe_type": "main"})
            self.road_network.append(main_segment)
            self.intersection_count[main_segment.start] = 1
            self.intersection_count[main_segment.end] = 1
            heapq.heappush(self.priority_queue, (main_segment.time_delay, main_segment))

    def _generate_buildings_along_roads(self):
        """Generate buildings along road segments, properly connecting and splitting roads."""
        existing_buildings = []

        for road in list(self.road_network):  # Use a copy since we'll modify self.road_network
            # Only generate buildings along side roads (skip main roads and building connections)
            if road.metadata.get("pipe_type") != "side":
                continue

            num_buildings = np.random.randint(2, 5)
            road_vector = np.array(road.end) - np.array(road.start)
            road_length = np.linalg.norm(road_vector)
            if road_length == 0:
                continue

            road_direction = road_vector / road_length
            perpendicular_vector = np.array([-road_direction[1], road_direction[0]])

            for _ in range(num_buildings):
                offset_dist = np.random.randint(10, 55)
                width_build = np.random.randint(25, 120)
                height_build = np.random.randint(25, 120)

                # Align longer side with the road most of the time
                if np.random.random() < 0.7:
                    aligned_width, aligned_height = max(width_build, height_build), min(width_build, height_build)
                else:
                    aligned_width, aligned_height = min(width_build, height_build), max(width_build, height_build)

                side_multiplier = 1 if np.random.random() < 0.5 else -1
                offset = perpendicular_vector * offset_dist * side_multiplier

                # Randomly position the building along the road segment
                placement_factor = np.random.uniform(0.2, 0.8)
                point_along_road = np.array(road.start) + road_vector * placement_factor
                building_center = point_along_road + offset

                # Rotation angle matching road direction
                angle = math.degrees(math.atan2(road_vector[1], road_vector[0]))

                half_width = aligned_width / 2
                half_height = aligned_height / 2

                rotation_matrix = np.array([
                    [math.cos(np.radians(angle)), -math.sin(np.radians(angle))],
                    [math.sin(np.radians(angle)), math.cos(np.radians(angle))]
                ])

                relative_corners = np.array([
                    [-half_width, -half_height],
                    [half_width, -half_height],
                    [half_width, half_height],
                    [-half_width, half_height]
                ])

                rotated_corners = [tuple(building_center + rotation_matrix @ corner) for corner in relative_corners]
                new_building_poly = Polygon(rotated_corners)

                # Check for intersections
                if any(new_building_poly.intersects(Polygon(b.corners)) for b in existing_buildings):
                    continue

                if any(new_building_poly.intersects(LineString([r.start, r.end])) for r in self.road_network):
                    continue

                # Successfully placed building
                new_building = Building(tuple(building_center), aligned_width, aligned_height, angle, rotated_corners)
                self.buildings.append(new_building)
                existing_buildings.append(new_building)

                # --- Road splitting and connection ---
                road_line = LineString([road.start, road.end])
                building_connection_point = road_line.interpolate(road_line.project(Point(building_center)))
                building_connection_point_coords = (building_connection_point.x, building_connection_point.y)

                # Check distance to endpoints to avoid tiny segments
                min_sep = CONFIG["min_intersection_separation"]
                if (Point(building_connection_point).distance(Point(road.start)) > min_sep and
                        Point(building_connection_point).distance(Point(road.end)) > min_sep):
                    # Split the road segment precisely at connection point
                    self._split_road(road, building_connection_point_coords)

                # Add building connection road segment
                connection_segment = RoadSegment(building_connection_point_coords, tuple(building_center), 0,
                                                 {"pipe_type": "building connection"})
                self.road_network.append(connection_segment)

    def connect_highways(self):
        """
        Connect main road endpoints using an MST approach.
        First, incorporate the connectivity of existing main road segments so that
        only disconnected main roads are connected by new segments.
        Then, adjust any side roads that cross the newly added main roads so that
        their free endpoints snap to the main road (avoiding crossing).
        """
        # 1. Extract unique main road endpoints.
        main_points = set()
        for road in self.road_network:
            if road.metadata.get("pipe_type") == "main":
                main_points.add(road.start)
                main_points.add(road.end)
        main_points = list(main_points)
        if len(main_points) < 2:
            return

        # 2. Initialize union-find with all main road endpoints.
        parent = {pt: pt for pt in main_points}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            parent[find(x)] = find(y)

        # 3. Union endpoints for each existing main road segment.
        for road in self.road_network:
            if road.metadata.get("pipe_type") == "main":
                if road.start in parent and road.end in parent:
                    union(road.start, road.end)

        # 4. Build all possible edges between main road endpoints.
        edges = []
        for i, p1 in enumerate(main_points):
            for p2 in main_points[i + 1:]:
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                edges.append((distance, p1, p2))
        edges.sort(key=lambda x: x[0])

        # 5. Compute MST edges only connecting disconnected main road groups.
        mst_edges = []
        for dist, p1, p2 in edges:
            if find(p1) != find(p2):
                union(p1, p2)
                mst_edges.append((p1, p2))

        # 6. Add new main road segments from the MST if they don't already exist.
        new_mains = []
        for p1, p2 in mst_edges:
            exists = False
            for road in self.road_network:
                if road.metadata.get("pipe_type") == "main":
                    if (np.allclose(road.start, p1) and np.allclose(road.end, p2)) or \
                            (np.allclose(road.start, p2) and np.allclose(road.end, p1)):
                        exists = True
                        break
            if not exists:
                new_main = RoadSegment(p1, p2, 0, {"pipe_type": "main"})
                self.road_network.append(new_main)
                new_mains.append(new_main)
                self.intersection_count[p1] = self.intersection_count.get(p1, 0) + 1
                self.intersection_count[p2] = self.intersection_count.get(p2, 0) + 1

        # 7. Adjust side roads that cross any of the newly added main roads.
        for main_road in new_mains:
            main_line = LineString([main_road.start, main_road.end])
            for road in self.road_network:
                # Only check side roads.
                if road.metadata.get("pipe_type") != "side":
                    continue
                side_line = LineString([road.start, road.end])
                if side_line.crosses(main_line):
                    intersection = side_line.intersection(main_line)
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

    def assign_building_districts(self):
        total_buildings = len(self.buildings)
        # Define an industrial center; here we choose a point near the top-right edge.
        industrial_center = np.array([self.map_size[0] * 0.9, self.map_size[1] * 0.9])

        # Get building centers as a numpy array.
        centers = np.array([b.center for b in self.buildings])
        # Compute distances from each building to the industrial center.
        distances = np.linalg.norm(centers - industrial_center, axis=1)

        # Sort building indices by distance (closest first).
        sorted_indices = np.argsort(distances)
        # Calculate how many should be industrial (~20%).
        num_industrial = int(total_buildings * 0.2)
        industrial_indices = set(sorted_indices[:num_industrial])

        for i, building in enumerate(self.buildings):
            if i in industrial_indices:
                building.district = "industrial"
                building.building_type = np.random.choice(
                    ["factory", "warehouse", "processing_plant"],
                    p=[0.4, 0.4, 0.2]
                )
            else:
                building.district = "residential"
                building.building_type = np.random.choice(
                    ["single_family", "apartment", "restaurant", "office"],
                    p=[0.5, 0.3, 0.1, 0.1]
                )

        # Optionally, designate a few special civic buildings among the residential ones.
        residential_buildings = [b for b in self.buildings if b.district == "residential"]
        if len(residential_buildings) >= 3:
            civic_types = ["hospital", "library", "school"]
            selected = random.sample(residential_buildings, len(civic_types))
            for building, civic in zip(selected, civic_types):
                building.building_type = civic

    def _generate_height_map(self, bounds):
        """
        Produce a 2D grid of heights over the given world‐space bounds,
        at CONFIG["height_map_resolution"].
        """
        min_x, min_y, max_x, max_y = bounds
        res_x, res_y = CONFIG["height_map_resolution"]

        xs = np.linspace(min_x, max_x, res_x)
        ys = np.linspace(min_y, max_y, res_y)

        grid = []
        for yy in ys:
            row = [self._get_terrain_height(xx, yy) for xx in xs]
            grid.append(row)
        return grid

    def _calculate_bounds(self, margin_ratio: float = 0.1):
        """
        Scan roads and building corners to find min/max X,Y, then pad by margin_ratio.
        Returns (min_x, min_y, max_x, max_y).
        """
        xs, ys = [], []
        for r in self.road_network:
            xs.extend([r.start[0], r.end[0]])
            ys.extend([r.start[1], r.end[1]])
        for b in self.buildings:
            for (x, y) in b.corners:
                xs.append(x); ys.append(y)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # add a little padding around edges
        dx = (max_x - min_x) * margin_ratio
        dy = (max_y - min_y) * margin_ratio
        return min_x - dx, min_y - dy, max_x + dx, max_y + dy

    def generate(self):
        """Run procedural road generation using a priority queue."""
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

        # populate height map
        bounds = self._calculate_bounds(margin_ratio=0.1)

        height_map = self._generate_height_map(bounds)

        # Build final JSON output, adding terrain heights to roads & buildings
        output_data = {
            "seed": self.seed,
            "height_map_bounds": list(bounds),                # [min_x, min_y, max_x, max_y]
            "height_map_resolution": CONFIG["height_map_resolution"],
            "height_map": height_map,
            "roads": [],
            "buildings": [],
        }

        for road in self.road_network:
            rdict = road.to_dict()
            start_x, start_y = rdict["start"]
            end_x, end_y = rdict["end"]
            rdict["start_height"] = self._get_terrain_height(start_x, start_y)
            rdict["end_height"] = self._get_terrain_height(end_x, end_y)
            output_data["roads"].append(rdict)

        for bldg in self.buildings:
            bdict = bldg.to_dict()
            cx, cy = bdict["center"]
            bdict["terrain_height"] = self._get_terrain_height(cx, cy)
            bdict["district"] = getattr(bldg, "district", "undefined")
            output_data["buildings"].append(bdict)

        # Instead of writing to a file, simply return the data.
        return output_data

