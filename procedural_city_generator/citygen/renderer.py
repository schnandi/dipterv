import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import json
from noise import pnoise2

# -------------------------- CONFIGURATION --------------------------
MAP_WIDTH = 2000
MAP_HEIGHT = 2000
HEIGHT_MAP_SCALE = 1000.0    # Larger => smoother, broader changes
HEIGHT_MAP_OCTAVES = 2       # Number of octaves for Perlin noise
HEIGHT_MAP_AMPLITUDE = 100   # Vertical range of heights
HEIGHT_MAP_RES = 300         # Increase resolution for full-map coverage
# --------------------------------------------------------------------

def _compute_data_bounds(city_data):
    all_x = []
    all_y = []
    
    # Roads
    for road in city_data["roads"]:
        all_x.append(road["start"][0])
        all_y.append(road["start"][1])
        all_x.append(road["end"][0])
        all_y.append(road["end"][1])
    
    # Buildings
    for building in city_data.get("buildings", []):
        if "corners" in building:
            for cx, cy in building["corners"]:
                all_x.append(cx)
                all_y.append(cy)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    return min_x, max_x, min_y, max_y

class ZoneRenderer:
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.city_data = json.load(f)
            self.seed = self.city_data.get("seed", 0)
            
        # Define a color mapping for the two districts.
        self.district_colors = {
            "residential": "lightblue",
            "industrial": "lightcoral",
            # fallback color if no district assigned:
            "undefined": "gray"
        }

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("City Zones and Building Types (Residential & Industrial)")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # 1. Compute bounding box
        min_x, max_x, min_y, max_y = _compute_data_bounds(self.city_data)
        padding = 200  # adjust as needed
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        # 2. Generate height map noise for background
        height_map = self._generate_height_map_for_extent(min_x, max_x, min_y, max_y, HEIGHT_MAP_RES)
        ax.imshow(
            height_map,
            origin="lower",
            extent=[min_x, max_x, min_y, max_y],
            cmap="terrain",
            alpha=0.5,
            interpolation="bilinear"
        )

        # 3. Draw roads and intersections (using three pipe types)
        intersection_points = set()
        labels_drawn = set()
        for road in self.city_data["roads"]:
            start, end = road["start"], road["end"]
            pipe_type = road.get("pipe_type", "side")
            if pipe_type == "main":
                style = "g-"      # green solid line
                lw = 3            # thick stroke for main roads
                label = "Main Road" if "Main Road" not in labels_drawn else None
                labels_drawn.add("Main Road")
            elif pipe_type == "side":
                style = "r-"      # red solid line
                lw = 1            # thinner stroke for side roads
                label = "Side Road" if "Side Road" not in labels_drawn else None
                labels_drawn.add("Side Road")
            elif pipe_type == "building connection":
                style = "b-"     # blue dashed line
                lw = 1            # intermediate stroke width
                label = "Building Connection" if "Building Connection" not in labels_drawn else None
                labels_drawn.add("Building Connection")
            else:
                style = "k-"
                lw = 1
                label = "Unknown Road" if "Unknown Road" not in labels_drawn else None
                labels_drawn.add("Unknown Road")
                
            ax.plot([start[0], end[0]], [start[1], end[1]], style, linewidth=lw, label=label)
            intersection_points.add(tuple(start))
            intersection_points.add(tuple(end))
        
        # Draw intersections
        for pt in intersection_points:
            scatter_label = "Intersection" if "Intersection" not in labels_drawn else None
            ax.scatter(pt[0], pt[1], color="blue", s=20, label=scatter_label)
            labels_drawn.add("Intersection")
        
        # 4. Draw buildings with colors based on their district
        for building in self.city_data.get("buildings", []):
            if "corners" not in building:
                continue
            corners = building["corners"]
            district = building.get("district", "undefined")
            face_color = self.district_colors.get(district, "gray")
            label = f"{district.capitalize()} Building" if district.capitalize() not in labels_drawn else None
            if label:
                labels_drawn.add(district.capitalize())
            poly = MplPolygon(corners, closed=True, color=face_color, alpha=0.7, label=label)
            ax.add_patch(poly)
            
            if "center" in building:
                center = building["center"]
                ax.scatter(center[0], center[1], color=face_color, edgecolor="k", s=30)

        ax.legend()
        plt.gca().set_aspect(1.0)
        plt.grid(True)
        plt.show()

    def _generate_height_map_for_extent(self, min_x, max_x, min_y, max_y, res):
        width = max_x - min_x
        height = max_y - min_y
        terrain = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                # Map grid (i,j) to coordinates in the extent.
                x = min_x + (i / (res - 1)) * width
                y = min_y + (j / (res - 1)) * height
                val = pnoise2(
                    x / HEIGHT_MAP_SCALE,
                    y / HEIGHT_MAP_SCALE,
                    octaves=HEIGHT_MAP_OCTAVES,
                    base=self.seed
                )
                val = (val + 1) / 2.0  # shift to [0..1]
                terrain[j, i] = val * HEIGHT_MAP_AMPLITUDE
        return terrain

if __name__ == "__main__":
    renderer = ZoneRenderer("city_data.json")
    renderer.render()
