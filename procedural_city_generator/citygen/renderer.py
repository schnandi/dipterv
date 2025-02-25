import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import json

class CityRenderer:
    """Renders a procedurally generated city using Matplotlib."""
    
    def __init__(self, json_file):
        """Load city data from JSON."""
        with open(json_file, "r") as f:
            self.city_data = json.load(f)

    def render(self):
        """Render the city layout with roads and intersections."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Procedural City Generation")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # Keep track of all intersection points
        intersection_points = set()

        # Draw roads
        for road in self.city_data["roads"]:
            start, end = tuple(road["start"]), tuple(road["end"])

            # Plot road segment
            ax.plot([start[0], end[0]], [start[1], end[1]], "r-", linewidth=2, label="Road" if "Road" not in ax.get_legend_handles_labels()[1] else "")

            # Collect intersection points
            intersection_points.add(start)
            intersection_points.add(end)

        # Draw intersections as blue dots
        for point in intersection_points:
            ax.scatter(point[0], point[1], color="blue", s=20, label="Intersection" if point == next(iter(intersection_points)) else "")

        for building in self.city_data["buildings"]:
            if "corners" in building:  # Ensure corners exist in the JSON
                corners = building["corners"]  # Get stored rotated corners
                poly = MplPolygon(corners, closed=True, color="gray", alpha=0.6, label="Building" if building == self.city_data["buildings"][0] else None)
                ax.add_patch(poly)

        # Avoid duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.gca().set_aspect(1.0)

        plt.grid(True)
        plt.show()

# Run the renderer if executed directly
if __name__ == "__main__":
    renderer = CityRenderer("city_data.json")
    renderer.render()
