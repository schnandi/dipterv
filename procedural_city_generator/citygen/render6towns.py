import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import json
from perlin_noise import PerlinNoise   # <-- import your generator file

from citygen.generator import CityGenerator

# -------------------------------------------
# Configuration for the renderer
# -------------------------------------------
HEIGHT_MAP_SCALE = 1000.0
HEIGHT_MAP_OCTAVES = 6
HEIGHT_MAP_AMPLITUDE = 100
HEIGHT_CURVE_EXPONENT = 1.5
HEIGHT_MAP_RES = 150     # Lower resolution for faster multi-plot rendering

# -------------------------------------------
# Render ONE town into a given Axes
# -------------------------------------------
def render_town(ax, city_data):
    seed = city_data["seed"]

    # Compute bounds
    xs, ys = [], []
    for r in city_data["roads"]:
        xs += [r["start"][0], r["end"][0]]
        ys += [r["start"][1], r["end"][1]]
    for b in city_data["buildings"]:
        for x, y in b["corners"]:
            xs.append(x)
            ys.append(y)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad = 200
    min_x -= pad; max_x += pad
    min_y -= pad; max_y += pad

    # Generate terrain background
    width = max_x - min_x
    height = max_y - min_y
    terrain = np.zeros((HEIGHT_MAP_RES, HEIGHT_MAP_RES))

    for j in range(HEIGHT_MAP_RES):
        for i in range(HEIGHT_MAP_RES):
            x = min_x + (i / (HEIGHT_MAP_RES - 1)) * width
            y = min_y + (j / (HEIGHT_MAP_RES - 1)) * height


    # Draw roads
    for r in city_data["roads"]:
        sx, sy = r["start"]
        ex, ey = r["end"]
        t = r.get("pipe_type", "side")

        if t == "main":
            style, lw, color = "-", 2.5, "green"
        elif t == "side":
            style, lw, color = "-", 1.2, "red"
        elif t == "building connection":
            style, lw, color = "--", 1.0, "blue"
        else:
            style, lw, color = "-", 1.0, "black"

        ax.plot([sx, ex], [sy, ey], style, lw=lw, color=color)

    # Draw buildings
    for b in city_data["buildings"]:
        poly = MplPolygon(
            b["corners"],
            closed=True,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=1,
            alpha=0.8
        )
        ax.add_patch(poly)

    # Title with seed
    ax.set_title(f"Seed = {seed}", fontsize=12)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# -------------------------------------------
# Generate + render 6 different towns
# -------------------------------------------
def render_six_towns():
    seeds = [1111, 2222, 3333, 4444, 5555, 6666]   # You can change these
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for seed, ax in zip(seeds, axes.flatten()):
        generator = CityGenerator(map_size=(2000, 2000), seed=seed)
        generator.generate()

        # Convert to dict format for renderer
        city_data = {
            "seed": generator.seed,
            "roads": [r.to_dict() for r in generator.road_network],
            "buildings": [
                {
                    **b.to_dict(),
                    "district": getattr(b, "district", "undefined")
                }
                for b in generator.buildings
            ]
        }

        render_town(ax, city_data)

    plt.tight_layout()
    plt.savefig("six_towns.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    render_six_towns()
