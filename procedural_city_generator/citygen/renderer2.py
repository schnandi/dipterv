import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import json
from perlin_noise import PerlinNoise

# -------------------------- CONFIGURATION --------------------------
MAP_WIDTH = 2000
MAP_HEIGHT = 2000
HEIGHT_MAP_SCALE = 1000.0
HEIGHT_MAP_OCTAVES = 6
HEIGHT_MAP_AMPLITUDE = 100
HEIGHT_MAP_RES = 300
HEIGHT_CURVE_EXPONENT = 1.5
# --------------------------------------------------------------------

def _compute_data_bounds(city_data):
    xs = []
    ys = []

    for r in city_data["roads"]:
        xs += [r["start"][0], r["end"][0]]
        ys += [r["start"][1], r["end"][1]]

    for b in city_data.get("buildings", []):
        if "corners" in b:
            for x, y in b["corners"]:
                xs.append(x)
                ys.append(y)

    return min(xs), max(xs), min(ys), max(ys)


class ZoneRenderer:
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.city_data = json.load(f)
            self.seed = self.city_data.get("seed", 0)

        # Same terrain generator as CityGenerator
        self.noise = PerlinNoise(
            octaves=HEIGHT_MAP_OCTAVES,
            seed=self.seed
        )

        self.district_colors = {
            "residential": "lightblue",
            "industrial": "lightcoral",
            "undefined": "gray"
        }

    def _height_at(self, x, y):
        n = self.noise([x / HEIGHT_MAP_SCALE, y / HEIGHT_MAP_SCALE])
        h = (n + 1) * 0.5     # normalize 0..1
        h = h ** HEIGHT_CURVE_EXPONENT
        return h * HEIGHT_MAP_AMPLITUDE

    def _generate_height_map_for_extent(self, min_x, max_x, min_y, max_y, res):
        width = max_x - min_x
        height = max_y - min_y
        terrain = np.zeros((res, res))

        for j in range(res):
            for i in range(res):
                x = min_x + (i / (res - 1)) * width
                y = min_y + (j / (res - 1)) * height
                terrain[j, i] = self._height_at(x, y)

        return terrain

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Determine bounds
        min_x, max_x, min_y, max_y = _compute_data_bounds(self.city_data)
        pad = 200
        min_x -= pad
        max_x += pad
        min_y -= pad
        max_y += pad

        # Terrain layer
        height_map = self._generate_height_map_for_extent(
            min_x, max_x, min_y, max_y, HEIGHT_MAP_RES
        )

        ax.imshow(
            height_map,
            origin="lower",
            extent=[min_x, max_x, min_y, max_y],
            cmap="terrain",
            alpha=0.55,
            interpolation="bilinear"
        )

        # ---- Draw all roads ----
        labels_done = set()
        intersections = set()

        for r in self.city_data["roads"]:
            s = r["start"]
            e = r["end"]
            t = r.get("pipe_type", "side")

            if t == "main":
                style, lw, lbl = "g-", 3, "Main Road"
            elif t == "side":
                style, lw, lbl = "r-", 1, "Side Road"
            elif t == "building connection":
                style, lw, lbl = "b--", 1, "Building Connection"
            else:
                style, lw, lbl = "k-", 1, "Unknown Road"

            if lbl not in labels_done:
                ax.plot([s[0], e[0]], [s[1], e[1]], style, lw=lw, label=lbl)
                labels_done.add(lbl)
            else:
                ax.plot([s[0], e[0]], [s[1], e[1]], style, lw=lw)

            intersections.add(tuple(s))
            intersections.add(tuple(e))

        # ---- Draw intersections ----
        if "Intersection" not in labels_done:
            ax.scatter(
                [p[0] for p in intersections],
                [p[1] for p in intersections],
                s=10, c="blue", label="Intersection"
            )
            labels_done.add("Intersection")
        else:
            ax.scatter(
                [p[0] for p in intersections],
                [p[1] for p in intersections],
                s=10, c="blue"
            )

        # ---- Draw buildings ----
        for b in self.city_data.get("buildings", []):
            if "corners" not in b:
                continue

            district = b.get("district", "undefined")
            color = self.district_colors.get(district, "gray")

            label = None
            display_name = district.capitalize()
            if display_name not in labels_done:
                label = display_name
                labels_done.add(display_name)

            # Buildings now have:
            #  ✔ No large dots
            #  ✔ A crisp 1px black outline
            poly = MplPolygon(
                b["corners"],
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                alpha=0.75,
                label=label
            )
            ax.add_patch(poly)

        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    ZoneRenderer("city_data.json").render()
