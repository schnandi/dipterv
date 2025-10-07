import numpy as np
from perlin_noise import PerlinNoise
from .config import CONFIG


class TerrainGenerator:
    def __init__(self, seed):
        self.noise = PerlinNoise(octaves=CONFIG["height_octaves"], seed=seed)

    def get_height(self, x, y):
        val = self.noise([x / CONFIG["height_noise_scale"], y / CONFIG["height_noise_scale"]])
        h = (val + 1) * 0.5
        h = h ** CONFIG["height_curve_exponent"]
        return h * CONFIG["height_amplitude"]

    def generate_height_map(self, bounds):
        min_x, min_y, max_x, max_y = bounds
        res_x, res_y = CONFIG["height_map_resolution"]

        xs = np.linspace(min_x, max_x, res_x)
        ys = np.linspace(min_y, max_y, res_y)

        grid = [[self.get_height(x, y) for x in xs] for y in ys]
        return grid
