import numpy as np
import random

class DistrictAssigner:
    """Handles assignment of districts and building types."""

    def __init__(self, map_size):
        self.map_size = map_size

    def assign(self, buildings):
        if not buildings:
            return

        total = len(buildings)
        industrial_center = np.array([self.map_size[0] * 0.9, self.map_size[1] * 0.9])

        centers = np.array([b.center for b in buildings])
        distances = np.linalg.norm(centers - industrial_center, axis=1)
        sorted_indices = np.argsort(distances)
        num_industrial = int(total * 0.2)
        industrial_indices = set(sorted_indices[:num_industrial])

        for i, b in enumerate(buildings):
            if i in industrial_indices:
                b.district = "industrial"
                b.building_type = np.random.choice(
                    ["factory", "warehouse", "processing_plant"], p=[0.4, 0.4, 0.2]
                )
            else:
                b.district = "residential"
                b.building_type = np.random.choice(
                    ["single_family", "apartment", "restaurant", "office"], p=[0.5, 0.3, 0.1, 0.1]
                )

        residential = [b for b in buildings if b.district == "residential"]
        if len(residential) >= 3:
            civic_types = ["hospital", "library", "school"]
            selected = random.sample(residential, len(civic_types))
            for b, civic in zip(selected, civic_types):
                b.building_type = civic
