import random
import pandas as pd
from pathlib import Path

from app.core.generator.city_generator import CityGenerator
from core.ai.burst_data_generator import analyze_simulation

OUTPUT_PATH = Path("burst_training_data.csv")


def generate_town(seed=None):
    """
    Same logic as used in TownController to generate a fresh city_data dict.
    """
    if seed is None:
        seed = random.randint(0, 2**12 - 1)

    generator = CityGenerator(map_size=(2000, 2000), seed=seed)
    city_data = generator.generate()
    return city_data, seed


def main(n_towns: int = 10):
    """
    Generate a full synthetic burst dataset across multiple towns.
    """
    all_dfs = []

    for i in range(n_towns):
        print(f"🏙️ Generating and simulating town {i+1}/{n_towns}...")

        town_data, seed = generate_town()
        df = analyze_simulation(town_data, town_id=seed)
        all_dfs.append(df)

    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Saved dataset to {OUTPUT_PATH.resolve()}")
    print(f"📊 Total rows: {len(dataset)}, burst ratio: {dataset['burst'].mean():.3f}")


if __name__ == "__main__":
    main(n_towns=30)
