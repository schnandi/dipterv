import pandas as pd

PATH = "burst_training_data.csv"

df = pd.read_csv(PATH)
print("=== BURST DATASET STATISZTIKÁK ===")

# Városok száma
num_towns = df["town_id"].nunique()
print(f"Városok száma: {num_towns}")

# Csövek száma városonként
pipes_per_town = df.groupby("town_id")["pipe_id"].count()

print(f"Átlagos csőszám városonként: {pipes_per_town.mean():.2f}")
print(f"Minimum csőszám egy városban: {pipes_per_town.min()}")
print(f"Maximum csőszám egy városban: {pipes_per_town.max()}")

# Összes cső
print(f"Összes csőszakasz a datasetben: {len(df)}")

# Burst események
num_bursts = df["burst"].sum()
ratio = num_bursts / len(df)

print(f"Burst események száma: {num_bursts}")
print(f"Burst arány: {ratio:.3f}")
