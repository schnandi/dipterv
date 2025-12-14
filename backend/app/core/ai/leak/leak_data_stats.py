import pandas as pd

PATH = "leak_training_data.csv"

df = pd.read_csv(PATH)
print("=== LEAK DATASET STATISZTIKÁK ===")

# Leak szimulációk száma
num_leaks = df["leak_pipe_id"].nunique()
print(f"Szivárgási események száma: {num_leaks}")

# Csövek száma szimulációnként
pipes_per_sim = df.groupby("leak_pipe_id")["pipe_id"].count()

print(f"Átlagos csőszám szimulációnként: {pipes_per_sim.mean():.2f}")
print(f"Minimum csőszám: {pipes_per_sim.min()}")
print(f"Maximum csőszám: {pipes_per_sim.max()}")

# Összes cső a datasetben
print(f"Összes sor (csőszakasz): {len(df)}")

# Pozitív példák (ahol a cső volt a szivárgási cső)
num_positive = (df["pipe_id"] == df["leak_pipe_id"]).sum()
print(f"Pozitív példák száma: {num_positive}")
print(f"Pozitív arány: {num_positive / len(df):.5f}")
