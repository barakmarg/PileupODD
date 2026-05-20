from huggingface_hub import HfFileSystem
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

EVENT_NAME = "ttbar_pu200"
FILE_INDEX = 0
NUMBER_OF_HF_REPO_FILES = 1000
OUT_PATH = "/storage/agrp/barakma/PileupODD/scripts/hist_unique_vertices.png"

fs = HfFileSystem()
file_path = (
    f"datasets/CERN/ColliderML-Release-1/data/{EVENT_NAME}_particles/"
    f"train-{FILE_INDEX:05d}-of-{NUMBER_OF_HF_REPO_FILES:05d}.parquet"
)
print(f"Reading: {file_path}")
with fs.open(file_path, "rb") as f:
    particles = pl.read_parquet(f, columns=["event_id", "vertex_primary"])

print(f"Rows (events): {particles.height}")

# each row is one event; vertex_primary is a list of per-particle vertex indices
counts = (
    particles
    .with_columns(
        pl.col("vertex_primary").list.n_unique().alias("n_vertices")
    )
    ["n_vertices"]
    .to_numpy()
)

print(f"min={counts.min()}, max={counts.max()}, "
      f"mean={counts.mean():.1f}, median={np.median(counts):.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(counts, bins=40, edgecolor="black")
ax.set_xlabel("Number of unique primary vertices per event")
ax.set_ylabel("Number of events")
ax.set_title(f"{EVENT_NAME}: unique primary vertices per event\n({len(counts)} events, file {FILE_INDEX})")
ax.axvline(counts.mean(), color="red", linestyle="--", label=f"mean={counts.mean():.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=120)
print(f"\nSaved: {OUT_PATH}")
