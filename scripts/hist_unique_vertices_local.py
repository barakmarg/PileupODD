import polars as pl
import numpy as np
import matplotlib.pyplot as plt

FILE = "/storage/agrp/barakma/PileupODD/data/ttbar_pu200_all_vertices_chunked/target_particles-00046.parquet"
OUT_PATH = "/storage/agrp/barakma/PileupODD/scripts/hist_unique_vertices_local.png"

df = pl.read_parquet(FILE, columns=["event_id", "vertex_primary"])
print(f"Events: {df.height}")

counts = df.with_columns(
    pl.col("vertex_primary").list.n_unique().alias("n_vertices")
)["n_vertices"].to_numpy()

print(f"min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}, median={np.median(counts):.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(counts, bins=40, edgecolor="black")
ax.set_xlabel("Number of unique primary vertices per event")
ax.set_ylabel("Number of events")
ax.set_title(f"ttbar_pu200 (local chunked): unique primary vertices per event\n({df.height} events, file 00046)")
ax.axvline(counts.mean(), color="red", linestyle="--", label=f"mean={counts.mean():.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=120)
print(f"Saved: {OUT_PATH}")
