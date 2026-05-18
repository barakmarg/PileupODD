"""
Step 1 of the rejection-sampling plan: inspect the primary-vertex Z (Vz)
distribution of the pileup-only pool.

Reads ONE pileup file from HuggingFace (same path the overlay pipeline uses)
and reports:
  - what column carries Vz (per-particle vs per-event)
  - basic stats (min/max/mean/std)
  - whether it looks Gaussian-smeared, uniform, or fixed at 0
  - histogram saved as PNG

If Vz is per-particle: we take ONE Vz per (event_id, primary_vertex) pair,
i.e. the vertex location, not one per particle (otherwise high-multiplicity
events would dominate the histogram).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from huggingface_hub import HfFileSystem

PU_PREFIX = "pileup_only_pu0"
FILE_IDX = 0
N_FILES_TOTAL = 1000

OUT_DIR = Path("/storage/agrp/barakma/PileupODD/scripts/out_pu0overlay_vs_pu200")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _hf_path(prefix: str, kind: str, idx: int) -> str:
    return (
        f"datasets/CERN/ColliderML-Release-1/data/"
        f"{prefix}_{kind}/train-{idx:05d}-of-{N_FILES_TOTAL:05d}.parquet"
    )


def main() -> None:
    fs = HfFileSystem()
    path = _hf_path(PU_PREFIX, "particles", FILE_IDX)
    print(f"Reading: {path}")
    with fs.open(path, "rb") as f:
        df = pl.read_parquet(f, columns=["event_id", "vertex_primary", "vz"])

    print(f"Rows (events): {df.height}")
    print(f"Schema: {dict(df.schema)}")

    # Per-vertex unique Vz: explode then group by (event_id, vertex_primary)
    # and take first vz — all particles from the same primary vertex share Vz.
    per_vertex = (
        df.lazy()
        .explode(["vertex_primary", "vz"])
        .group_by(["event_id", "vertex_primary"])
        .agg(pl.col("vz").first().alias("vz"))
        .collect()
    )
    print(f"Unique (event, primary_vertex) rows: {per_vertex.height}")
    n_vtx_per_event = (
        per_vertex.group_by("event_id").len().rename({"len": "n_vtx"})
    )
    print(f"Vertices per event: "
          f"min={n_vtx_per_event['n_vtx'].min()}, "
          f"max={n_vtx_per_event['n_vtx'].max()}, "
          f"mean={n_vtx_per_event['n_vtx'].mean():.2f}")

    vz = per_vertex["vz"].to_numpy().astype(np.float64)
    vz = vz[np.isfinite(vz)]

    print("\n--- Vz (per primary vertex) ---")
    print(f"  n         = {len(vz)}")
    print(f"  min       = {vz.min():.4f}")
    print(f"  max       = {vz.max():.4f}")
    print(f"  mean      = {vz.mean():.4f}")
    print(f"  std       = {vz.std():.4f}")
    print(f"  median    = {np.median(vz):.4f}")
    pcts = [1, 5, 25, 50, 75, 95, 99]
    print("  percentiles: " + ", ".join(
        f"p{p}={np.percentile(vz, p):.2f}" for p in pcts
    ))
    print(f"  fraction within +/- 1mm of 0: {(np.abs(vz) < 1).mean():.4f}")
    print(f"  fraction within +/- 55.5mm of 0: {(np.abs(vz) < 55.5).mean():.4f}")
    print(f"  unique values (sample): {np.unique(np.round(vz, 4))[:20]}")
    print(f"  n unique (rounded 4dp): {len(np.unique(np.round(vz, 4)))}")

    # Quick classification hint:
    if vz.std() < 1e-3:
        verdict = "FIXED at ~0 — would need manual hit-coordinate shift."
    elif abs(vz.mean()) < 5 and 30 < vz.std() < 80:
        verdict = "Looks Gaussian-smeared near target (sigma ~ 55mm)."
    elif abs(vz.mean()) < 5 and vz.std() > 80:
        verdict = "Smeared but wider than target — rejection sampling works."
    else:
        verdict = "Non-trivial distribution — inspect histogram."
    print(f"\n  verdict: {verdict}")

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(vz.min(), vz.max(), 100)

    axes[0].hist(vz, bins=bins, histtype="step", linewidth=1.5,
                 label=f"Vz (n={len(vz)})")
    # overlay a Gaussian sigma=55.5 normalised to same area for visual reference
    x = np.linspace(vz.min(), vz.max(), 400)
    sigma = 55.5
    g = np.exp(-x**2 / (2 * sigma**2))
    g /= g.sum() * (x[1] - x[0])
    counts, _ = np.histogram(vz, bins=bins)
    g *= counts.sum() * (bins[1] - bins[0])
    axes[0].plot(x, g, linestyle="--", label=r"Gaussian $\sigma=55.5$mm (target)")
    axes[0].set_xlabel("primary vertex Vz [mm]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"PU Vz distribution — {PU_PREFIX} file {FILE_IDX}")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].hist(vz, bins=bins, histtype="step", linewidth=1.5)
    axes[1].plot(x, g, linestyle="--", label=r"Gaussian $\sigma=55.5$mm")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("primary vertex Vz [mm]")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title("same, log y")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    fig.tight_layout()
    out = OUT_DIR / f"pu_vz_distribution_{PU_PREFIX}_file{FILE_IDX}.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
