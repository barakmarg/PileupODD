"""
Distribution of #tracks/event in the pileup-only pool.
Goal: see if any pileup events carry ZERO tracks.

Reads ONE pileup tracks file from HuggingFace (same path the overlay
pipeline uses). Tracks are stored as list-per-event, so #tracks/event is
just the list length.
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
    path = _hf_path(PU_PREFIX, "tracks", FILE_IDX)
    print(f"Reading: {path}")
    with fs.open(path, "rb") as f:
        df = pl.read_parquet(f, columns=["event_id", "track_id"])

    print(f"Rows (events): {df.height}")
    n_tracks = df["track_id"].list.len().to_numpy()

    print("\n--- #tracks / pileup event ---")
    print(f"  events    = {len(n_tracks)}")
    print(f"  min       = {n_tracks.min()}")
    print(f"  max       = {n_tracks.max()}")
    print(f"  mean      = {n_tracks.mean():.2f}")
    print(f"  std       = {n_tracks.std():.2f}")
    print(f"  median    = {np.median(n_tracks):.1f}")
    pcts = [1, 5, 25, 50, 75, 95, 99]
    print("  percentiles: " + ", ".join(
        f"p{p}={np.percentile(n_tracks, p):.0f}" for p in pcts
    ))

    n_zero = int((n_tracks == 0).sum())
    n_le1 = int((n_tracks <= 1).sum())
    n_le5 = int((n_tracks <= 5).sum())
    print(f"\n  events with 0 tracks : {n_zero}  ({100*n_zero/len(n_tracks):.2f}%)")
    print(f"  events with <=1 track: {n_le1}  ({100*n_le1/len(n_tracks):.2f}%)")
    print(f"  events with <=5 track: {n_le5}  ({100*n_le5/len(n_tracks):.2f}%)")

    if n_zero > 0:
        # Show a few example event_ids that have zero tracks
        zero_ids = df.filter(pl.col("track_id").list.len() == 0)["event_id"].head(10).to_list()
        print(f"  sample zero-track event_ids: {zero_ids}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lo = int(n_tracks.min())
    hi = int(n_tracks.max())
    step = max(1, int(np.ceil((hi - lo + 1) / 80)))
    edges = np.arange(lo - 0.5, hi + 0.5 + step, step)

    axes[0].hist(n_tracks, bins=edges, histtype="step", linewidth=1.5,
                 label=f"n={len(n_tracks)}")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7,
                    label=f"zero-track: {n_zero} ev")
    axes[0].set_xlabel("# tracks / pileup event")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"PU tracks/event — {PU_PREFIX} file {FILE_IDX}")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].hist(n_tracks, bins=edges, histtype="step", linewidth=1.5)
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("# tracks / pileup event")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title("same, log y")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = OUT_DIR / f"pu_tracks_per_event_{PU_PREFIX}_file{FILE_IDX}.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
