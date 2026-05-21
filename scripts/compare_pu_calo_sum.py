"""
Compare per-vertex calo-hit energy sums between two pileup-vertex sources:
  (a) Local pool built from ttbar_pu200 via create_pileup_pool_from_pu200.py
  (b) HuggingFace `pileup_only_pu0` (the original pileup-only events)

Take 5k vertices from each, compute sum(total_energy) per vertex, draw two
overlapping histograms for visual comparison.
"""

import sys
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import HfFileSystem
from primary.calibration import CALIBRATION

N_TARGET = 5000
LOCAL_DIR = "/storage/agrp/barakma/PileupODD/data/pileup_from_ttbar_pu200"
HF_PREFIX = "pileup_only_pu0"
HF_BASE = "datasets/CERN/ColliderML-Release-1/data"
NUMBER_OF_HF_REPO_FILES = 1000
OUT_PATH = "/storage/agrp/barakma/PileupODD/scripts/compare_pu_calo_sum.png"


def _sum_per_event(df: pl.DataFrame) -> np.ndarray:
    """Per-event sum of CALIBRATED hit energies (total_energy * calib_factor per detector).
    Mirrors the calibration applied in create_trainning_dataset_pileup.py:639.
    """
    sums = (
        df.lazy()
        .select(['event_id', 'detector', 'total_energy'])
        .explode(['detector', 'total_energy'])
        .join(CALIBRATION.lazy().select(['detector', 'calib_factor']),
              on='detector', how='left')
        .with_columns(
            cal_E=pl.col('total_energy') * pl.col('calib_factor')
        )
        .group_by('event_id', maintain_order=True)
        .agg(pl.col('cal_E').sum().alias('e_sum'))
        .sort('event_id')
        .collect(streaming=True)
    )
    # group_by drops events with empty lists; reattach them with e_sum=0 so the
    # input event order/count is preserved (an invisible vertex contributes 0).
    out = (
        df.select('event_id')
        .join(sums, on='event_id', how='left')
        .with_columns(pl.col('e_sum').fill_null(0.0))
    )
    return out['e_sum'].to_numpy()


def load_local(n: int) -> np.ndarray:
    """Take first n events from local file 0; load more files if needed."""
    sums = []
    i = 0
    while sum(len(s) for s in sums) < n:
        path = f"{LOCAL_DIR}/calo_hits-{i:05d}.parquet"
        df = pl.read_parquet(path, columns=['event_id', 'detector', 'total_energy'])
        sums.append(_sum_per_event(df))
        i += 1
    out = np.concatenate(sums)[:n]
    print(f"local: loaded {len(out)} vertices from {i} file(s)")
    return out


def load_hf(n: int) -> np.ndarray:
    """Stream HF pileup_only_pu0 calo_hits files until we have n events."""
    fs = HfFileSystem()
    sums = []
    i = 0
    while sum(len(s) for s in sums) < n:
        path = (f"{HF_BASE}/{HF_PREFIX}_calo_hits/"
                f"train-{i:05d}-of-{NUMBER_OF_HF_REPO_FILES:05d}.parquet")
        print(f"  reading {path}")
        with fs.open(path, "rb") as f:
            df = pl.read_parquet(f, columns=['event_id', 'detector', 'total_energy'])
        sums.append(_sum_per_event(df))
        i += 1
    out = np.concatenate(sums)[:n]
    print(f"HF:    loaded {len(out)} vertices from {i} file(s)")
    return out


def main():
    local = load_local(N_TARGET)
    hf = load_hf(N_TARGET)

    for name, arr in [("local (ttbar_pu200 split)", local), ("HF pileup_only_pu0", hf)]:
        print(f"{name}: n={len(arr)}, mean={arr.mean():.2f}, median={np.median(arr):.2f}, "
              f"min={arr.min():.2f}, max={arr.max():.2f}, zero-frac={(arr == 0).mean():.3f}")

    # Shared bins on the union range; log scale on y to see the tails.
    lo = 0.0
    hi = float(np.quantile(np.concatenate([local, hf]), 0.995))
    bins = np.linspace(lo, hi, 80)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(local, bins=bins, histtype='step', linewidth=2,
            label=f"local ttbar_pu200 split (n={len(local)})", color='C0')
    ax.hist(hf, bins=bins, histtype='step', linewidth=2,
            label=f"HF pileup_only_pu0 (n={len(hf)})", color='C1')
    ax.set_xlabel("sum of CALIBRATED calo energy per pileup vertex (GeV)")
    ax.set_ylabel("vertices")
    ax.set_yscale('log')
    ax.set_title("Per-vertex calo energy sum: local (ttbar_pu200 split) vs HF pileup_only_pu0")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
