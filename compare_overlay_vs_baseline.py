"""
compare_overlay_vs_baseline.py

Compare synthetic overlay PU200 vs baseline PU200 across key distributions.

Usage:
    python compare_overlay_vs_baseline.py \
        --overlay  data/ttbar_pu0_overlay_pu200 \
        --baseline data/ttbar_pu200 \
        --n_events 1000 \
        --out      comparison_plots.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_parquets(directory: str, prefix: str, n_events: int) -> pl.DataFrame:
    """Concat enough parquet shards to reach n_events rows (one row = one event)."""
    d = Path(directory)
    files = sorted(d.glob(f"{prefix}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {prefix}-*.parquet in {directory}")
    frames, total = [], 0
    for f in files:
        df = pl.read_parquet(f)
        frames.append(df)
        total += len(df)
        if total >= n_events:
            break
    out = pl.concat(frames)
    return out[:n_events]


def load_dataset(directory: str, n_events: int):
    cc = _load_parquets(directory, "calo_clusters", n_events)
    tr = _load_parquets(directory, "tracks", n_events)
    tp = _load_parquets(directory, "target_particles", n_events)
    td = _load_parquets(directory, "target_particles_deps", n_events)
    return cc, tr, tp, td


# ---------------------------------------------------------------------------
# Stat extraction helpers
# ---------------------------------------------------------------------------

def clusters_per_event(cc: pl.DataFrame) -> np.ndarray:
    return cc["cluster_id"].list.len().to_numpy()


def cluster_energies_flat(cc: pl.DataFrame) -> np.ndarray:
    return cc["total_cluster_energy"].explode().drop_nulls().to_numpy()


def track_pt_flat(tr: pl.DataFrame) -> np.ndarray:
    return tr["pt"].explode().drop_nulls().to_numpy()


def particles_per_cluster(td: pl.DataFrame) -> np.ndarray:
    """For each (event, cluster), how many distinct target particles deposited."""
    return (
        td.lazy()
        .select(["event_id", "particle_idx", "cluster_idx"])
        .explode(["particle_idx", "cluster_idx"])
        .group_by(["event_id", "cluster_idx"])
        .agg(pl.col("particle_idx").n_unique().alias("n_particles"))
        .collect()
        ["n_particles"]
        .to_numpy()
    )


def clusters_per_particle(td: pl.DataFrame) -> np.ndarray:
    """For each (event, target particle), how many distinct clusters it deposited in."""
    return (
        td.lazy()
        .select(["event_id", "particle_idx", "cluster_idx"])
        .explode(["particle_idx", "cluster_idx"])
        .group_by(["event_id", "particle_idx"])
        .agg(pl.col("cluster_idx").n_unique().alias("n_clusters"))
        .collect()
        ["n_clusters"]
        .to_numpy()
    )


def target_particles_per_event(tp: pl.DataFrame) -> np.ndarray:
    return tp["particle_id"].list.len().to_numpy()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def compare_hist(ax, a, b, label_a, label_b, title, xlabel, bins=50,
                 log_y=False, log_x=False, clip_quantile=0.999):
    hi = np.quantile(np.concatenate([a, b]), clip_quantile)
    lo = max(0, min(a.min(), b.min()))
    if log_x:
        lo = max(lo, 1e-3)
        bins_arr = np.logspace(np.log10(lo), np.log10(hi + 1e-9), bins + 1)
    else:
        bins_arr = np.linspace(lo, hi, bins + 1)
    kw = dict(bins=bins_arr, density=True, histtype="step", linewidth=1.5)
    ax.hist(a, label=label_a, color="steelblue", **kw)
    ax.hist(b, label=label_b, color="tomato", **kw)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay",  default="data/ttbar_pu0_overlay_pu200")
    parser.add_argument("--baseline", default="data/ttbar_pu200")
    parser.add_argument("--n_events", type=int, default=1000)
    parser.add_argument("--out",      default="comparison_plots.pdf")
    args = parser.parse_args()

    print(f"Loading overlay  ({args.n_events} events) from {args.overlay} ...")
    ov_cc, ov_tr, ov_tp, ov_td = load_dataset(args.overlay,  args.n_events)
    print(f"Loading baseline ({args.n_events} events) from {args.baseline} ...")
    bl_cc, bl_tr, bl_tp, bl_td = load_dataset(args.baseline, args.n_events)

    LA, LB = "Overlay PU200", "Baseline PU200"

    print("Computing statistics ...")
    stats = {
        "clusters_per_event":      (clusters_per_event(ov_cc),      clusters_per_event(bl_cc)),
        "cluster_energy":          (cluster_energies_flat(ov_cc),    cluster_energies_flat(bl_cc)),
        "track_pt":                (track_pt_flat(ov_tr),            track_pt_flat(bl_tr)),
        "particles_per_cluster":   (particles_per_cluster(ov_td),    particles_per_cluster(bl_td)),
        "clusters_per_particle":   (clusters_per_particle(ov_td),    clusters_per_particle(bl_td)),
        "target_particles_per_event": (target_particles_per_event(ov_tp), target_particles_per_event(bl_tp)),
    }

    # Print summary table
    print(f"\n{'Metric':<35} {'Overlay mean':>14} {'Baseline mean':>14}")
    print("-" * 65)
    for k, (a, b) in stats.items():
        print(f"{k:<35} {a.mean():>14.2f} {b.mean():>14.2f}")

    print(f"\nSaving plots to {args.out} ...")
    with PdfPages(args.out) as pdf:
        # Page 1: event-level counts
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Event-level counts  (n={args.n_events} events each)", fontsize=13)

        compare_hist(axes[0], *stats["clusters_per_event"],
                     LA, LB, "Clusters / event", "N clusters", bins=60)
        compare_hist(axes[1], *stats["target_particles_per_event"],
                     LA, LB, "Target particles / event", "N particles", bins=60)
        compare_hist(axes[2], *stats["track_pt"],
                     LA, LB, "Track pT", "pT [GeV]", bins=80, log_y=True, log_x=True,
                     clip_quantile=0.999)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: energy distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Energy distributions", fontsize=13)

        compare_hist(axes[0], *stats["cluster_energy"],
                     LA, LB, "Cluster energy distribution", "Energy [GeV]",
                     bins=80, log_y=True, log_x=True, clip_quantile=0.999)
        # Total cluster energy per event
        ov_tot = ov_cc["total_cluster_energy"].list.sum().to_numpy()
        bl_tot = bl_cc["total_cluster_energy"].list.sum().to_numpy()
        compare_hist(axes[1], ov_tot, bl_tot,
                     LA, LB, "Total cluster energy / event", "Energy [GeV]",
                     bins=60, log_y=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: particle-cluster multiplicity
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Particle–cluster associations (target_particles_deps)", fontsize=13)

        compare_hist(axes[0], *stats["particles_per_cluster"],
                     LA, LB, "Target particles / cluster", "N particles",
                     bins=30, log_y=True)
        compare_hist(axes[1], *stats["clusters_per_particle"],
                     LA, LB, "Clusters / target particle", "N clusters",
                     bins=30, log_y=True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Done. Plots written to {args.out}")


if __name__ == "__main__":
    main()
