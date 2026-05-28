"""
Compare cluster / track / particle statistics between two datasets:

  A) ttbar_pu0_overlay_pu200  (1 file, 1000 events)
  B) ttbar_pu200              (10 files x 100 events = 1000 events)

Histograms (overlaid A vs B):
  1. # clusters per event
  2. cluster energy distribution (all clusters from all events)
  3. track energy distribution (E = pt * cosh(eta))
  4. # particles per cluster      (target-particle deposits)
  5. # clusters per particle      (target-particle deposits)
  6. sum cluster energy per event
  7. # tracks per event

Author note: counts in (4)/(5) are derived from `target_particles_deps`
(cluster_idx, particle_idx) WITHIN each event. They are per-event positional
counts, so no cross-run join on indices is performed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

DATA_ROOT = Path("/storage/agrp/barakma/PileupODD/data")
DIR_A = DATA_ROOT / "ttbar_pu0_overlay_pu200"   # 1 file, 1000 events
DIR_B = DATA_ROOT / "ttbar_pu200"               # many files, 100 ev each

N_FILES_A = 1     # read all
N_FILES_B = 10    # 10 * 100 = 1000 events

OUT_DIR = Path("/storage/agrp/barakma/PileupODD/scripts/out_pu200tosfilter_overlay_vs_pu200")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------- IO ---------------------------------------------------------------

def _files(dir_: Path, stem: str, n: int) -> list[Path]:
    files = sorted(dir_.glob(f"{stem}-*.parquet"))[:n]
    if not files:
        raise FileNotFoundError(f"no {stem}-*.parquet in {dir_}")
    return files


def load(dir_: Path, n_files: int) -> dict[str, pl.DataFrame]:
    """Read calo_clusters / tracks / target_particles_deps for one dataset."""
    out: dict[str, pl.DataFrame] = {}
    for stem in ("calo_clusters", "tracks", "target_particles_deps"):
        paths = _files(dir_, stem, n_files)
        out[stem] = pl.concat([pl.read_parquet(p) for p in paths], how="vertical")
    return out


# -------- per-stat extractors (return 1-D numpy) ---------------------------

def n_clusters_per_event(calo: pl.DataFrame) -> np.ndarray:
    return calo["cluster_id"].list.len().to_numpy()


def cluster_energy(calo: pl.DataFrame) -> np.ndarray:
    return calo["total_cluster_energy"].explode().drop_nulls().to_numpy()


def sum_cluster_energy_per_event(calo: pl.DataFrame) -> np.ndarray:
    return calo["total_cluster_energy"].list.sum().to_numpy()


def n_tracks_per_event(tracks: pl.DataFrame) -> np.ndarray:
    """One row per event; pt is a per-track list — its length is # tracks."""
    return tracks["pt"].list.len().to_numpy()


def track_energy(tracks: pl.DataFrame) -> np.ndarray:
    """E ~= pt * cosh(eta), massless approximation."""
    flat = tracks.select(["pt", "eta"]).explode(["pt", "eta"]).drop_nulls()
    pt = flat["pt"].to_numpy().astype(np.float64)
    eta = flat["eta"].to_numpy().astype(np.float64)
    return pt * np.cosh(eta)


def particles_per_cluster(deps: pl.DataFrame) -> np.ndarray:
    """For each (event, cluster_idx), count distinct particle_idx."""
    flat = (
        deps.lazy()
        .select(["event_id", "particle_idx", "cluster_idx"])
        .explode(["particle_idx", "cluster_idx"])
        .group_by(["event_id", "cluster_idx"])
        .agg(pl.col("particle_idx").n_unique().alias("n_particles"))
        .collect()
    )
    return flat["n_particles"].to_numpy()


def clusters_per_particle(deps: pl.DataFrame) -> np.ndarray:
    """For each (event, particle_idx), count distinct cluster_idx."""
    flat = (
        deps.lazy()
        .select(["event_id", "particle_idx", "cluster_idx"])
        .explode(["particle_idx", "cluster_idx"])
        .group_by(["event_id", "particle_idx"])
        .agg(pl.col("cluster_idx").n_unique().alias("n_clusters"))
        .collect()
    )
    return flat["n_clusters"].to_numpy()


# -------- plotting ---------------------------------------------------------

def _print_summary(name: str, a: np.ndarray, b: np.ndarray) -> None:
    def s(x):
        return f"n={len(x):>7d}  mean={np.mean(x):8.3f}  median={np.median(x):8.3f}  p99={np.percentile(x, 99):8.3f}  max={np.max(x):8.3f}"
    print(f"--- {name} ---")
    print(f"  A (pu0_overlay_pu200): {s(a)}")
    print(f"  B (pu200            ): {s(b)}")


def overlay_hist(
    ax, a: np.ndarray, b: np.ndarray, *, bins, log_y: bool = False,
    log_x: bool = False, xlabel: str, title: str, integer: bool = False,
):
    if log_x:
        a = a[a > 0]
        b = b[b > 0]
        edges = np.logspace(np.log10(min(a.min(), b.min())),
                            np.log10(max(a.max(), b.max())), bins)
    elif integer:
        lo = int(min(a.min(), b.min()))
        hi = int(max(a.max(), b.max()))
        step = max(1, int(np.ceil((hi - lo + 1) / bins)))
        edges = np.arange(lo - 0.5, hi + 0.5 + step, step)
    else:
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        edges = np.linspace(lo, hi, bins)

    ax.hist(a, bins=edges, histtype="step", linewidth=1.6,
            label=f"pu0_overlay_pu200 (n={len(a)})", density=True)
    ax.hist(b, bins=edges, histtype="step", linewidth=1.6,
            label=f"pu200 (n={len(b)})", density=True)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main() -> None:
    print(f"Reading A: {DIR_A}  ({N_FILES_A} file)")
    A = load(DIR_A, N_FILES_A)
    print(f"Reading B: {DIR_B}  ({N_FILES_B} files)")
    B = load(DIR_B, N_FILES_B)

    print(f"  A events: calo={len(A['calo_clusters'])}, "
          f"tracks={len(A['tracks'])}, deps={len(A['target_particles_deps'])}")
    print(f"  B events: calo={len(B['calo_clusters'])}, "
          f"tracks={len(B['tracks'])}, deps={len(B['target_particles_deps'])}")

    stats = {
        "n_clusters_per_event": (
            n_clusters_per_event(A["calo_clusters"]),
            n_clusters_per_event(B["calo_clusters"]),
        ),
        "cluster_energy_gev": (
            cluster_energy(A["calo_clusters"]),
            cluster_energy(B["calo_clusters"]),
        ),
        "track_energy_gev": (
            track_energy(A["tracks"]),
            track_energy(B["tracks"]),
        ),
        "particles_per_cluster": (
            particles_per_cluster(A["target_particles_deps"]),
            particles_per_cluster(B["target_particles_deps"]),
        ),
        "clusters_per_particle": (
            clusters_per_particle(A["target_particles_deps"]),
            clusters_per_particle(B["target_particles_deps"]),
        ),
        "sum_cluster_energy_per_event_gev": (
            sum_cluster_energy_per_event(A["calo_clusters"]),
            sum_cluster_energy_per_event(B["calo_clusters"]),
        ),
        "n_tracks_per_event": (
            n_tracks_per_event(A["tracks"]),
            n_tracks_per_event(B["tracks"]),
        ),
    }

    for name, (a, b) in stats.items():
        _print_summary(name, a, b)

    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    axes = axes.ravel()

    overlay_hist(
        axes[0], *stats["n_clusters_per_event"],
        bins=60, integer=True, log_y=True,
        xlabel="# clusters / event",
        title="Clusters per event",
    )
    overlay_hist(
        axes[1], *stats["cluster_energy_gev"],
        bins=80, log_x=True, log_y=True,
        xlabel="cluster energy [GeV]",
        title="Cluster energy distribution",
    )
    overlay_hist(
        axes[2], *stats["track_energy_gev"],
        bins=80, log_x=True, log_y=True,
        xlabel="track energy E = pt*cosh(eta) [GeV]",
        title="Track energy distribution",
    )
    overlay_hist(
        axes[3], *stats["particles_per_cluster"],
        bins=60, integer=True, log_y=True,
        xlabel="# target particles depositing in cluster",
        title="# particles per cluster",
    )
    overlay_hist(
        axes[4], *stats["clusters_per_particle"],
        bins=30, integer=True, log_y=True,
        xlabel="# clusters touched by target particle",
        title="# clusters per particle",
    )
    overlay_hist(
        axes[5], *stats["sum_cluster_energy_per_event_gev"],
        bins=60, log_y=True,
        xlabel="sum cluster energy / event [GeV]",
        title="Sum cluster energy per event",
    )
    overlay_hist(
        axes[6], *stats["n_tracks_per_event"],
        bins=60, integer=True, log_y=True,
        xlabel="# tracks / event",
        title="Tracks per event",
    )
    # Hide the unused 8th axis in the 4x2 grid.
    axes[7].set_axis_off()

    fig.suptitle("ttbar_pu0_overlay_pu200  vs  ttbar_pu200   (1000 events each)",
                 fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "compare_pu0overlay_vs_pu200.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
