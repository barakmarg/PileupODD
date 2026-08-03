"""Subprocess entry point for the equivalence tests.

Runs either the ``master``-branch implementation or this branch's, over the
same cached input frames, and writes the four output tables to a directory for
comparison.

Old and new run in *separate processes* for two reasons: polars' allocator does
not return memory between runs, and writing real parquet files means the
comparison is on what actually lands on disk rather than on in-memory frames.

Usage (invoked by ``test_equivalence.py``, not by hand)::

    python _equiv_worker.py {old|new} {hard_scatter|all_vertices|overlay} \\
        <fixture_dir> <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

#: Checkout holding the original scripts this branch is verified against.
MASTER_ROOT = "/storage/agrp/barakma/PileupODD"

#: Settings shared by both sides, matching the shipped configs. Both sides must
#: use the same backend, since CLUE's output depends on it.
CUTS = dict(truth_eta_cut=3.0, truth_pt_cut=1.0, target_pt_cut=0.3, clusters_cutoff=0.15)
BACKEND = "gpu cuda"
PILEUP_LEVEL = 20
SEED = 42

OUTPUT_KEYS = ("target_particles", "calo_clusters", "tracks", "target_particles_deps")


def _load(fixture_dir: Path, name: str) -> pl.DataFrame:
    return pl.read_parquet(fixture_dir / f"{name}.parquet")


def _force_backend(module) -> None:
    """Pin a master module's clustering backend to :data:`BACKEND`.

    The ``hard_scatter`` and ``all_vertices`` scripts call ``clue_clustering``
    without a ``backend`` argument, so they take its default. Wrapping the
    module-level reference pins both sides to the same backend explicitly.
    """
    original = module.clue_clustering

    def patched(calo_hits, **kwargs):
        kwargs["backend"] = BACKEND
        return original(calo_hits, **kwargs)

    module.clue_clustering = patched


def _patch_master_sample_map(module) -> None:
    """Give master's pileup sampler the determinism fix this branch has.

    Master draws its pileup sample by walking the pool array in whatever order
    polars produced it, and that order is not stable between runs -- so master's
    overlay picks a *different* pileup sample on every run, seed regardless.
    Comparing overlay output against it would compare two different physics
    samples, not two implementations.

    Sorting both id arrays -- exactly what
    :func:`colliderml_pflow.overlay.build_sample_map` now does -- makes the two
    sides draw the same sample, so any remaining difference is attributable to
    the overlay and aggregation code, which is what the test is for.

    The branch's own sampling reproducibility is covered separately by
    ``test_overlay.py``.
    """
    import numpy as np
    original = module._build_sample_map

    def patched(hs_event_ids, pu_event_ids, pileup_level, seed, invisible_pu_prob=0.0):
        return original(np.sort(np.asarray(hs_event_ids)),
                        np.sort(np.asarray(pu_event_ids)),
                        pileup_level, seed, invisible_pu_prob=invisible_pu_prob)

    module._build_sample_map = patched


def run_old(mode: str, fixture_dir: Path) -> dict:
    sys.path.insert(0, MASTER_ROOT)

    if mode == "overlay":
        import primary.create_training_dataset_pileup_overlay as overlay_mod
        _patch_master_sample_map(overlay_mod)
        preprocess_for_model = overlay_mod.preprocess_for_model
        return preprocess_for_model(
            hs_particles=_load(fixture_dir, "hs_particles"),
            hs_tracks=_load(fixture_dir, "hs_tracks"),
            hs_calo_hits=_load(fixture_dir, "hs_calo_hits"),
            pu_particles=_load(fixture_dir, "pu_particles"),
            pu_tracks=_load(fixture_dir, "pu_tracks"),
            pu_calo_hits=_load(fixture_dir, "pu_calo_hits"),
            pileup_level=PILEUP_LEVEL,
            seed=SEED,
            num_of_events=-1,
            clue_backend=BACKEND,
            chunk_size=-1,
            invisible_pu_prob=0.0,
            tof_enabled=True,
            **CUTS,
        )

    if mode == "hard_scatter":
        import primary.create_trainning_dataset_pileup as old_mod
    elif mode == "all_vertices":
        import primary.create_trainning_dataset_pileup_research as old_mod
    else:
        raise ValueError(f"unknown mode {mode!r}")

    _force_backend(old_mod)
    return old_mod.preprocess_for_model(
        particles=_load(fixture_dir, "particles"),
        tracks=_load(fixture_dir, "tracks"),
        calo_hits=_load(fixture_dir, "calo_hits"),
        num_of_events=-1,
        **CUTS,
    )


def run_new(mode: str, fixture_dir: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from colliderml_pflow.config import ClusteringConfig, Cuts, OverlayConfig
    from colliderml_pflow.pipeline import preprocess_events

    cuts = Cuts(truth_pt=CUTS["truth_pt_cut"], truth_eta=CUTS["truth_eta_cut"],
                target_pt=CUTS["target_pt_cut"], cluster_energy=CUTS["clusters_cutoff"])
    clustering = ClusteringConfig(backend=BACKEND)

    if mode == "overlay":
        return preprocess_events(
            "overlay",
            particles=_load(fixture_dir, "hs_particles"),
            tracks=_load(fixture_dir, "hs_tracks"),
            calo_hits=_load(fixture_dir, "hs_calo_hits"),
            pu_particles=_load(fixture_dir, "pu_particles"),
            pu_tracks=_load(fixture_dir, "pu_tracks"),
            pu_calo_hits=_load(fixture_dir, "pu_calo_hits"),
            cuts=cuts,
            clustering=clustering,
            keep_all_vertices=False,
            overlay_cfg=OverlayConfig(pileup_level=PILEUP_LEVEL, seed=SEED,
                                      invisible_pu_prob=0.0),
            seed=SEED,
        )

    # The cached ttbar_pu200 hits carry contrib_times because master needs them
    # for cluster_time. This branch drops cluster_time, so its loader
    # (hf_io.CALO_COLS) never requests the times -- drop the column here so the
    # new side sees exactly the frame production would hand it.
    calo_hits = _load(fixture_dir, "calo_hits")
    if "contrib_times" in calo_hits.columns:
        calo_hits = calo_hits.drop("contrib_times")

    return preprocess_events(
        mode,
        particles=_load(fixture_dir, "particles"),
        tracks=_load(fixture_dir, "tracks"),
        calo_hits=calo_hits,
        cuts=cuts,
        clustering=clustering,
        keep_all_vertices=(mode == "all_vertices"),
    )


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    side, mode, fixture_dir, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])

    out = run_old(mode, fixture_dir) if side == "old" else run_new(mode, fixture_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for key in OUTPUT_KEYS:
        out[key].write_parquet(out_dir / f"{key}.parquet")
    print(f"[{side}/{mode}] wrote {len(OUTPUT_KEYS)} tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
