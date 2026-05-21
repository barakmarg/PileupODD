"""
Overlay synthetic PU<pileup_level> onto ttbar_pu0 hard-scatter events from
HuggingFace, sampling pileup events from the LOCAL pool built by
`create_pileup_pool_from_pu200.py` (one event per PU vertex of ttbar_pu200).

Wraps `preprocess_for_model` from create_training_dataset_pileup_overlay.py
and processes HS events in chunks (default 334) to bound peak RAM.

Run:
    python run_overlay_from_local_pu.py --hs-indices 0 --pu-indices 0
"""

import argparse
import gc
import sys
import time
from pathlib import Path

# Make the `primary` package importable (matches the convention in submit_preprocess_overlay.py)
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')

import polars as pl
import tqdm
from huggingface_hub import HfFileSystem

from primary.create_training_dataset_pileup_overlay import preprocess_for_model

PU_DIR = Path("/storage/agrp/barakma/PileupODD/data/pileup_from_ttbar_pu200")
HS_EVENT_NAME = "ttbar_pu0"
PILEUP_LEVEL = 200
NUMBER_OF_HF_REPO_FILES = 1000
OUT_DIR = Path(
    f"/storage/agrp/barakma/PileupODD/data/{HS_EVENT_NAME}_overlay_pu{PILEUP_LEVEL}_from_ttbar"
)

# Match the column lists used by run_preprocessing_pipeline in the overlay file
# so per-source preprocessing sees the expected schema. Notably: no contrib_times,
# no orig_event_id (we drop it from local files at load time).
PARTICLE_COLS = [
    'event_id', 'particle_id', 'vertex_primary', 'pdg_id',
    'energy', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'parent_id',
]
HS_CALO_COLS = [
    'event_id', 'detector', 'total_energy', 'x', 'y', 'z',
    'contrib_particle_ids', 'contrib_energies',
]
# PU adds contrib_times for the ToF hit-time precompute in _preprocess_source.
PU_CALO_COLS = HS_CALO_COLS + ['contrib_times']


def _load_local_pu(pu_file_indices: list[int]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load local PU files and concat with event_id offsets so ids are unique across files.
    Mirrors `_load_pu_batch` in create_training_dataset_pileup_overlay.py:1742.
    """
    p_list, c_list, t_list = [], [], []
    offset = 0
    for idx in pu_file_indices:
        p_path = PU_DIR / f"particles-{idx:05d}.parquet"
        c_path = PU_DIR / f"calo_hits-{idx:05d}.parquet"
        t_path = PU_DIR / f"tracks-{idx:05d}.parquet"
        for fp in (p_path, c_path, t_path):
            if not fp.exists():
                raise FileNotFoundError(fp)

        p = pl.read_parquet(p_path, columns=PARTICLE_COLS)
        c = pl.read_parquet(c_path, columns=PU_CALO_COLS)
        t = pl.read_parquet(t_path)
        if 'orig_event_id' in t.columns:
            t = t.drop('orig_event_id')

        # Drop trackless events from the tracks frame only. Reason:
        # calculate_extrapolated_features_polars left-joins back onto tracks
        # and produces NULL (not empty list) for events with 0 tracks, which
        # then crashes the downstream explode. The PU sampler enumerates
        # sampleable events from calo_hits.event_id (overlay file line 1043),
        # so keeping these vertices in particles+calo_hits preserves their
        # calo energy in the overlay; the inner-join in _overlay_tracks just
        # contributes 0 tracks for them.
        t = t.filter(pl.col('majority_particle_id').list.len() > 0)

        max_eid = int(max(p['event_id'].max(), c['event_id'].max(), t['event_id'].max())) + 1
        p_list.append(p.with_columns(pl.col('event_id') + offset))
        c_list.append(c.with_columns(pl.col('event_id') + offset))
        t_list.append(t.with_columns(pl.col('event_id') + offset))
        offset += max_eid

    return pl.concat(p_list), pl.concat(c_list), pl.concat(t_list)


def _load_hs(fs: HfFileSystem, kind: str, file_index: int, columns=None) -> pl.DataFrame:
    path = (
        f"datasets/CERN/ColliderML-Release-1/data/{HS_EVENT_NAME}_{kind}/"
        f"train-{file_index:05d}-of-{NUMBER_OF_HF_REPO_FILES:05d}.parquet"
    )
    print(f"  loading {path}")
    with fs.open(path, "rb") as f:
        return pl.read_parquet(f, columns=columns)


def run(
    hs_indices: list[int],
    pu_indices: list[int],
    chunk_size: int = 334,
    seed: int = 42,
    clusters_cutoff: float = 0.15,
    clue_backend: str = 'cpu serial',
    invisible_pu_prob: float = 0.0,
    chunk_tmp_dir: str = "/storage/agrp/barakma/PileupODD/data/tmp",
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    print(f"Loading local PU pool from files {pu_indices}")
    pu_particles, pu_calo_hits, pu_tracks = _load_local_pu(pu_indices)
    n_pu = pu_calo_hits['event_id'].n_unique()
    print(f"PU pool: {n_pu} unique pileup events from {len(pu_indices)} file(s)")

    for i in tqdm.tqdm(hs_indices, desc="HS files"):
        print(f"\n=== HS file {i:05d} ===")
        t0 = time.perf_counter()

        hs_particles = _load_hs(fs, 'particles', i, columns=PARTICLE_COLS)
        hs_calo_hits = _load_hs(fs, 'calo_hits', i, columns=HS_CALO_COLS)
        hs_tracks = _load_hs(fs, 'tracks', i)

        result = preprocess_for_model(
            hs_particles=hs_particles, hs_tracks=hs_tracks, hs_calo_hits=hs_calo_hits,
            pu_particles=pu_particles, pu_tracks=pu_tracks, pu_calo_hits=pu_calo_hits,
            pileup_level=PILEUP_LEVEL,
            seed=seed + i,
            truth_pt_cut=1, truth_eta_cut=3.0, target_pt_cut=0.3,
            clusters_cutoff=clusters_cutoff,
            clue_backend=clue_backend,
            chunk_size=chunk_size,
            chunk_tmp_dir=chunk_tmp_dir,
            invisible_pu_prob=invisible_pu_prob,
        )

        for key, df in result.items():
            out_path = OUT_DIR / f"{key}-{i:05d}.parquet"
            df.write_parquet(out_path)
            print(f"  wrote {out_path}")

        del hs_particles, hs_tracks, hs_calo_hits, result
        gc.collect()
        dt = time.perf_counter() - t0
        print(f"=== HS file {i:05d} done in {dt:.1f} s ({dt/60:.2f} min) ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs-indices', type=int, nargs='+', default=[0],
                        help='HS file indices (ttbar_pu0) on HF to process')
    parser.add_argument('--pu-indices', type=int, nargs='+', default=[0],
                        help='Local PU file indices to load into the sampling pool')
    parser.add_argument('--chunk-size', type=int, default=334,
                        help='HS events per chunk in preprocess_for_model (default 334)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clusters-cutoff', type=float, default=0.15)
    parser.add_argument('--clue-backend', type=str, default='cpu serial',
                        help="CLUEstering backend: 'cpu serial', 'cpu tbb', or 'gpu cuda'")
    parser.add_argument('--invisible-pu-prob', type=float, default=0.0)
    parser.add_argument('--chunk-tmp-dir', type=str,
                        default="/storage/agrp/barakma/PileupODD/data/tmp")
    args = parser.parse_args()

    run(
        hs_indices=args.hs_indices,
        pu_indices=args.pu_indices,
        chunk_size=args.chunk_size,
        seed=args.seed,
        clusters_cutoff=args.clusters_cutoff,
        clue_backend=args.clue_backend,
        invisible_pu_prob=args.invisible_pu_prob,
        chunk_tmp_dir=args.chunk_tmp_dir,
    )


if __name__ == '__main__':
    main()
