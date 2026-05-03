"""
Runs preprocess_for_model on a small slice of HF shard 0 and writes the
result parquets to the directory passed as the first CLI argument.

Used for old-vs-new pipeline comparison.

Usage:
    python scripts/run_small_shard.py /path/to/output_dir [num_events]
"""
import sys
import pathlib
import polars as pl
from huggingface_hub import HfFileSystem

from primary.create_trainning_dataset_pileup import preprocess_for_model


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = pathlib.Path(sys.argv[1])
    num_events = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    out_dir.mkdir(parents=True, exist_ok=True)

    event = "ttbar_pu200"
    i = 0
    n_files = 1000
    fs = HfFileSystem()

    def fetch(name, cols=None):
        p = (
            f"datasets/CERN/ColliderML-Release-1/data/{event}_{name}/"
            f"train-{i:05d}-of-{n_files:05d}.parquet"
        )
        with fs.open(p, "rb") as f:
            return pl.read_parquet(f, columns=cols)

    particles = fetch(
        "particles",
        [
            "event_id", "particle_id", "vertex_primary", "pdg_id",
            "energy", "px", "py", "pz", "vx", "vy", "vz", "parent_id",
        ],
    )
    calo_hits = fetch(
        "calo_hits",
        [
            "event_id", "detector", "total_energy", "x", "y", "z",
            "contrib_particle_ids", "contrib_energies", "contrib_times",
        ],
    )
    tracks = fetch("tracks")

    out = preprocess_for_model(
        particles=particles,
        tracks=tracks,
        calo_hits=calo_hits,
        num_of_events=num_events,
        truth_pt_cut=1.0,
        truth_eta_cut=3.0,
        target_pt_cut=0.3,
        clusters_cutoff=0.15,
    )

    for k, df in out.items():
        path = out_dir / f"{k}.parquet"
        df.write_parquet(path)
        print(f"wrote {path}  rows={len(df)}  cols={len(df.columns)}")


if __name__ == "__main__":
    main()
