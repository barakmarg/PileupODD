"""
Process 1k H -> b b-bar events from `ggf_pu200` via the spawn-chunked
preprocessing pipeline.

- Reads the per-file label parquets under data/hf_decay_labels/ and selects the
  first 1000 events where channel == "bb̄".
- Groups the selected (file_idx, event_id) pairs into
  {file_idx: [event_id, ...]} and hands the dict to `run_preprocessing_pipeline`.
- chunk_size=50 -> each spawn child pulls only its chunk's events from HF via
  the same `scan_parquet(url).filter(event_id.is_in(...))` predicate-pushdown
  pattern used in analysis/load_higgs_diphoton_events.py.

Outputs land under /storage/agrp/barakma/PileupODD/data/ggf_pu200/
(one set of 4 parquets per source file_idx that contributed events).
"""

import sys
from collections import defaultdict
from pathlib import Path

# Make the `primary` package importable.
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')

import polars as pl

from primary.create_trainning_dataset_pileup import run_preprocessing_pipeline

LABELS_DIR = Path("/storage/agrp/barakma/PileupODD/data/hf_decay_labels")
TARGET_CHANNEL = "bb̄"          # H -> b b-bar label as stored in labels parquets
N_EVENTS = 1000
CHUNK_SIZE = 50
EVENT_NAME = "ggf_pu200"


def collect_first_n_bb_events(n: int) -> dict[int, list[int]]:
    """
    Scan labels_file_*.parquet in order; accumulate (file_idx, event_id) pairs
    where channel matches the H->bb label until `n` events are collected.
    Returns {file_idx: [event_id, ...]} preserving discovery order per file.
    """
    label_files = sorted(LABELS_DIR.glob("labels_file_*.parquet"))
    if not label_files:
        raise FileNotFoundError(f"no labels_file_*.parquet under {LABELS_DIR}")

    event_ids: dict[int, list[int]] = defaultdict(list)
    collected = 0
    for path in label_files:
        df = (
            pl.read_parquet(path, columns=["file_idx", "event_id", "channel"])
            .filter(pl.col("channel") == TARGET_CHANNEL)
            .sort(["file_idx", "event_id"])
        )
        for file_idx, ev in df.select("file_idx", "event_id").iter_rows():
            event_ids[int(file_idx)].append(int(ev))
            collected += 1
            if collected >= n:
                break
        if collected >= n:
            break

    if collected < n:
        print(f"warning: only found {collected} H->bb events (< {n}) "
              f"after scanning {len(label_files)} label files")

    print(f"collected {collected} H->bb events across {len(event_ids)} HF files")
    return dict(event_ids)


def main():
    event_ids = collect_first_n_bb_events(N_EVENTS)
    # Spot-check distribution
    sizes = sorted(((k, len(v)) for k, v in event_ids.items()), key=lambda t: t[0])
    print(f"per-file H->bb count (first 10): {sizes[:10]}")
    print(f"per-file H->bb count (last 5):   {sizes[-5:]}")

    run_preprocessing_pipeline(
        event_ids=event_ids,
        event_name=EVENT_NAME,
        chunk_size=CHUNK_SIZE,
    )


if __name__ == "__main__":
    main()
