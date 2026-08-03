"""Train / validation / test splitting.

Splits by *event*, never by row: every output table is filtered by the same
event-id sets, so a given event's particles, clusters, tracks and deposits all
land in the same split. Splitting rows independently would leak information
between splits, since the tables are different views of the same events.

Ported from ``split_train_val_test`` on ``master``, with a shard-aware wrapper
added for splitting a written dataset on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import polars as pl
from sklearn.model_selection import train_test_split

from colliderml_pflow.config import OUTPUT_KEYS

SPLIT_NAMES = ("train", "val", "test")


def split_train_val_test(
    datasets: Dict[str, pl.DataFrame],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, pl.DataFrame]]:
    """Partition in-memory tables into train / val / test by event id.

    Args:
        datasets: the output tables, keyed by name. ``target_particles``
            supplies the event-id universe.
        train_frac: fraction of events for training.
        val_frac: fraction for validation.
        test_frac: fraction for test. ``val_frac`` and ``test_frac`` are
            renormalised against each other, so only their ratio matters once
            ``train_frac`` is fixed.
        seed: RNG seed for reproducible splits.

    Returns:
        ``{split_name: {table_name: frame}}``.
    """
    event_ids = datasets['target_particles']['event_id'].unique().to_numpy()
    train_ids, temp_ids = train_test_split(event_ids, train_size=train_frac, random_state=seed)
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_frac / (val_frac + test_frac), random_state=seed)

    split_mapping = {"train": train_ids, "val": val_ids, "test": test_ids}
    results: Dict[str, Dict[str, pl.DataFrame]] = {name: {} for name in SPLIT_NAMES}

    for key, df in datasets.items():
        for split_name, ids in split_mapping.items():
            results[split_name][key] = df.filter(pl.col("event_id").is_in(ids))
    return results


def split_dataset_dir(
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Path:
    """Split a written dataset into ``train/``, ``val/`` and ``test/`` subdirectories.

    Shards are split individually and written under the same name in each split
    directory, so the sharded layout -- and its lazy-loading behaviour
    downstream -- is preserved.

    Args:
        data_dir: directory holding ``<table>-<shard>.parquet`` files.
        output_dir: where to write the split tree. Defaults to ``data_dir``.
        train_frac: fraction of events for training.
        val_frac: fraction for validation.
        test_frac: fraction for test.
        seed: RNG seed. Held constant across shards, but each shard splits its
            own events, so the overall fractions hold.

    Returns:
        The directory the split tree was written to.

    Raises:
        RuntimeError: if no shards were found.
    """
    src = Path(data_dir)
    dst = Path(output_dir) if output_dir else src

    shard_ids: List[str] = sorted(
        f.name.split('-')[-1].split('.')[0] for f in src.glob("target_particles-*.parquet")
    )
    if not shard_ids:
        raise RuntimeError(f"no target_particles-*.parquet files found in {src}")

    for name in SPLIT_NAMES:
        (dst / name).mkdir(parents=True, exist_ok=True)

    for shard in shard_ids:
        tables = {}
        for key in OUTPUT_KEYS:
            fpath = src / f"{key}-{shard}.parquet"
            if fpath.exists():
                tables[key] = pl.read_parquet(fpath)
        if 'target_particles' not in tables:
            print(f"  skipping shard {shard}: no target_particles file")
            continue

        splits = split_train_val_test(
            tables, train_frac=train_frac, val_frac=val_frac,
            test_frac=test_frac, seed=seed)
        for split_name, frames in splits.items():
            for key, df in frames.items():
                df.write_parquet(dst / split_name / f"{key}-{shard}.parquet")
        counts = {n: splits[n]['target_particles'].height for n in SPLIT_NAMES}
        print(f"[SPLIT {shard}] events -> {counts}")

    print(f"[SPLIT] wrote {len(shard_ids)} shard(s) into {dst}/{{train,val,test}}")
    return dst
