"""Reading ColliderML shards from HuggingFace.

Everything goes through predicate pushdown: ``scan_parquet(url).filter(
event_id.is_in(...))`` makes polars fetch only the row groups holding the
requested events. That matters here -- a PU200 ``calo_hits`` shard is over 2 GB,
and a chunk of 100 events is a small fraction of it. It also means
``max_events_per_file`` genuinely limits I/O, so a smoke run costs seconds
rather than a multi-gigabyte download.

On ``master`` only the hard-scatter script did this; the all-vertices and
overlay scripts downloaded whole shards and then sliced them in memory. Both
read the same bytes into the same frames, so output is unaffected -- the
``all_vertices`` file-level equivalence check exercises exactly that.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional, Sequence

import polars as pl

#: Columns needed from the particles table. Anything else is left unread.
PARTICLE_COLS = [
    'event_id', 'particle_id', 'vertex_primary', 'pdg_id',
    'energy', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'parent_id',
]

#: Columns needed from calo_hits. ``contrib_times`` is pulled in only for the
#: pileup side of an overlay run, where the ToF cut consumes it.
CALO_COLS = [
    'event_id', 'detector', 'total_energy', 'x', 'y', 'z',
    'contrib_particle_ids', 'contrib_energies',
]

CALO_COLS_WITH_TIMES = CALO_COLS + ['contrib_times']


def resolve_url(repo: str, event_name: str, kind: str, file_idx: int, shards_total: int) -> str:
    """Build the direct-download URL of one shard.

    Args:
        repo: HF dataset repo id, e.g. ``CERN/ColliderML-Release-1``.
        event_name: dataset prefix, e.g. ``ttbar_pu200``.
        kind: ``particles``, ``tracks``, or ``calo_hits``.
        file_idx: shard index.
        shards_total: total shard count, part of the filename.
    """
    return (
        f"https://huggingface.co/datasets/{repo}/resolve/main/data/"
        f"{event_name}_{kind}/train-{file_idx:05d}-of-{shards_total:05d}.parquet"
    )


def _with_retry(build_frame, description: str, max_retries: int = 3, wait_seconds: int = 60):
    """Retry a HuggingFace read on transient network failures.

    Timeouts surface from polars as ``PanicException``, which is not an
    ``Exception`` subclass, so this catches ``BaseException`` and re-raises the
    two cases that must never be swallowed.

    Args:
        build_frame: zero-argument callable performing the read.
        description: what is being read, for the log line.
        max_retries: total attempts before giving up.
        wait_seconds: pause between attempts.

    Returns:
        Whatever ``build_frame`` returns.

    Raises:
        RuntimeError: when every attempt failed.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return build_frame()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - PanicException is not an Exception
            print(f"[DOWNLOAD RETRY] Attempt {attempt}/{max_retries} failed reading "
                  f"{description}: {type(exc).__name__}: {exc}", flush=True)
            if attempt < max_retries:
                print(f"[DOWNLOAD RETRY] Waiting {wait_seconds}s before retrying...", flush=True)
                time.sleep(wait_seconds)
            else:
                raise RuntimeError(
                    f"giving up after {max_retries} attempts reading {description}"
                ) from exc


def list_event_ids(
    repo: str,
    event_name: str,
    file_idx: int,
    shards_total: int,
    limit: Optional[int] = None,
) -> List[int]:
    """Discover the event ids present in a shard.

    Reads only the ``event_id`` column, so this is cheap even for a 2 GB shard.

    Args:
        repo: HF dataset repo id.
        event_name: dataset prefix.
        file_idx: shard index.
        shards_total: total shard count.
        limit: return only the first N ids, sorted ascending.

    Returns:
        Sorted event ids.
    """
    url = resolve_url(repo, event_name, 'particles', file_idx, shards_total)

    def _read() -> List[int]:
        return (
            pl.scan_parquet(url)
            .select('event_id')
            .unique()
            .sort('event_id')
            .collect()['event_id']
            .to_list()
        )

    ids = _with_retry(_read, f"{event_name} particles shard {file_idx:05d} (event ids)")
    return ids[:limit] if limit is not None else ids


def scan_events(
    repo: str,
    event_name: str,
    kind: str,
    file_idx: int,
    shards_total: int,
    event_ids: Optional[Sequence[int]] = None,
    columns: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """Read one shard, restricted to the given events and columns.

    Args:
        repo: HF dataset repo id.
        event_name: dataset prefix.
        kind: ``particles``, ``tracks``, or ``calo_hits``.
        file_idx: shard index.
        shards_total: total shard count.
        event_ids: events to fetch. ``None`` reads the whole shard.
        columns: columns to project. ``None`` reads all of them (used for
            ``tracks``, where every column is consumed).

    Returns:
        The requested slice, one row per event.
    """
    url = resolve_url(repo, event_name, kind, file_idx, shards_total)

    def _read() -> pl.DataFrame:
        lf = pl.scan_parquet(url)
        if event_ids is not None:
            # Plain list, not a Series: avoids an is_in dtype deprecation.
            lf = lf.filter(pl.col('event_id').is_in(list(event_ids)))
        if columns is not None:
            lf = lf.select(list(columns))
        return lf.collect()

    return _with_retry(_read, f"{event_name} {kind} shard {file_idx:05d}")


def load_triplet(
    repo: str,
    event_name: str,
    file_idx: int,
    shards_total: int,
    event_ids: Optional[Sequence[int]] = None,
    with_contrib_times: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read the particles / tracks / calo_hits triplet for one shard.

    Args:
        repo: HF dataset repo id.
        event_name: dataset prefix.
        file_idx: shard index.
        shards_total: total shard count.
        event_ids: events to fetch; ``None`` reads the whole shard.
        with_contrib_times: also read ``contrib_times`` from calo_hits, needed
            by the ToF cut on the pileup side of an overlay run.

    Returns:
        ``(particles, tracks, calo_hits)``.
    """
    calo_cols = CALO_COLS_WITH_TIMES if with_contrib_times else CALO_COLS
    particles = scan_events(repo, event_name, 'particles', file_idx, shards_total,
                            event_ids, PARTICLE_COLS)
    tracks = scan_events(repo, event_name, 'tracks', file_idx, shards_total,
                         event_ids, None)
    calo_hits = scan_events(repo, event_name, 'calo_hits', file_idx, shards_total,
                            event_ids, calo_cols)
    return particles, tracks, calo_hits


def load_pileup_pool(
    repo: str,
    pu_event_name: str,
    file_indices: Sequence[int],
    shards_total: int,
    max_events: Optional[int] = None,
    with_contrib_times: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load and concatenate several pileup shards into one sampling pool.

    Event ids are only unique within a shard, so each shard's ids are offset
    before concatenation. All hard-scatter shards in a run then draw from this
    single pool, which mixes pileup across shards instead of pairing them one
    to one.

    Args:
        repo: HF dataset repo id.
        pu_event_name: pileup dataset prefix.
        file_indices: shards to load.
        shards_total: total shard count.
        max_events: cap the number of events taken from each shard.
        with_contrib_times: read ``contrib_times`` for the ToF cut.

    Returns:
        ``(particles, tracks, calo_hits)`` for the combined pool.
    """
    p_list, t_list, c_list = [], [], []
    offset = 0
    for idx in file_indices:
        event_ids = None
        if max_events is not None:
            event_ids = list_event_ids(repo, pu_event_name, idx, shards_total, limit=max_events)
        p, t, c = load_triplet(repo, pu_event_name, idx, shards_total,
                               event_ids, with_contrib_times)
        # Offset so ids stay distinct across shards.
        max_eid = int(max(p['event_id'].max(), c['event_id'].max(), t['event_id'].max())) + 1
        p_list.append(p.with_columns(pl.col('event_id') + offset))
        t_list.append(t.with_columns(pl.col('event_id') + offset))
        c_list.append(c.with_columns(pl.col('event_id') + offset))
        offset += max_eid
    return pl.concat(p_list), pl.concat(t_list), pl.concat(c_list)
