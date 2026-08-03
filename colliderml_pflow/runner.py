"""Shard orchestration: chunked execution in spawned worker processes.

Each chunk of events runs in its own child process. That is not for
parallelism -- children run one at a time -- but for memory. Polars' allocator
holds on to arenas after a large transient (the contributor-level explodes peak
in the tens of GB), and it does not hand them back within a process. Letting
the child exit makes the OS reclaim everything, so RAM does not accumulate
across chunks or shards.

``spawn``, not ``fork``: polars uses a Rayon thread pool, and a forked child
inherits the pool's mutex state without its worker threads, so the first polars
operation deadlocks.

Two chunking strategies, matching what each mode requires:

*Non-overlay* -- the child fetches its own events by predicate pushdown and
runs the whole pipeline on them. The parent never holds event data, so its
memory stays flat regardless of shard size.

*Overlay* -- Stage A must run over the whole hard-scatter shard and the whole
pileup pool before chunking, because the pool is the shared sampling
population. The parent prepares both, then ships each hard-scatter chunk plus
the prepared pool to a child that runs Stage B and C. Frames travel as pickled
Arrow buffers over a pipe, with no disk round-trip.

Ported from the chunk drivers in the three ``master`` scripts.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from colliderml_pflow import hf_io
from colliderml_pflow.config import OUTPUT_KEYS, Config


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def _write_chunk(out: Dict[str, pl.DataFrame], tmp_dir: str, ci: int) -> None:
    tmp = Path(tmp_dir)
    for key in OUTPUT_KEYS:
        out[key].write_parquet(tmp / f'chunk_{ci:04d}_{key}.parquet')


def _child_preamble(ci: int) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    print(f"[CHUNK {ci + 1} CHILD ALIVE] pid={os.getpid()}", flush=True)


def _standalone_chunk_worker(ci: int, cfg: Config, file_idx: int,
                             chunk_event_ids: List[int], tmp_dir: str) -> None:
    """Child entry point for ``hard_scatter`` / ``all_vertices``.

    Fetches only this chunk's events, runs the full pipeline, writes the four
    chunk parquets and exits. Must be a module-level function so ``spawn`` can
    re-import it.
    """
    try:
        _child_preamble(ci)
        from colliderml_pflow.pipeline import preprocess_events

        t0 = time.perf_counter()
        particles, tracks, calo_hits = hf_io.load_triplet(
            cfg.dataset.repo, cfg.dataset.event_name, file_idx,
            cfg.dataset.shards_total, chunk_event_ids, with_contrib_times=False,
        )
        print(f"[CHUNK {ci + 1} CHILD] HF scan done in {time.perf_counter() - t0:.1f}s "
              f"(particles={particles.height}, tracks={tracks.height}, "
              f"calo_hits={calo_hits.height})", flush=True)

        out = preprocess_events(
            cfg.mode,
            particles=particles, tracks=tracks, calo_hits=calo_hits,
            cuts=cfg.cuts, clustering=cfg.clustering,
            keep_all_vertices=cfg.keep_all_vertices,
        )
        _write_chunk(out, tmp_dir, ci)
    except BaseException:  # noqa: BLE001 - report and signal failure via exit code
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)


def _overlay_chunk_worker(ci: int, cfg: Config, hs_chunk: Dict[str, pl.DataFrame],
                          pileup: Dict[str, pl.DataFrame], chunk_seed: int,
                          tmp_dir: str) -> None:
    """Child entry point for ``overlay``: runs Stage B and C on one chunk."""
    try:
        _child_preamble(ci)
        from colliderml_pflow.overlay import run_overlay
        from colliderml_pflow.pipeline import run_tail

        merged_hits, merged_tracks = run_overlay(
            hs_chunk, pileup, cfg.overlay, seed=chunk_seed)
        out = run_tail(hs_chunk, merged_hits, merged_tracks,
                       cuts=cfg.cuts, clustering=cfg.clustering)
        _write_chunk(out, tmp_dir, ci)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)


def _run_child(ctx, target, args, label: str) -> None:
    """Spawn one child, wait for it, and raise if it failed."""
    t0 = time.perf_counter()
    proc = ctx.Process(target=target, args=args)
    proc.start()
    proc.join()
    dt = time.perf_counter() - t0
    if proc.exitcode != 0:
        raise RuntimeError(
            f"{label} (pid {proc.pid}) failed with exit code {proc.exitcode} after {dt:.1f}s")
    print(f"[{label} DONE] {dt:.1f}s. PARENT RAM: {_rss_mb():.2f} MB", flush=True)


def _concat_chunks(tmp: Path, out_dir: Path, file_idx: int) -> None:
    """Merge per-chunk parquets into one file per output table.

    Chunks are consecutive slices of the sorted event-id list, so concatenating
    them in chunk order already yields events in order.
    """
    for key in OUTPUT_KEYS:
        parts = sorted(tmp.glob(f'chunk_*_{key}.parquet'))
        if not parts:
            print(f"  WARNING: no chunk outputs for {key} (shard {file_idx:05d})")
            continue
        merged = pl.concat([pl.read_parquet(p) for p in parts])
        target = out_dir / f"{key}-{file_idx:05d}.parquet"
        merged.write_parquet(target)
        print(f"[CONCAT {key}] {len(parts)} chunks -> {target} ({merged.height} rows)")
        del merged
        for part in parts:
            part.unlink()
        gc.collect()


def _chunk_event_ids(event_ids: List[int], chunk_size: int) -> List[List[int]]:
    if chunk_size is None or chunk_size <= 0:
        return [event_ids]
    return [event_ids[i:i + chunk_size] for i in range(0, len(event_ids), chunk_size)]


def _process_shard_standalone(cfg: Config, file_idx: int, event_ids: List[int],
                              out_dir: Path, tmp_root: Path, ctx) -> None:
    """Run one shard in ``hard_scatter`` / ``all_vertices`` mode."""
    chunks = _chunk_event_ids(event_ids, cfg.runtime.chunk_size)
    print(f"\n=== Shard {file_idx:05d}: {len(event_ids)} events -> {len(chunks)} chunk(s) ===")

    with tempfile.TemporaryDirectory(prefix=f'pp_chunks_{file_idx:05d}_', dir=tmp_root) as tmpdir:
        for ci, chunk_ids in enumerate(chunks):
            print(f"\n[CHUNK {ci + 1}/{len(chunks)}] spawning child for {len(chunk_ids)} events. "
                  f"PARENT RAM: {_rss_mb():.2f} MB", flush=True)
            _run_child(ctx, _standalone_chunk_worker,
                       (ci, cfg, file_idx, chunk_ids, tmpdir),
                       f"CHUNK {ci + 1}/{len(chunks)}")
        _concat_chunks(Path(tmpdir), out_dir, file_idx)


def _process_shard_overlay(cfg: Config, file_idx: int, event_ids: List[int],
                           pileup: Dict[str, pl.DataFrame],
                           out_dir: Path, tmp_root: Path, ctx) -> None:
    """Run one shard in ``overlay`` mode.

    Stage A runs here in the parent over the whole shard, because the chunks
    that follow all sample from the same prepared pileup pool.
    """
    import numpy as np

    from colliderml_pflow.pipeline import prepare_source

    particles, tracks, calo_hits = hf_io.load_triplet(
        cfg.dataset.repo, cfg.dataset.event_name, file_idx,
        cfg.dataset.shards_total, event_ids, with_contrib_times=False,
    )
    primary = prepare_source(
        particles, tracks, calo_hits,
        role="primary", keep_all_vertices=cfg.keep_all_vertices,
        cuts=cfg.cuts, tof=cfg.overlay.tof,
    )
    del particles, tracks, calo_hits
    gc.collect()

    hs_event_ids = primary['calo_hits']['event_id'].to_numpy()
    # Per-shard seed, so each shard draws a different pileup sample while
    # staying reproducible from the config alone.
    file_seed = cfg.overlay.seed + file_idx

    chunk_size = cfg.runtime.chunk_size
    if chunk_size is None or chunk_size <= 0 or len(hs_event_ids) <= chunk_size:
        chunks = [hs_event_ids.tolist()]
    else:
        chunks = [hs_event_ids[i:i + chunk_size].tolist()
                  for i in range(0, len(hs_event_ids), chunk_size)]

    print(f"\n=== Shard {file_idx:05d}: {len(hs_event_ids)} HS events -> "
          f"{len(chunks)} chunk(s) ===")

    keyed = ['particles', 'tracks', 'calo_hits',
             'particles_pid_to_vertex', 'particles_selected_ids']

    with tempfile.TemporaryDirectory(prefix=f'ov_chunks_{file_idx:05d}_', dir=tmp_root) as tmpdir:
        for ci, chunk_ids in enumerate(chunks):
            print(f"\n[CHUNK {ci + 1}/{len(chunks)}] slicing {len(chunk_ids)} HS events "
                  f"in parent. PARENT RAM: {_rss_mb():.2f} MB", flush=True)
            hs_chunk = {
                k: primary[k].filter(pl.col('event_id').is_in(chunk_ids)) for k in keyed
            }
            # Derive the chunk seed by hashing (file_seed, chunk_index) through
            # SeedSequence rather than adding them. Addition would collide
            # whenever file_a's chunk_n and file_b's chunk_m satisfy
            # a + n == b + m, silently reusing a pileup sample across shards.
            chunk_seed = int(np.random.SeedSequence(
                entropy=int(file_seed), spawn_key=(int(ci),)
            ).generate_state(1, dtype=np.uint64)[0])

            _run_child(ctx, _overlay_chunk_worker,
                       (ci, cfg, hs_chunk, pileup, chunk_seed, tmpdir),
                       f"CHUNK {ci + 1}/{len(chunks)}")
            del hs_chunk
            gc.collect()

        del primary
        gc.collect()
        _concat_chunks(Path(tmpdir), out_dir, file_idx)


def run_preprocessing(cfg: Config) -> Path:
    """Run the configured preprocessing over every selected shard.

    Args:
        cfg: the run specification.

    Returns:
        The output directory that was written to.
    """
    print("=" * 72)
    print(cfg.describe())
    print("=" * 72, flush=True)

    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = cfg.resolved_tmp_dir()
    tmp_root.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context('spawn')
    file_indices = cfg.dataset.resolved_file_indices()
    pinned = cfg.dataset.explicit_event_ids()

    pileup: Optional[Dict[str, pl.DataFrame]] = None
    if cfg.is_overlay:
        # Prepared once and reused for every shard. Stage A on the pool depends
        # only on the pool itself, so caching it changes nothing in the output
        # and skips repeating the expensive hit-time precompute per shard.
        from colliderml_pflow.pipeline import prepare_source

        print(f"\n=== Loading pileup pool from {cfg.overlay.pu_event_name} "
              f"shards {cfg.overlay.pu_file_indices} ===", flush=True)
        pu_particles, pu_tracks, pu_calo_hits = hf_io.load_pileup_pool(
            cfg.dataset.repo, cfg.overlay.pu_event_name, cfg.overlay.pu_file_indices,
            cfg.dataset.shards_total, max_events=cfg.overlay.pu_max_events,
            with_contrib_times=cfg.overlay.tof.enabled,
        )
        print(f"    pool: {pu_calo_hits['event_id'].n_unique()} pileup events")
        pileup = prepare_source(
            pu_particles, pu_tracks, pu_calo_hits,
            role="pileup", keep_all_vertices=False,
            cuts=cfg.cuts, tof=cfg.overlay.tof,
        )
        del pu_particles, pu_tracks, pu_calo_hits
        gc.collect()

    overall_t0 = time.perf_counter()
    for file_idx in file_indices:
        t0 = time.perf_counter()
        if pinned is not None:
            event_ids = pinned[file_idx]
        else:
            event_ids = hf_io.list_event_ids(
                cfg.dataset.repo, cfg.dataset.event_name, file_idx,
                cfg.dataset.shards_total, limit=cfg.dataset.max_events_per_file,
            )

        if cfg.is_overlay:
            assert pileup is not None
            _process_shard_overlay(cfg, file_idx, event_ids, pileup, out_dir, tmp_root, ctx)
        else:
            _process_shard_standalone(cfg, file_idx, event_ids, out_dir, tmp_root, ctx)

        dt = time.perf_counter() - t0
        print(f"=== Shard {file_idx:05d} done in {dt:.1f}s ({dt / 60:.2f} min) ===", flush=True)

    total = time.perf_counter() - overall_t0
    print(f"\n[ALL SHARDS DONE] {len(file_indices)} shard(s) in {total:.1f}s "
          f"({total / 60:.2f} min) -> {out_dir}")
    return out_dir
