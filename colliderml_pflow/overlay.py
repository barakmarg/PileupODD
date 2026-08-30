"""Stage B: synthetic pileup overlay.

Builds a PU200-like sample without simulating PU200. Each hard-scatter event
from a PU0 sample has ``N ~ Poisson(pileup_level)`` pileup events drawn from a
shared pool overlaid onto it: their calorimeter energy is added cell-by-cell,
and their tracks are appended.

Why this is worth doing: it decouples the hard-scatter process from the pileup
level, so the same hard-scatter events can be studied at several pileup levels,
and it makes the pileup content of every cluster exactly known.

*Time of flight* keeps the result physical: real pileup interactions are spread
in time, so hits landing outside the read-out window are never recorded.
Simulated PU0 events all sit at t=0, so naive overlay inflates pileup energy.
Each sampled pileup vertex is given a Gaussian time offset, the flight time to
the hit is subtracted, and hits outside the window are dropped. Hard-scatter
hits are untouched -- they were already windowed in simulation.

Ported from ``create_training_dataset_pileup_overlay.py`` on ``master``, with
the time constants moved into :class:`colliderml_pflow.config.ToFConfig`.
"""

from __future__ import annotations

import gc
from typing import Dict, Tuple

import numpy as np
import polars as pl

#: Speed of light in mm/ns, for the flight-time correction.
TOF_C_MM_NS = 299.792458


def build_sample_map(
    hs_event_ids: np.ndarray,
    pu_event_ids: np.ndarray,
    pileup_level: int,
    seed: int,
) -> pl.DataFrame:
    """Decide which pileup events are overlaid on each hard-scatter event.

    For each hard-scatter event, draw ``N ~ Poisson(pileup_level)`` and choose
    N *distinct* pileup events from the pool. Distinctness holds within a
    hard-scatter event; the same pileup event may be reused across different
    ones, which is what makes a pool of a few hundred events sufficient.

    Args:
        hs_event_ids: hard-scatter event ids to overlay onto. Sorted here, so
            a given event gets the same draw regardless of the order the caller
            supplies.
        pu_event_ids: the pileup pool. Sorted here, for the same reason.
        pileup_level: Poisson mean.
        seed: RNG seed; the caller varies it per shard.

    Returns:
        Columns ``hs_event_id`` and ``pu_event_id`` (a list per row).

    Note:
        Both id arrays are sorted before sampling, which is what makes ``seed``
        meaningful. ``numpy``'s ``poisson`` draws are consumed positionally and
        ``choice`` walks the pool in the order given, so the sample depends on
        the order of both arrays -- and those arrive from polars ``group_by`` /
        ``unique`` operations whose row order is not stable between runs. On
        ``master`` this was left unsorted, so re-running the overlay with the
        same seed produced a *different* pileup sample every time. Sorting makes
        the output a function of the seed and the pool contents alone.
    """
    rng = np.random.default_rng(seed=seed)
    hs_event_ids = np.sort(np.asarray(hs_event_ids))
    pool = np.sort(np.asarray(pu_event_ids))
    pool_size = len(pool)
    ns = rng.poisson(pileup_level, size=len(hs_event_ids))
    ns = np.minimum(ns, pool_size).astype(np.int64)
    pu_per_hs = [rng.choice(pool, size=int(n), replace=False).astype(pool.dtype) for n in ns]
    return pl.DataFrame({
        'hs_event_id': hs_event_ids,
        'pu_event_id': pu_per_hs,
    })


def overlay_calo_hits(
    hs_calo_hits: pl.DataFrame,
    pu_calo_hits: pl.DataFrame,
    sample_map_flat: pl.DataFrame,
    tof_enabled: bool = True,
    tof_window_ns: Tuple[float, float] = (-1.0, 10.0),
) -> pl.DataFrame:
    """Add sampled pileup energy onto hard-scatter cells.

    Cells are matched on ``(event_id, detector, x, y, z)`` with coordinates
    rounded to 3 decimals, and merged with a full outer join so pileup-only
    cells survive as new hits.

    Truth contributions are deliberately *not* carried over from pileup: only
    its energy is added. Hard-scatter ``contrib_particle_ids`` /
    ``contrib_energies`` pass through untouched, and pileup-only cells get
    empty lists. That is what keeps the truth attribution downstream pointing
    only at hard-scatter particles, while the pileup shows up as unattributed
    energy -- exactly the reconstruction problem the model has to solve.

    Args:
        hs_calo_hits: hard-scatter hits, one row per event.
        pu_calo_hits: pileup pool hits; needs ``t_hit`` when ``tof_enabled``.
        sample_map_flat: exploded sample map with ``hs_event_id`` /
            ``pu_event_id``, plus ``time_shift`` when ``tof_enabled``.
        tof_enabled: apply the read-out time-window cut to pileup hits.
        tof_window_ns: ``(t_min, t_max)`` acceptance window.

    Returns:
        Merged hits, one row per hard-scatter event, ready for clustering.
    """
    pu_select_cols = ['event_id', 'detector', 'total_energy', 'x', 'y', 'z']
    if tof_enabled:
        pu_select_cols.append('t_hit')
    pu_cells = (
        pu_calo_hits.lazy()
        .select(pu_select_cols)
        .explode([c for c in pu_select_cols if c != 'event_id'])
        # Exploding an empty hit list yields a phantom all-null row; drop it.
        # The pileup event stays in the sampler pool regardless, so a vertex
        # with no calorimeter hits still gets sampled and simply contributes
        # nothing -- which is the correct model of an invisible vertex.
        .filter(pl.col('detector').is_not_null())
        .with_columns([
            pl.col('x').round(3),
            pl.col('y').round(3),
            pl.col('z').round(3),
        ])
    )

    # Joining the sample map replicates each pileup hit onto every hard-scatter
    # event that sampled it.
    pu_cell_energy = sample_map_flat.lazy().join(
        pu_cells, left_on='pu_event_id', right_on='event_id')

    if tof_enabled:
        t_min, t_max = tof_window_ns
        pu_cell_energy = (
            pu_cell_energy
            .with_columns(
                t_corr=(
                    pl.col('t_hit') + pl.col('time_shift')
                    - (pl.col('x').pow(2) + pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt()
                    / TOF_C_MM_NS
                )
            )
            .filter((pl.col('t_corr') >= t_min) & (pl.col('t_corr') <= t_max))
        )

    pu_cell_energy = (
        pu_cell_energy
        .group_by([pl.col('hs_event_id').alias('event_id'), 'detector', 'x', 'y', 'z'])
        .agg(pl.col('total_energy').sum().alias('pu_energy'))
    )

    hs_flat = (
        hs_calo_hits.lazy()
        .select(['event_id', 'detector', 'total_energy', 'x', 'y', 'z',
                 'contrib_particle_ids', 'contrib_energies'])
        .explode(['detector', 'total_energy', 'x', 'y', 'z',
                  'contrib_particle_ids', 'contrib_energies'])
        .with_columns([
            pl.col('x').round(3),
            pl.col('y').round(3),
            pl.col('z').round(3),
        ])
    )

    merged_flat = (
        hs_flat
        .join(pu_cell_energy,
              on=['event_id', 'detector', 'x', 'y', 'z'],
              how='full', coalesce=True)
        .with_columns([
            (pl.col('total_energy').fill_null(0.0) + pl.col('pu_energy').fill_null(0.0))
            .alias('total_energy'),
            pl.col('contrib_particle_ids').fill_null(pl.lit([], dtype=pl.List(pl.UInt64))),
            pl.col('contrib_energies').fill_null(pl.lit([], dtype=pl.List(pl.Float32))),
        ])
        .drop('pu_energy')
    )

    return (
        merged_flat
        .group_by('event_id', maintain_order=True)
        .agg(pl.all())
        .sort('event_id')
        .collect(streaming=True)
    )


def overlay_tracks(
    hs_tracks: pl.DataFrame,
    pu_tracks: pl.DataFrame,
    sample_map_flat: pl.DataFrame,
) -> pl.DataFrame:
    """Append sampled pileup tracks to each hard-scatter event's track list.

    Hard-scatter tracks come first in every per-event list, then pileup ones.

    Args:
        hs_tracks: hard-scatter tracks, one row per event.
        pu_tracks: pileup pool tracks.
        sample_map_flat: exploded sample map.

    Returns:
        Combined tracks with an extra ``source_pileup_event_id`` list column:
        null on hard-scatter rows, the originating pileup event id on pileup
        rows. Downstream,
        :func:`colliderml_pflow.aggregate.filter_orphans_and_reindex` keys off
        that column to stop pileup tracks being mis-attributed to hard-scatter
        particles whose event-local ``particle_id`` happens to collide.
    """
    hs_track_cols = [c for c in hs_tracks.columns if c != 'event_id']
    hs_flat = (
        hs_tracks.lazy()
        .explode(hs_track_cols)
        .with_columns(pl.lit(None, dtype=pl.UInt32).alias('source_pileup_event_id'))
    )

    pu_track_cols = [c for c in pu_tracks.columns if c != 'event_id']
    pu_flat = (
        pu_tracks.lazy()
        .explode(pu_track_cols)
        .rename({'event_id': 'pu_event_id'})
    )
    pu_overlaid = (
        sample_map_flat.lazy()
        .join(pu_flat, on='pu_event_id', how='inner')
        .rename({'hs_event_id': 'event_id',
                 'pu_event_id': 'source_pileup_event_id'})
        # The join emits rows in an unspecified order, which would make the
        # pileup block of each event's track list differ between runs.
        # (source_pileup_event_id, track_id) is unique per pileup track, so
        # sorting on it pins the order without changing the contents.
        .sort(['event_id', 'source_pileup_event_id', 'track_id'])
    )

    final_cols = ['event_id'] + hs_track_cols + ['source_pileup_event_id']
    return (
        # Hard-scatter tracks first in every per-event list, then pileup.
        pl.concat(
            [hs_flat.select(final_cols), pu_overlaid.select(final_cols)],
            how='vertical_relaxed',
        )
        .group_by('event_id', maintain_order=True)
        .agg(pl.all())
        .sort('event_id')
        .collect(streaming=True)
    )


def run_overlay(
    primary: Dict[str, pl.DataFrame],
    pileup: Dict[str, pl.DataFrame],
    overlay_cfg,
    seed: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Draw the sample map and produce overlaid hits and tracks.

    Args:
        primary: Stage-A output for the hard-scatter source. Its ``calo_hits``
            and ``tracks`` entries are consumed and deleted here; the particle
            entries are left for :func:`colliderml_pflow.pipeline.run_tail`.
        pileup: Stage-A output for the pileup pool. Left intact so the caller
            can reuse it across chunks.
        overlay_cfg: :class:`colliderml_pflow.config.OverlayConfig`.
        seed: RNG seed for this batch.

    Returns:
        ``(merged_calo_hits, merged_tracks)``.
    """
    hs_event_ids = primary['calo_hits']['event_id'].to_numpy()

    # Enumerate the pool from particles, not calo hits: particles are the
    # canonical record of which vertices exist, so vertices that deposited no
    # energy are still sampled at their Poisson rate and contribute nothing.
    if 'particle_event_ids' in pileup:
        pu_event_ids = pileup['particle_event_ids']['event_id'].to_numpy()
    else:
        pu_event_ids = pileup['calo_hits']['event_id'].unique(maintain_order=True).to_numpy()

    sample_map = build_sample_map(
        hs_event_ids, pu_event_ids, overlay_cfg.pileup_level, seed,
    )
    sample_map_flat = sample_map.explode('pu_event_id')
    del sample_map

    tof = overlay_cfg.tof
    if tof.enabled:
        # One Gaussian time shift per sampled pileup vertex, i.e. per
        # hard-scatter/pileup pair.
        rng = np.random.default_rng(seed)
        time_shifts = rng.normal(
            loc=0.0, scale=tof.sigma_ns, size=len(sample_map_flat)).astype(np.float32)
        sample_map_flat = sample_map_flat.with_columns(
            pl.Series('time_shift', time_shifts, dtype=pl.Float32)
        )

    print(f"[SAMPLE MAP] {len(sample_map_flat)} HS-PU pairs across "
          f"{len(hs_event_ids)} HS events.")

    print(f"[OVERLAY CALO HITS] Merging HS and pileup hits per cell "
          f"(tof_enabled={tof.enabled})...")
    merged_calo_hits = overlay_calo_hits(
        primary['calo_hits'], pileup['calo_hits'], sample_map_flat,
        tof_enabled=tof.enabled,
        tof_window_ns=tuple(tof.window_ns),
    )
    del primary['calo_hits']
    gc.collect()

    print("[OVERLAY TRACKS] Merging HS and pileup tracks...")
    merged_tracks = overlay_tracks(primary['tracks'], pileup['tracks'], sample_map_flat)
    del primary['tracks'], sample_map_flat
    gc.collect()

    return merged_calo_hits, merged_tracks
