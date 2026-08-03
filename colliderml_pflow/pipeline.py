"""The preprocessing pipeline: Stage A (prepare) and Stage C (aggregate).

The three dataset variants differ far less than the original three scripts
suggest. Factored out, the work is:

**Stage A --** :func:`prepare_source` -- run once per input source. Casts to
Float32, computes extrapolated track features, filters and regroups tracks,
and adds particle kinematics. The *primary* source additionally applies the
vertex policy and builds the particle masks that define the truth targets; the
*pileup* source runs a strictly reduced version (no masks, no targets) and is
the only one to precompute per-hit times for the ToF cut.

**Stage B --** mode-specific, and only ``overlay`` has one. See
:mod:`colliderml_pflow.overlay`.

**Stage C --** :func:`run_tail` -- shared by every mode. Clusters the hits,
drops sub-threshold clusters, attributes deposited energy back to target
particles, and emits the four output tables.

:func:`preprocess_events` wires A -> B -> C according to ``mode``.

Ported from the three ``create_*_dataset_pileup*.py`` scripts on ``master``.
The stage sequence and the polars expressions are unchanged, and
``tests/test_equivalence.py`` checks the result against those scripts: the
pre-clustering tables (``target_particles``, ``tracks``) must match exactly,
while the cluster-dependent tables are compared through label-invariant physics
quantities, because clustering is stochastic. See the README.
"""

from __future__ import annotations

import gc
import os
from typing import Dict, Optional

import polars as pl

from colliderml_pflow.aggregate import create_calo_clusters, filter_orphans_and_reindex
from colliderml_pflow.calibration import CALIBRATION
from colliderml_pflow.clustering import clue_clustering
from colliderml_pflow.config import ClusteringConfig, Cuts, ToFConfig
from colliderml_pflow.preprocessing import (
    add_created_inside_calo_mask,
    add_eta_and_phi_and_pt,
    add_orphan_mask,
    add_particle_have_track_mask,
    backtrack_to_target,
    calculate_extrapolated_features_polars,
    cluster_contrib_energy,
    cluster_purity,
    cluster_vertex_primary_deps,
    get_particles_id_parent_of_inside_calo_particles_maskv3,
    set_target_particles_maskv4,
)

#: Pileup events per batch in the ToF hit-time precompute. The level-2 explode
#: of contributor times over a full pileup pool peaks near a billion rows, and
#: the allocator does not hand that memory back, so it is done in slices.
_T_HIT_BATCH = 1000


def _rss_mb() -> float:
    """Resident set size of this process in MiB, or 0 if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def _malloc_trim() -> None:
    """Ask glibc to return free arenas to the OS after a large transient."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _precompute_hit_times(calo_hits: pl.DataFrame, tag: str) -> pl.DataFrame:
    """Replace ``contrib_times`` with a per-hit energy-weighted mean ``t_hit``.

    Each calorimeter hit carries a list of contributing particles with their
    individual energies and arrival times. The ToF cut needs one time per hit,
    so contributions are collapsed to their energy-weighted mean.

    Args:
        calo_hits: pileup hits, one row per event, including ``contrib_times``.
        tag: label for progress output.

    Returns:
        ``calo_hits`` with ``contrib_times`` dropped and a ``t_hit`` list column
        appended, aligned to the existing per-hit lists.
    """
    n_pu = calo_hits.height
    print(f"[{tag} T_HIT PRECOMPUTE] energy-weighted hit time "
          f"({n_pu} events in batches of {_T_HIT_BATCH})...")
    t_hit_parts = []
    for start in range(0, n_pu, _T_HIT_BATCH):
        sub = calo_hits.slice(start, _T_HIT_BATCH)
        part = (
            sub.lazy()
            .select(['total_energy', 'contrib_energies', 'contrib_times'])
            .with_row_index('_ev')
            .with_columns(
                _hit_pos=pl.int_ranges(
                    0, pl.col('total_energy').list.len(), dtype=pl.UInt32
                )
            )
            .drop('total_energy')
            # Level-1: one row per hit.
            .explode(['contrib_energies', 'contrib_times', '_hit_pos'])
            .with_row_index('_hit')
            # Level-2: one row per contributor.
            .explode(['contrib_energies', 'contrib_times'])
            .group_by('_hit', maintain_order=True)
            .agg([
                pl.col('_ev').first(),
                pl.col('_hit_pos').first(),
                (
                    (pl.col('contrib_times') * pl.col('contrib_energies')).sum()
                    / pl.col('contrib_energies').sum().clip(lower_bound=1e-30)
                ).cast(pl.Float32).alias('t_hit'),
            ])
            # Restore within-event hit order before re-aggregating.
            .sort(['_ev', '_hit_pos'])
            .group_by('_ev', maintain_order=True)
            .agg(pl.col('t_hit'))
            .collect(streaming=True)
        )
        t_hit_parts.append(part.select('t_hit'))
        del sub, part
        gc.collect()

    # Slices are consecutive, so concat preserves the original event order and
    # the column can be stitched back on positionally.
    t_hit_col = pl.concat(t_hit_parts)
    del t_hit_parts
    gc.collect()
    calo_hits = calo_hits.drop('contrib_times').hstack(t_hit_col)
    del t_hit_col
    gc.collect()
    _malloc_trim()
    print(f"[{tag} T_HIT DONE] RAM: {_rss_mb():.2f} MB")
    return calo_hits


def prepare_source(
    particles: pl.DataFrame,
    tracks: pl.DataFrame,
    calo_hits: pl.DataFrame,
    *,
    role: str,
    keep_all_vertices: bool,
    cuts: Cuts,
    tof: Optional[ToFConfig] = None,
    num_of_events: int = -1,
) -> Dict[str, pl.DataFrame]:
    """Stage A: prepare one input source for clustering.

    Args:
        particles: per-event particle records with parentage and vertices.
        tracks: per-event reconstructed tracks.
        calo_hits: per-event calorimeter hits with truth contributions.
        role: ``'primary'`` for the hard-scatter source, ``'pileup'`` for the
            overlay pileup pool. Only the primary source produces targets.
        keep_all_vertices: primary source only. ``False`` keeps just
            ``vertex_primary == 1``; ``True`` keeps every vertex. Ignored for
            the pileup role, which never applies a vertex filter.
        cuts: selection thresholds.
        tof: ToF settings; when enabled and ``role == 'pileup'``, per-hit times
            are precomputed here.
        num_of_events: primary source only -- keep just the first N event ids.
            Shrinking the pileup pool instead would starve the Poisson sampler,
            so it is deliberately not applied there.

    Returns:
        Always ``tracks`` and ``calo_hits``. The primary role adds
        ``particles`` (with target masks), ``particles_pid_to_vertex``
        (particle -> originating vertex, snapshotted *before* any vertex
        filter so overlaid clusters can still be labelled by vertex), and
        ``particles_selected_ids``. The pileup role instead returns
        ``particle_event_ids``.
    """
    is_primary = role == "primary"
    tag = "HS" if is_primary else "PU"
    print(f"\n[{tag} PREPROCESS START] RAM: {_rss_mb():.2f} MB")

    if num_of_events >= 0 and is_primary:
        # Files carry global event ids, not 0-based ones, so slice by id.
        first_n_ids = particles['event_id'].unique().sort()[:num_of_events]
        particles = particles.filter(pl.col('event_id').is_in(first_n_ids))
        tracks = tracks.filter(pl.col('event_id').is_in(first_n_ids))
        calo_hits = calo_hits.filter(pl.col('event_id').is_in(first_n_ids))

    # Float32 cast: halves memory and matches the precision the model consumes.
    particles = particles.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32)),
    ])
    tracks = tracks.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32)),
    ])
    calo_hits = calo_hits.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32)),
    ])

    if (not is_primary) and tof is not None and tof.enabled and 'contrib_times' in calo_hits.columns:
        calo_hits = _precompute_hit_times(calo_hits, tag)

    tracks = calculate_extrapolated_features_polars(tracks)
    print(f"[{tag} EXTRAPOLATED] RAM: {_rss_mb():.2f} MB")

    # Track pt/eta cut, with each track's originating vertex joined on.
    # `local_order` preserves the within-event track order across the
    # explode/regroup round-trip.
    track_cols = [c for c in tracks.columns if c != 'event_id']
    tracks = (
        tracks.lazy()
        .with_columns(
            local_order=pl.int_ranges(
                start=0,
                end=pl.col('majority_particle_id').list.len(),
                dtype=pl.UInt32,
            )
        )
        .select(['event_id', 'local_order'] + track_cols)
        .explode(['local_order'] + track_cols)
        .filter(pl.col('pt') > cuts.truth_pt)
        .filter(pl.col('eta').abs() < cuts.truth_eta)
        .join(
            particles.lazy().select(['event_id', 'particle_id', 'vertex_primary'])
            .explode('particle_id', 'vertex_primary'),
            left_on=['event_id', 'majority_particle_id'],
            right_on=['event_id', 'particle_id'],
            how='left',
        )
        .with_columns(pl.col('majority_particle_id').cast(pl.Int64))
        .sort(['event_id', 'local_order'])
        .group_by('event_id', maintain_order=True)
        .agg([pl.col(c) for c in track_cols] + [pl.col('vertex_primary')])
        .sort('event_id')
        .collect(streaming=True)
    )
    print(f"[{tag} TRACKS FILTERED] RAM: {_rss_mb():.2f} MB")

    particles_pid_to_vertex = None
    particles_selected_ids = None

    if is_primary:
        # Snapshot particle -> vertex for ALL particles, including pileup ones,
        # BEFORE any vertex filter. cluster_vertex_primary_deps later uses this
        # to break each cluster's energy down by originating vertex, which is
        # only possible if the mapping was captured pre-filter.
        particles_pid_to_vertex = (
            particles.lazy()
            .select(['event_id', 'particle_id', 'vertex_primary'])
            .explode('particle_id', 'vertex_primary')
            .with_columns(
                pl.col('particle_id').cast(pl.Int64),
                pl.col('vertex_primary').cast(pl.UInt16),
            )
            .collect(streaming=True)
        )

        if not keep_all_vertices:
            # Keep only hard-scatter particles. Find the positions where
            # vertex_primary == 1, then gather those positions out of every
            # other list column so the per-particle lists stay aligned.
            particles = (
                particles.lazy()
                .with_columns(
                    _indices=pl.col('vertex_primary').list.eval(
                        (pl.element() == 1).arg_true()
                    )
                )
                .with_columns(
                    pl.exclude('event_id', '_indices').list.gather(pl.col('_indices'))
                )
                .drop('_indices')
                .sort('event_id')
            ).collect()

        particles_selected_ids = (
            particles.lazy().select('event_id', 'particle_id')
        ).collect()

        particles = add_orphan_mask(particles)
        particles = add_created_inside_calo_mask(particles)
        particles = add_particle_have_track_mask(particles, tracks)

    particles = add_eta_and_phi_and_pt(particles)
    print(f"[{tag} ETA PHI PT] RAM: {_rss_mb():.2f} MB")

    # Attach each track's originating-particle info (production vertex and
    # true pT) as extra track features.
    particle_info_lf = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'vx', 'vy', 'vz', 'pt'])
        .explode('particle_id', 'vx', 'vy', 'vz', 'pt')
        .rename({'pt': 'particle_pt'})
    )
    track_particle_cols = (
        tracks.lazy()
        .select(['event_id', 'majority_particle_id'])
        .with_columns(
            _local_order=pl.int_ranges(
                start=0, end=pl.col('majority_particle_id').list.len(), dtype=pl.UInt32)
        )
        .explode(['majority_particle_id', '_local_order'])
        .join(
            particle_info_lf,
            left_on=['event_id', 'majority_particle_id'],
            right_on=['event_id', 'particle_id'],
            how='left',
        )
        .sort(['event_id', '_local_order'])
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('vx').cast(pl.Float32),
            pl.col('vy').cast(pl.Float32),
            pl.col('vz').cast(pl.Float32),
            pl.col('particle_pt').cast(pl.Float32),
        ])
        .collect(streaming=True)
    )
    tracks = tracks.join(track_particle_cols, on='event_id', how='left')
    del track_particle_cols, particle_info_lf
    print(f"[{tag} TRACK<-PARTICLE INFO] RAM: {_rss_mb():.2f} MB")

    out: Dict[str, pl.DataFrame] = {
        'tracks': tracks,
        'calo_hits': calo_hits,
    }
    if is_primary:
        particles = get_particles_id_parent_of_inside_calo_particles_maskv3(particles, calo_hits)
        particles = set_target_particles_maskv4(
            particles,
            truth_eta_cut=cuts.truth_eta,
            truth_pt_cut=cuts.truth_pt,
            target_pt_cut=cuts.target_pt,
            tracks=tracks,
        )
        out['particles'] = particles
        out['particles_pid_to_vertex'] = particles_pid_to_vertex
        out['particles_selected_ids'] = particles_selected_ids
        print(f"[{tag} MASKS DONE] RAM: {_rss_mb():.2f} MB")
    else:
        # Pileup particles are not part of the dataset; release them. Keep only
        # the event-id list first: particles are the canonical record of which
        # pileup vertices exist, so vertices that produced no calorimeter hits
        # ("invisible" ones) still get sampled at their Poisson rate and
        # correctly contribute nothing.
        out['particle_event_ids'] = (
            particles.lazy().select('event_id').unique(maintain_order=True).collect()
        )
        del particles
        gc.collect()
    return out


def apply_cluster_energy_cutoff(calo_hits: pl.DataFrame, cluster_energy_cut: float) -> pl.DataFrame:
    """Drop hits belonging to sub-threshold clusters, and CLUE noise hits.

    A cluster is kept only if its total calibrated energy exceeds
    ``cluster_energy_cut``; hits CLUE left unclustered (``cluster_id < 0``) are
    always dropped.

    Args:
        calo_hits: clustered hits, one row per event with list columns.
        cluster_energy_cut: threshold in GeV.

    Returns:
        ``calo_hits`` with every list column filtered to the surviving hits,
        in their original within-event order.

    Note:
        Kept positions are computed on a narrow three-column side-frame and
        then applied with ``list.gather``, so the wide truth columns
        (``contrib_particle_ids``, ``contrib_energies``) are never exploded.
        An event whose clusters are all sub-threshold keeps its row with empty
        lists rather than vanishing from the frame.
    """
    keep_idx = (
        calo_hits.lazy()
        .select(['cluster_id', 'total_energy', 'detector'])
        .with_row_index('_rid')
        .with_columns(
            _pos=pl.int_ranges(0, pl.col('cluster_id').list.len(), dtype=pl.UInt32)
        )
        .explode(['cluster_id', 'total_energy', 'detector', '_pos'])
        .join(CALIBRATION.lazy().select(['detector', 'calib_factor']),
              on='detector', how='left')
        .with_columns(_cal_e=pl.col('total_energy') * pl.col('calib_factor'))
        .with_columns(_clu_sum=pl.col('_cal_e').sum().over(['_rid', 'cluster_id']))
        .filter((pl.col('_clu_sum') > cluster_energy_cut) & (pl.col('cluster_id') >= 0))
        .group_by('_rid', maintain_order=True)
        .agg(_indices=pl.col('_pos').sort())
        .select(['_rid', '_indices'])
    )

    return (
        calo_hits.lazy()
        .with_row_index('_rid')
        .join(keep_idx, on='_rid', how='left')
        .with_columns(
            pl.col('_indices').fill_null(pl.lit([], dtype=pl.List(pl.UInt32)))
        )
        .with_columns(
            pl.exclude('event_id', '_rid', '_indices').list.gather(pl.col('_indices'))
        )
        .drop(['_rid', '_indices'])
        .collect(streaming=True)
    )


def run_tail(
    prepared: Dict[str, pl.DataFrame],
    calo_hits: pl.DataFrame,
    tracks: pl.DataFrame,
    *,
    cuts: Cuts,
    clustering: ClusteringConfig,
) -> Dict[str, pl.DataFrame]:
    """Stage C: cluster, attribute energy to targets, emit the output tables.

    Identical for every mode -- the only thing that varies upstream is which
    hits and tracks arrive here.

    Args:
        prepared: the primary source's Stage-A output. ``particles``,
            ``particles_pid_to_vertex`` and ``particles_selected_ids`` are
            popped from it and progressively released.
        calo_hits: hits to cluster (overlaid ones in overlay mode).
        tracks: tracks to attach (overlaid ones in overlay mode).
        cuts: selection thresholds; only ``cluster_energy`` is used here.
        clustering: CLUE parameters and backend.

    Returns:
        The four output tables: ``target_particles``, ``calo_clusters``,
        ``tracks``, ``target_particles_deps``.
    """
    print("[CLUE CLUSTERING] Running CLUE clustering...")
    calo_hits = clue_clustering(
        calo_hits,
        dc=clustering.dc,
        rhoc=clustering.rhoc,
        dm=clustering.dm,
        ppbin=clustering.ppbin,
        backend=clustering.backend,
        deterministic=clustering.deterministic,
    )
    gc.collect()
    print(f"[CLUE CLUSTERING DONE] RAM: {_rss_mb():.2f} MB")

    calo_hits = apply_cluster_energy_cutoff(calo_hits, cuts.cluster_energy)
    print(f"[CLUSTER ENERGY CUTOFF DONE] RAM: {_rss_mb():.2f} MB")

    # pop() rather than index, so the dict stops holding these references and
    # the `del`s below actually free them.
    particles = prepared.pop('particles')
    particles_pid_to_vertex = prepared.pop('particles_pid_to_vertex')
    particles_selected_ids = prepared.pop('particles_selected_ids')

    # Which selected particles actually deposited energy anywhere.
    print("[DEPOSITORS LIST] Creating depositors list...")
    depositors_list = (
        calo_hits.lazy()
        .select(['event_id', 'contrib_particle_ids'])
        .explode('contrib_particle_ids')
        .explode('contrib_particle_ids')  # double explode: list[list]
        .rename({'contrib_particle_ids': 'particle_id'})
        .unique(subset=['event_id', 'particle_id'])
        .join(
            particles_selected_ids.lazy().select(['event_id', 'particle_id']).explode('particle_id'),
            on=['event_id', 'particle_id'],
            how='inner',
        )
        .select([
            pl.col('event_id'),
            pl.col('particle_id').cast(pl.Int64),
        ])
    ).collect(streaming=True)
    del particles_selected_ids
    gc.collect()
    print(f"[DEPOSITORS LIST DONE] RAM: {_rss_mb():.2f} MB")

    print("[TARGET PARTICLES AGG] Aggregating target particles...")
    target_particles = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_target_particle', 'pdg_id',
                 'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt', 'has_track', 'vertex_primary',
                 'vx', 'vy', 'vz'])
        .explode('particle_id', 'is_target_particle', 'pdg_id',
                 'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt',
                 'has_track', 'vertex_primary', 'vx', 'vy', 'vz')
        .filter(pl.col('is_target_particle'))
        .sort('event_id')
        .with_row_index('global_order')
        .sort('global_order')
        .drop('is_target_particle', 'global_order')
        .group_by('event_id', maintain_order=True)
        .agg('*')
        .collect(streaming=True)
    )
    print(f"[TARGET PARTICLES AGG DONE] RAM: {_rss_mb():.2f} MB")

    particles_for_backtrack = (
        particles.lazy()
        .select(pl.col('event_id'), pl.col('particle_id'),
                pl.col('parent_id'), pl.col('is_parent_missing'))
        .collect()
    )
    del particles
    gc.collect()

    print("[CREATE CALO CLUSTERS] Creating calo clusters...")
    calo_clusters = create_calo_clusters(calo_hits)
    print(f"[CREATE CALO CLUSTERS DONE] RAM: {_rss_mb():.2f} MB")

    print("[CLUSTER IDX MAPPING] Creating cluster to cluster index mapping...")
    cluster_to_cluster_idx = (
        calo_clusters.lazy()
        .select(['event_id', 'cluster_id'])
        .explode('cluster_id')
        .with_row_index('cluster_idx')
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('cluster_id'),
            (pl.col('cluster_idx') - pl.col('cluster_idx').min()).alias('cluster_idx'),
        ])
        .explode(['cluster_id', 'cluster_idx'])
        .collect()
    )

    # Walk each depositing particle up its parentage chain to the target
    # particle it belongs to, so deposits land on targets rather than on
    # secondaries produced inside the calorimeter.
    print("[BACKTRACK TO TARGET] Backtracking particles to target...")
    points_to_target = backtrack_to_target(
        particles=particles_for_backtrack,
        src_df=depositors_list,
        target_df=target_particles.select(['event_id', 'particle_id']).explode('particle_id'),
    )
    del particles_for_backtrack, depositors_list
    gc.collect()

    # The heavy contrib_particle_ids x contrib_energies double explode happens
    # once here; both consumers below reuse the result.
    print("[CONTRIB ENERGY] Building shared per-(event, cluster, particle) energies...")
    contrib_energy = cluster_contrib_energy(calo_hits_with_clusters=calo_hits)
    del calo_hits
    gc.collect()
    print(f"[CONTRIB ENERGY DONE] RAM: {_rss_mb():.2f} MB")

    print("[CLUSTER VERTEX DEPS] Aggregating calibrated energy by vertex_primary...")
    cluster_vertex_deps = cluster_vertex_primary_deps(
        contrib_energy=contrib_energy,
        pid_to_vertex=particles_pid_to_vertex,
        cluster_to_cluster_idx=cluster_to_cluster_idx,
    )
    del particles_pid_to_vertex
    gc.collect()

    calo_clusters = calo_clusters.join(cluster_vertex_deps, on='event_id', how='left')
    del cluster_vertex_deps
    gc.collect()

    print("[CLUSTER PURITY] Computing cluster purity...")
    target_particles_deps = cluster_purity(
        contrib_energy=contrib_energy,
        ancestors=points_to_target,
    )
    del contrib_energy
    gc.collect()
    print(f"[CLUSTER PURITY DONE] RAM: {_rss_mb():.2f} MB")

    print("[FILTER ORPHANS] Filtering orphan particles and reindexing...")
    filtered_data = filter_orphans_and_reindex(
        target_particles=target_particles,
        target_particles_deps=target_particles_deps,
        tracks=tracks,
        cluster_to_cluster_idx=cluster_to_cluster_idx,
    )
    print(f"[FILTER ORPHANS DONE] RAM: {_rss_mb():.2f} MB")
    print("[PREPROCESS COMPLETE]\n")

    return {
        'target_particles': filtered_data['target_particles'],
        'calo_clusters': calo_clusters,
        'tracks': filtered_data['tracks'],
        'target_particles_deps': filtered_data['target_particles_deps'],
    }


def preprocess_events(
    mode: str,
    *,
    particles: pl.DataFrame,
    tracks: pl.DataFrame,
    calo_hits: pl.DataFrame,
    cuts: Cuts,
    clustering: ClusteringConfig,
    keep_all_vertices: bool,
    pu_particles: Optional[pl.DataFrame] = None,
    pu_tracks: Optional[pl.DataFrame] = None,
    pu_calo_hits: Optional[pl.DataFrame] = None,
    overlay_cfg=None,
    seed: Optional[int] = None,
    num_of_events: int = -1,
) -> Dict[str, pl.DataFrame]:
    """Run the full pipeline over one batch of events.

    Args:
        mode: ``hard_scatter``, ``all_vertices``, or ``overlay``.
        particles, tracks, calo_hits: the primary (hard-scatter) source.
        cuts: selection thresholds.
        clustering: CLUE parameters and backend.
        keep_all_vertices: vertex policy for the primary source; derived from
            ``mode`` by :attr:`colliderml_pflow.config.Config.keep_all_vertices`.
        pu_particles, pu_tracks, pu_calo_hits: the pileup pool. Required for
            ``overlay``, ignored otherwise.
        overlay_cfg: :class:`colliderml_pflow.config.OverlayConfig`, required
            for ``overlay``.
        seed: overrides ``overlay_cfg.seed`` for this batch. The runner passes
            ``seed + shard_index`` so each shard samples differently.
        num_of_events: optional cap on primary events.

    Returns:
        The four output tables.
    """
    tof = overlay_cfg.tof if (overlay_cfg is not None and mode == "overlay") else None

    primary = prepare_source(
        particles, tracks, calo_hits,
        role="primary",
        keep_all_vertices=keep_all_vertices,
        cuts=cuts,
        tof=tof,
        num_of_events=num_of_events,
    )
    del particles, tracks, calo_hits
    gc.collect()

    if mode != "overlay":
        out_tracks = primary.pop('tracks')
        out_hits = primary.pop('calo_hits')
        return run_tail(primary, out_hits, out_tracks, cuts=cuts, clustering=clustering)

    # Imported here so non-overlay runs never touch the overlay module.
    from colliderml_pflow.overlay import run_overlay

    if pu_particles is None or pu_tracks is None or pu_calo_hits is None:
        raise ValueError("overlay mode requires pu_particles, pu_tracks and pu_calo_hits")
    if overlay_cfg is None:
        raise ValueError("overlay mode requires overlay_cfg")

    pileup = prepare_source(
        pu_particles, pu_tracks, pu_calo_hits,
        role="pileup",
        keep_all_vertices=False,  # never applied to the pileup role
        cuts=cuts,
        tof=tof,
    )
    del pu_particles, pu_tracks, pu_calo_hits
    gc.collect()
    print(f"[PER-SOURCE DONE] RAM: {_rss_mb():.2f} MB")

    merged_hits, merged_tracks = run_overlay(
        primary, pileup, overlay_cfg,
        seed=overlay_cfg.seed if seed is None else seed,
    )
    return run_tail(primary, merged_hits, merged_tracks, cuts=cuts, clustering=clustering)
