"""Cluster feature construction and final orphan filtering.

Two Stage-C steps that are independent of which mode produced the hits:

- :func:`create_calo_clusters` turns labelled calorimeter hits into the
  per-cluster feature table the network consumes.
- :func:`filter_orphans_and_reindex` drops target particles that ended up with
  neither a track nor a calorimeter deposit, then renumbers particles and
  clusters so the incidence table refers to compact per-event indices.

Both are ported from ``create_training_dataset_pileup_overlay.py`` on
``master``, which holds the versions that work for every mode:

- its ``filter_orphans_and_reindex`` has a *guarded* pileup-track branch that
  falls back to the original behaviour when ``source_pileup_event_id`` is
  absent, so one implementation serves overlay and non-overlay runs alike;
- its ``create_calo_clusters`` is already free of the ``cluster_time`` branch,
  which this branch drops in every mode (see the README).
"""

from typing import Dict

import polars as pl

from colliderml_pflow.calibration import CALIBRATION


def create_calo_clusters(calo_hits: pl.DataFrame) -> pl.DataFrame:
    """Build the per-cluster feature table from CLUE-labelled hits.

    Args:
        calo_hits: one row per event, with list columns ``cluster_id``,
            ``cluster_cx``/``cy``/``cz``, ``detector``, ``total_energy``,
            ``x``, ``y``, ``z``. Hits with ``cluster_id < 0`` (CLUE noise) are
            ignored.

    Returns:
        One row per event, list columns indexed by cluster: ``cluster_id``,
        ``total_cluster_energy``, ``hcal_energy``, ``hcal_fraction``,
        ``sigma_eta``/``sigma_phi``/``sigma_rho`` (hit spread within the
        cluster), ``number_of_hits``, ``energy_hits_std``, ``max_hit_energy``,
        and the centroid position as ``cluster_phi``/``cluster_eta``/
        ``cluster_rho``.

    Note:
        Energies are calibrated to GeV via
        :data:`colliderml_pflow.calibration.CALIBRATION` before aggregation.
        No ``cluster_time`` column is produced -- see the README for why and
        for the downstream consequences.
    """
    # Pre-compute 'is_hcal' so the per-hit aggregation below is a single pass.
    calib_optimized = CALIBRATION.lazy().select([
        pl.col('detector'),
        pl.col('calib_factor'),
        pl.col('system_label').str.contains("Hcal").fill_null(False).alias('is_hcal')
    ])

    # --- BRANCH A: CENTROIDS & GEOMETRY ---
    centroids_df = (
        calo_hits.lazy()
        .select(['event_id', 'cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz'])
        .explode(['cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz'])
        # Deduplicate: every hit in a cluster carries the same centroid.
        .filter(pl.col('cluster_id') >= 0)
        .group_by(['event_id', 'cluster_id'])
        .agg([
            pl.col('cluster_cx').first(),
            pl.col('cluster_cy').first(),
            pl.col('cluster_cz').first(),
        ])
        # Angles, computed once per cluster.
        .with_columns([
            pl.arctan2(pl.col('cluster_cy'), pl.col('cluster_cx')).alias('cluster_phi'),
            # Cluster eta: arcsinh(z / r_perp)
            (pl.col('cluster_cz') / (pl.col('cluster_cx').pow(2) + pl.col('cluster_cy').pow(2)).sqrt())
            .arcsinh()
            .alias('cluster_eta'),
            (pl.col('cluster_cx').pow(2) + pl.col('cluster_cy').pow(2)).sqrt().alias('cluster_rho'),
        ])
        .drop(['cluster_cx', 'cluster_cy', 'cluster_cz'])
    )

    # --- BRANCH B: HIT PHYSICS & TOPOLOGY ---
    physics_df = (
        calo_hits.lazy()
        .select(['event_id', 'cluster_id', 'detector', 'total_energy', 'x', 'y', 'z'])
        .explode(['cluster_id', 'detector', 'total_energy', 'x', 'y', 'z'])
        .join(calib_optimized, on='detector', how='left')
        .with_columns([
            (pl.col('total_energy') * pl.col('calib_factor')).alias('cal_E'),
            (pl.col('x').pow(2) + pl.col('y').pow(2)).sqrt().alias('hit_rho')
        ])
        .with_columns([
            (pl.col('z') / pl.col('hit_rho')).arcsinh().alias('hit_eta'),
            pl.arctan2(pl.col('y'), pl.col('x')).alias('hit_phi'),
        ])
        .group_by(['event_id', 'cluster_id'])
        .agg([
            pl.col('cal_E').sum().alias('total_cluster_energy'),
            pl.col('cal_E').filter(pl.col('is_hcal')).sum().alias('hcal_energy'),

            # Topological widths
            pl.col('hit_eta').std().fill_null(0.0).alias('sigma_eta'),
            pl.col('hit_phi').std().fill_null(0.0).alias('sigma_phi'),
            pl.col('hit_rho').std().fill_null(0.0).alias('sigma_rho'),

            # Hit-level features
            pl.col('cal_E').count().alias('number_of_hits'),
            pl.col('cal_E').std().fill_null(0.0).alias('energy_hits_std'),
            pl.col('cal_E').max().alias('max_hit_energy'),
        ])
    )

    # --- FINAL MERGE ---
    calo_clusters = (
        physics_df
        .join(
            centroids_df,
            on=['event_id', 'cluster_id'],
            how='left'
        )
        .with_columns(
            (pl.col('hcal_energy') / pl.col('total_cluster_energy'))
            .fill_nan(0.0)
            .alias('hcal_fraction'),
        )
        .sort(['event_id', 'cluster_id'])
        .group_by('event_id', maintain_order=True)
        .agg(pl.all())
        .collect(streaming=True)
    )

    return calo_clusters


def filter_orphans_and_reindex(
    target_particles: pl.DataFrame,
    target_particles_deps: pl.DataFrame,
    tracks: pl.DataFrame,
    cluster_to_cluster_idx: pl.DataFrame,
) -> Dict[str, pl.DataFrame]:
    """Drop unreconstructable target particles and renumber what remains.

    A target particle is an *orphan* if it has neither a reconstructed track
    nor any calorimeter deposit -- nothing in the detector records it, so the
    network cannot be asked to find it. Orphans are removed, and the surviving
    particles are renumbered to a dense per-event ``particle_idx`` that the
    incidence table and the track table then refer to.

    Args:
        target_particles: one row per event, list columns over target particles.
        target_particles_deps: per-(event, cluster, ancestor) deposited energy.
        tracks: one row per event, list columns over tracks. If a
            ``source_pileup_event_id`` column is present (overlay output), it
            drives the pileup-track handling described below.
        cluster_to_cluster_idx: map from ``cluster_id`` to the dense
            per-event ``cluster_idx``.

    Returns:
        Dict with ``target_particles`` (filtered, with a fresh ``particle_idx``),
        ``target_particles_deps`` (re-keyed to ``particle_idx``/``cluster_idx``),
        and ``tracks`` (``majority_particle_id`` replaced by ``particle_idx``).

    Note:
        Prints before/after particle counts and the fraction of target energy
        carried by the removed orphans.
    """
    # 1. Initial statistics.
    initial_stats = (
        target_particles.lazy()
        .select(pl.col('energy'))
        .explode('energy')
        .select([
            pl.count().alias('count'),
            pl.sum('energy').alias('total_energy')
        ])
        .collect()
    )
    total_particles_before = initial_stats['count'][0]
    total_energy_before = initial_stats['total_energy'][0]

    # 2. Identify valid particles: those appearing in deps or in tracks.
    ids_in_deps = (
        target_particles_deps.lazy()
        .select(['event_id', 'ultimate_ancestor_id'])
        .rename({'ultimate_ancestor_id': 'particle_id'})
        .filter(pl.col('particle_id').is_not_null())
        .unique()
    )

    ids_in_tracks = (
        tracks.lazy()
        .select(['event_id', 'majority_particle_id'])
        .explode('majority_particle_id')
        .rename({'majority_particle_id': 'particle_id'})
        .unique()
    )

    valid_ids = (
        pl.concat([ids_in_deps, ids_in_tracks])
        .unique()
    )

    # 3. Filter target_particles, preserving the original in-event order.
    tp_cols = [c for c in target_particles.columns if c != 'event_id']
    target_particles_filtered = (
        target_particles.lazy()
        .with_columns(
            pl.int_ranges(0, pl.col('particle_id').list.len()).alias('_orig_idx')
        )
        .explode(['_orig_idx'] + tp_cols)
        .join(
            valid_ids,
            on=['event_id', 'particle_id'],
            how='inner'
        )
        .sort(['event_id', '_orig_idx'])
        .drop('_orig_idx')
        .group_by('event_id', maintain_order=True)
        .agg(pl.all())
        .with_columns(
            pl.int_ranges(0, pl.col('particle_id').list.len(), dtype=pl.UInt32).alias('particle_idx')
        )
        .collect(streaming=True)
    )

    # 4. Final statistics and report.
    final_stats = (
        target_particles_filtered.lazy()
        .select(pl.col('energy'))
        .explode('energy')
        .select([
            pl.count().alias('count'),
            pl.sum('energy').alias('total_energy')
        ])
        .collect()
    )
    total_particles_after = final_stats['count'][0]
    total_energy_after = final_stats['total_energy'][0]

    orphans_count = total_particles_before - total_particles_after

    val_energy_percentage = 0.0
    if total_energy_before > 0:
        val_energy_percentage = total_energy_after / total_energy_before

    orphan_energy_percentage = 1.0 - val_energy_percentage

    print(f"--- Orphan Filtering Stats ---")
    print(f"Total target particles before: {total_particles_before}")
    print(f"Total target particles after:  {total_particles_after}")
    print(f"Orphans removed:               {orphans_count}")
    print(f"Total Energy before:           {total_energy_before:.4f}")
    print(f"Total Energy after:            {total_energy_after:.4f}")
    print(f"Orphan Energy Percentage:      {orphan_energy_percentage:.2%}")
    print(f"------------------------------")

    # 5. particle_id -> particle_idx map over the surviving particles.
    particle_mapping = (
        target_particles_filtered.lazy()
        .select(['event_id', 'particle_id'])
        .explode('particle_id')
        .with_row_index('particle_idx')
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('particle_id'),
            (pl.col('particle_idx') - pl.col('particle_idx').min()).alias('particle_idx')
        ])
        .explode(['particle_id', 'particle_idx'])
    )

    # 6. Re-key the incidence table onto particle_idx / cluster_idx.
    target_particles_deps_aggrigated = (
        target_particles_deps.lazy()
        .select(['event_id', 'cluster_id', 'ultimate_ancestor_id', 'total_energy_deps_in_cluster'])
        .rename({'ultimate_ancestor_id': 'particle_id'})
        .filter(pl.col("particle_id").is_not_null())
        .join(
            particle_mapping,
            on=['event_id', 'particle_id'],
            how='inner'
        )
        .join(
            cluster_to_cluster_idx.lazy(),
            on=['event_id', 'cluster_id'],
            how='left'
        )
        .sort(['event_id', 'cluster_id'])
        .drop('particle_id', 'cluster_id')
        .group_by('event_id', maintain_order=True)
        .agg('*')
    ).collect(streaming=True)

    # 7. Update tracks: replace particle_id with particle_idx.
    #
    # `particle_id` is event-local and reused across source events. After
    # overlay, HS and PU tracks share the HS event_id, so a pileup track whose
    # majority_particle_id happens to collide with a hard-scatter target
    # particle's id would join to a valid HS particle_idx and get spuriously
    # wired to that particle (~45% of incidence links in practice). When
    # `source_pileup_event_id` is present and non-null -- overlay output, pileup
    # rows only -- force those tracks to the -1 sentinel, so `particle_idx >= 0`
    # becomes a clean "this is a hard-scatter track" flag.
    #
    # Guarded on column presence, so non-overlay callers get the original
    # behaviour unchanged.
    has_pu_src = 'source_pileup_event_id' in tracks.columns
    mapping_cols = ['event_id', 'majority_particle_id']
    explode_cols = ['majority_particle_id', 'local_order']
    if has_pu_src:
        mapping_cols.append('source_pileup_event_id')
        explode_cols.append('source_pileup_event_id')

    if has_pu_src:
        particle_idx_expr = (
            pl.when(pl.col('source_pileup_event_id').is_not_null())
            .then(pl.lit(-1))
            .otherwise(pl.col('particle_idx').fill_null(-1))
            .alias('particle_idx')
        )
    else:
        # Mark tracks that lost their particle mapping (orphans) with -1.
        particle_idx_expr = pl.col('particle_idx').fill_null(-1)

    tracks_mappings = (
        tracks.lazy()
        .select(mapping_cols)
        .with_columns(
            local_order=pl.int_ranges(
                start=0,
                end=pl.col('majority_particle_id').list.len(),
                dtype=pl.UInt32
            )
        )
        .explode(explode_cols)
        .rename({'majority_particle_id': 'particle_id'})
        .join(
            particle_mapping,
            on=['event_id', 'particle_id'],
            how='left'
        )
        .with_columns(particle_idx_expr)
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('particle_idx').sort_by('local_order'),
            pl.col('particle_id').sort_by('local_order'),
        ])
    )

    tracks_updated = (
        tracks.lazy()
        .drop('majority_particle_id')
        .join(
            tracks_mappings,
            on='event_id',
            how='inner'
        )
        .sort('event_id')
        .collect(streaming=True)
    )

    return {
        "target_particles": target_particles_filtered,
        "target_particles_deps": target_particles_deps_aggrigated,
        "tracks": tracks_updated,
    }
