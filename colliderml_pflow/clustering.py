"""Calorimeter clustering: voxelisation followed by CLUE.

Two steps, both operating on the per-event list-valued ``calo_hits`` frame:

1. :func:`voxelize_hits` bins hits onto a per-subsystem spatial grid
   (see :data:`colliderml_pflow.calibration.voxel_config`) and sums the energy
   in each occupied voxel. This suppresses the cell-level granularity
   difference between ECal and HCal before clustering.
2. :func:`clue_clustering` runs CLUE (CLUstering of Energy) over the voxel
   centres of each event, then maps the resulting cluster ids back onto the
   original, un-voxelised hits via the shared voxel index.

The mapping in step 2 is what makes the output usable: clusters are found on
the coarse voxel grid, but every original hit -- and therefore every truth
``contrib_particle_ids`` entry -- ends up carrying a cluster id.

Ported from ``primary/clue_clustering.py`` and ``primary/downsample.py`` on
``master``. Two changes: the ``sys.path`` hack that pointed at a source
checkout of CLUEstering is gone (it is installed as a package), and the CLUE
parameters are passed in from configuration rather than baked into default
arguments.
"""

import os

import numpy as np
import polars as pl
import tqdm

from colliderml_pflow.calibration import CALIBRATION, voxel_config

# Tuned on the ColliderML ODD sample; see configs/*.yaml, which is where the
# values actually used for a run come from. These defaults exist so the
# function is usable standalone.
DEFAULT_CLUE_PARAMS = {
    "dc": 75.88106168184893,
    "rhoc": 104.34315216716726,
    "dm": 87.0967630118376,
    "ppbin": 16,
}

# Per-worker clusterer cache (one clusterer per worker process).
_worker_clusterer = None


def voxelize_hits(calo_hits: pl.DataFrame, deterministic: bool = True) -> pl.DataFrame:
    """Bin calorimeter hits onto the per-subsystem voxel grid.

    Args:
        calo_hits: one row per event, with list columns ``detector``,
            ``total_energy``, ``x``, ``y``, ``z``.
        deterministic: keep voxel row order stable. Polars' ``group_by`` is
            hash-based and returns groups in an arbitrary order unless asked
            otherwise, and that order becomes the order CLUE sees its points
            in -- which changes the cluster labelling it produces. See
            :func:`clue_clustering`.

    Returns:
        One row per event, with list columns holding one entry per *occupied
        voxel*: summed ``total_energy``, the mean ``x``/``y``/``z`` of the hits
        that fell in it, the grid ``idx_x``/``idx_y``/``idx_z``, the voxel
        ``v_size``, and ``hit_count``.
    """
    return (
        calo_hits.lazy()
        # Reduce data volume before exploding.
        .select(["event_id", "detector", "x", "y", "z", "total_energy"])
        .explode(["detector", "total_energy", "x", "y", "z"])
        # Attach the per-subsystem voxel size.
        .join(voxel_config.lazy(), on="detector", how="inner")
        # Integer voxel indices: Int32 grouping is far faster than Float64.
        .with_columns([
            (pl.col("x") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_x"),
            (pl.col("y") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_y"),
            (pl.col("z") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_z"),
        ])
        .group_by(["event_id", "detector", "idx_x", "idx_y", "idx_z"],
                  maintain_order=deterministic)
        .agg([
            pl.col("total_energy").sum(),
            pl.col("x").mean().alias("x"),
            pl.col("y").mean().alias("y"),
            pl.col("z").mean().alias("z"),
            pl.col("v_size").first(),
            pl.col("x").len().alias("hit_count"),
        ])
        .group_by(["event_id"], maintain_order=deterministic)
        .agg("*")
        .sort("event_id", maintain_order=deterministic)
        .collect()
    )


def _worker_init(omp_threads: int):
    """Pool initializer: cap each worker's OpenMP fanout to avoid oversubscription."""
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)


def _cluster_one_event(args):
    """Cluster one event's points; returns ``(idx, cluster_ids, cx, cy, cz)``."""
    idx, points, dc, rhoc, dm, ppbin, backend = args
    global _worker_clusterer
    if _worker_clusterer is None:
        # Import inside the worker: CLUEstering initialises CUDA on import, and
        # doing that in the parent would poison forked children.
        import CLUEstering as _clue
        _worker_clusterer = _clue.clusterer(dc=dc, rhoc=rhoc, dm=dm, ppbin=ppbin)

    clusterer = _worker_clusterer
    clusterer.read_data(points.T)
    clusterer.run_clue(backend=backend, verbose=False, block_size=1024)

    c_ids = clusterer.cluster_ids
    centroids = clusterer.cluster_centroids()

    n_points = len(c_ids)
    cx = np.full(n_points, float("inf"), dtype=np.float32)
    cy = np.full(n_points, float("inf"), dtype=np.float32)
    cz = np.full(n_points, float("inf"), dtype=np.float32)
    valid_mask = c_ids != -1
    if np.any(valid_mask):
        valid_ids = c_ids[valid_mask]
        cx[valid_mask] = centroids[valid_ids, 0]
        cy[valid_mask] = centroids[valid_ids, 1]
        cz[valid_mask] = centroids[valid_ids, 2]
    return idx, c_ids, cx, cy, cz


def clue_clustering(
    calo_hits: pl.DataFrame,
    dc: float = DEFAULT_CLUE_PARAMS["dc"],
    rhoc: float = DEFAULT_CLUE_PARAMS["rhoc"],
    dm: float = DEFAULT_CLUE_PARAMS["dm"],
    ppbin: int = DEFAULT_CLUE_PARAMS["ppbin"],
    backend: str = "gpu cuda",
    deterministic: bool = True,
) -> pl.DataFrame:
    """Run CLUE over each event's voxelised hits and label the original hits.

    Args:
        calo_hits: one row per event, list columns ``detector``,
            ``total_energy``, ``x``, ``y``, ``z`` (plus any truth columns,
            which are passed through untouched).
        dc: critical distance defining a point's local-density neighbourhood.
        rhoc: minimum local density for a point to seed a cluster.
        dm: maximum distance over which a point may be assigned to a seed.
        ppbin: target points per spatial bin in CLUE's internal tiling.
        backend: CLUEstering backend -- ``'gpu cuda'``, ``'cpu serial'``,
            ``'cpu tbb'``, or ``'cpu omp'``. Only ``'cpu omp'`` uses the
            multiprocessing path below; the rest reuse one clusterer serially.
        deterministic: pin the order in which points reach CLUE, so repeated
            runs over the same input give the same clusters.

    Returns:
        ``calo_hits`` with four extra list columns aligned to the original
        hits: ``cluster_id`` (-1 for unclustered noise) and the cluster
        centroid ``cluster_cx`` / ``cluster_cy`` / ``cluster_cz``.

    Note:
        CLUE's output depends on the order its input points arrive in: cluster
        ids are handed out in discovery order, and order also breaks ties in
        the density/assignment steps. The ``master`` branch left the upstream
        ``group_by`` and ``sort`` unordered, so two runs over identical input
        produced different cluster labels and around 0.03% different cluster
        counts. ``deterministic=True`` (the default here) fixes that order and
        makes a run reproducible. Set it to ``False`` to reproduce the original
        unordered behaviour.

        Results also depend on the backend, so comparisons between runs must
        hold ``backend`` fixed too.
    """
    voxel_hits = voxelize_hits(calo_hits=calo_hits, deterministic=deterministic)
    calo = calo_hits
    calo_voxel = voxel_hits

    # Flat table of voxel centres, energy-calibrated and cast for CLUE.
    data_flat = (
        calo_voxel.lazy()
        .select(["event_id", "x", "y", "z", "total_energy", "detector"])
        .explode(["x", "y", "z", "total_energy", "detector"])
        .join(CALIBRATION.lazy(), on="detector")
        .with_columns([
            (pl.col("total_energy") * pl.col("calib_factor") * 1000).alias("energy"),
            pl.col("x").cast(pl.Float32),
            pl.col("y").cast(pl.Float32),
            pl.col("z").cast(pl.Float32),
        ])
        .select(["event_id", "x", "y", "z", "energy", "detector"])
        # Unstable by default, which would reshuffle rows within an event and
        # hand CLUE its points in a different order on every run.
        .sort("event_id", maintain_order=deterministic)
        .collect()
    )

    # Split into one (n_points, 4) array per event.
    event_counts = data_flat.group_by("event_id", maintain_order=True).count()["count"].to_numpy()
    all_points = data_flat.select(["x", "y", "z", "energy"]).to_numpy().astype(np.float32)
    split_indices = np.cumsum(event_counts)[:-1]
    points_list = np.split(all_points, split_indices)

    n_events = len(points_list)
    res_cluster_ids = [None] * n_events
    res_cx = [None] * n_events
    res_cy = [None] * n_events
    res_cz = [None] * n_events

    use_parallel = backend.strip().lower() == "cpu omp" and n_events > 1
    if use_parallel:
        import multiprocessing as mp
        n_cores = os.cpu_count() or 1
        n_workers = max(1, n_cores // 2)
        n_workers = min(n_workers, n_events)
        omp_threads = max(1, n_cores // max(n_workers, 1))
        print(f"[CLUE PARALLEL] cpu omp on {n_workers} workers x {omp_threads} OMP threads "
              f"(cores={n_cores}, events={n_events})")

        args_iter = ((i, points_list[i], dc, rhoc, dm, ppbin, backend) for i in range(n_events))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers,
                      initializer=_worker_init,
                      initargs=(omp_threads,)) as pool:
            for idx, c_ids, cx, cy, cz in tqdm.tqdm(
                    pool.imap_unordered(_cluster_one_event, args_iter, chunksize=1),
                    total=n_events, desc="Clustering events (parallel)"):
                res_cluster_ids[idx] = c_ids
                res_cx[idx] = cx
                res_cy[idx] = cy
                res_cz[idx] = cz
    else:
        # Sequential: reuse one clusterer across events. Imported lazily so CUDA
        # is not initialised at module-import time in the parent process.
        import CLUEstering as clue
        clusterer = clue.clusterer(dc=dc, rhoc=rhoc, dm=dm, ppbin=ppbin)
        for i, points in enumerate(tqdm.tqdm(points_list, desc="Clustering events")):
            clusterer.read_data(points.T)
            clusterer.run_clue(backend=backend, verbose=False, block_size=1024)

            c_ids = clusterer.cluster_ids
            centroids = clusterer.cluster_centroids()

            n_points = len(c_ids)
            cx = np.full(n_points, float("inf"), dtype=np.float32)
            cy = np.full(n_points, float("inf"), dtype=np.float32)
            cz = np.full(n_points, float("inf"), dtype=np.float32)
            valid_mask = c_ids != -1
            if np.any(valid_mask):
                valid_ids = c_ids[valid_mask]
                cx[valid_mask] = centroids[valid_ids, 0]
                cy[valid_mask] = centroids[valid_ids, 1]
                cz[valid_mask] = centroids[valid_ids, 2]

            res_cluster_ids[i] = c_ids
            res_cx[i] = cx
            res_cy[i] = cy
            res_cz[i] = cz

    final_ids = np.concatenate(res_cluster_ids)
    final_cx = np.concatenate(res_cx)
    final_cy = np.concatenate(res_cy)
    final_cz = np.concatenate(res_cz)

    data_clustered = data_flat.with_columns([
        pl.Series("cluster_id", final_ids),
        pl.Series("cluster_cx", final_cx),
        pl.Series("cluster_cy", final_cy),
        pl.Series("cluster_cz", final_cz),
    ])

    # Map voxel-level cluster labels back onto the original hits by recomputing
    # each hit's voxel index and joining on (event, voxel, detector).
    aggrigated = (
        (calo.lazy().select('event_id', 'x', 'y', 'z', 'detector')
             .explode(['x', 'y', 'z', 'detector'])
             .join(voxel_config.lazy(), on='detector')
             .with_columns([
                 (pl.col("x") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_x"),
                 (pl.col("y") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_y"),
                 (pl.col("z") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_z"),
             ]))
        .join(
            (data_clustered.lazy()
             .select('event_id', 'x', 'y', 'z', 'detector', 'cluster_id',
                     'cluster_cx', 'cluster_cy', 'cluster_cz')
             .join(voxel_config.lazy(), on='detector')
             .with_columns([
                 (pl.col("x") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_x"),
                 (pl.col("y") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_y"),
                 (pl.col("z") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_z"),
             ])),
            on=['event_id', 'idx_x', 'idx_y', 'idx_z', 'detector'], how='inner',
        )
        .select(['event_id', 'cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz'])
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('cluster_id'),
            pl.col('cluster_cx'),
            pl.col('cluster_cy'),
            pl.col('cluster_cz'),
        ])
        .sort('event_id')
    ).collect()

    return calo.join(aggrigated, on='event_id', how='left')
