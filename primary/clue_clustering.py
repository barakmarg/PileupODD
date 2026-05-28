import numpy as np
import os
import polars as pl
from primary.calibration import CALIBRATION
from primary.downsample import voxel_config, voxelize_hits
import tqdm


# Per-worker clusterer cache (one clusterer per worker process).
_worker_clusterer = None


def _worker_init(omp_threads: int):
    """Pool initializer: limit each worker's OpenMP fanout to avoid oversubscription."""
    import os
    os.environ['OMP_NUM_THREADS'] = str(omp_threads)
    os.environ['MKL_NUM_THREADS'] = str(omp_threads)


def _cluster_one_event(args):
    """Cluster one event's points; returns (idx, c_ids, cx, cy, cz)."""
    idx, points, dc, rhoc, dm, ppbin, backend = args
    global _worker_clusterer
    if _worker_clusterer is None:
        # Local import inside worker so spawn-start workers pick up sys.path edits.
        import sys
        if '/storage/agrp/barakma/CLUEstering' not in sys.path:
            sys.path.insert(0, '/storage/agrp/barakma/CLUEstering')
        import CLUEstering as _clue
        _worker_clusterer = _clue.clusterer(dc=dc, rhoc=rhoc, dm=dm, ppbin=ppbin)

    clusterer = _worker_clusterer
    clusterer.read_data(points.T)
    clusterer.run_clue(backend=backend, verbose=False, block_size=1024)

    c_ids = clusterer.cluster_ids
    centroids = clusterer.cluster_centroids()

    n_points = len(c_ids)
    cx = np.full(n_points, float('inf'), dtype=np.float32)
    cy = np.full(n_points, float('inf'), dtype=np.float32)
    cz = np.full(n_points, float('inf'), dtype=np.float32)
    valid_mask = c_ids != -1
    if np.any(valid_mask):
        valid_ids = c_ids[valid_mask]
        cx[valid_mask] = centroids[valid_ids, 0]
        cy[valid_mask] = centroids[valid_ids, 1]
        cz[valid_mask] = centroids[valid_ids, 2]
    return idx, c_ids, cx, cy, cz


def clue_clustering(calo_hits: pl.DataFrame, dc=75.88106168184893, rhoc=104.34315216716726, dm=87.0967630118376, ppbin=16, backend='gpu cuda') -> pl.DataFrame:
    # --------------------------------------------------------------------------------
    # 1. SETUP & DEFINITIONS
    # --------------------------------------------------------------------------------
    voxel_hits = voxelize_hits(calo_hits=calo_hits)
    calo = calo_hits
    calo_voxel = voxel_hits
    # --------------------------------------------------------------------------------
    # 2. DATA PROCESSING (Vectorized)
    # --------------------------------------------------------------------------------
    # Filter and Explode early to create a flat table of all hits
    data_flat = (
        calo_voxel.lazy()
        .select(['event_id', 'x', 'y', 'z', 'total_energy', 'detector'])
        .explode(['x', 'y', 'z', 'total_energy', 'detector'])
        .join(CALIBRATION.lazy(), on='detector')
        .with_columns([
            (pl.col('total_energy') * pl.col('calib_factor') * 1000).alias('energy'),
            pl.col('x').cast(pl.Float32),
            pl.col('y').cast(pl.Float32),
            pl.col('z').cast(pl.Float32)
        ])
        .select(['event_id', 'x', 'y', 'z', 'energy', 'detector'])
        .sort('event_id')
        .collect()
    )

    # Prepare Numpy splits (same as previous optimization)
    event_counts = data_flat.group_by('event_id', maintain_order=True).count()['count'].to_numpy()
    all_points = data_flat.select(['x', 'y', 'z', 'energy']).to_numpy().astype(np.float32)
    split_indices = np.cumsum(event_counts)[:-1]
    points_list = np.split(all_points, split_indices)

    # --------------------------------------------------------------------------------
    # 3. CLUSTERING & CENTROID MAPPING
    # --------------------------------------------------------------------------------
    # Best Parameters:
    #   dc        : 75.88106168184893
    #   rhoc      : 104.34315216716726
    #   dm        : 87.0967630118376
    #   ppbin     : 16
    # Lists to store results for all events (indexed by event order).
    n_events = len(points_list)
    res_cluster_ids = [None] * n_events
    res_cx = [None] * n_events
    res_cy = [None] * n_events
    res_cz = [None] * n_events

    use_parallel = backend.strip().lower() == 'cpu omp' and n_events > 1
    if use_parallel:
        import multiprocessing as mp
        n_cores = os.cpu_count() or 1
        n_workers = max(1, n_cores // 2)
        n_workers = min(n_workers, n_events)
        omp_threads = max(1, n_cores // max(n_workers, 1))
        print(f"[CLUE PARALLEL] cpu omp on {n_workers} workers x {omp_threads} OMP threads "
              f"(cores={n_cores}, events={n_events})")

        args_iter = ((i, points_list[i], dc, rhoc, dm, ppbin, backend) for i in range(n_events))
        ctx = mp.get_context('spawn')
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
        # Sequential path: reuse a single clusterer for all events.
        # Import lazily so CUDA is not initialized in the parent at module
        # import time — that would poison the child after fork().
        import CLUEstering as clue
        clusterer = clue.clusterer(dc=dc, rhoc=rhoc, dm=dm, ppbin=ppbin)
        for i, points in enumerate(tqdm.tqdm(points_list, desc="Clustering events")):
            clusterer.read_data(points.T)
            clusterer.run_clue(backend=backend, verbose=False, block_size=1024)

            c_ids = clusterer.cluster_ids
            centroids = clusterer.cluster_centroids()

            n_points = len(c_ids)
            cx = np.full(n_points, float('inf'), dtype=np.float32)
            cy = np.full(n_points, float('inf'), dtype=np.float32)
            cz = np.full(n_points, float('inf'), dtype=np.float32)
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

    # --------------------------------------------------------------------------------
    # 4. MERGE RESULTS BACK TO POLARS
    # --------------------------------------------------------------------------------

    # Concatenate all event results into long 1D arrays
    final_ids = np.concatenate(res_cluster_ids)
    final_cx = np.concatenate(res_cx)
    final_cy = np.concatenate(res_cy)
    final_cz = np.concatenate(res_cz)

    # Add columns to the flat dataframe
    data_clustered = data_flat.with_columns([
        pl.Series("cluster_id", final_ids),
        pl.Series("cluster_cx", final_cx),
        pl.Series("cluster_cy", final_cy),
        pl.Series("cluster_cz", final_cz)
    ])

    aggrigated = ((calo.lazy().select('event_id', 'x', 'y', 'z', 'detector')
                            .explode(['x', 'y', 'z', 'detector'])
                            .join(voxel_config.lazy(), on='detector')
                            .with_columns([
                                (pl.col("x") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_x"),
                                (pl.col("y") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_y"),
                                (pl.col("z") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_z"),  ]))
                    .join(
                            (data_clustered.lazy() 
                        .select('event_id', 'x', 'y', 'z', 'detector', 'cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz')
                        .join(voxel_config.lazy(), on='detector')
                        .with_columns([
                            (pl.col("x") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_x"),
                            (pl.col("y") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_y"),
                            (pl.col("z") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_z"),  ])),
                                on=['event_id', 'idx_x', 'idx_y', 'idx_z', 'detector'], how='inner'
                    )

                        .select(['event_id','cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz'])
                        .group_by('event_id', maintain_order=True)
                            .agg([
                                pl.col('cluster_id'),
                                pl.col('cluster_cx'),
                                pl.col('cluster_cy'),
                                pl.col('cluster_cz')
                            ])
                            .sort('event_id')
                        ).collect()
                                
                                


    # --------------------------------------------------------------------------------
    # 5. (Optional) GROUP BACK TO LISTS
    # If you need the original structure (one row per event_id, with lists inside)
    # --------------------------------------------------------------------------------

    # add it to calo as columns
    calo = calo.join(aggrigated, on='event_id', how='left')
    return calo