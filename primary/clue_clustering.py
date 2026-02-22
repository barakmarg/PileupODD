import numpy as np
import polars as pl
from primary.calibration import CALIBRATION
import CLUEstering as clue
from primary.downsample import voxel_config, voxelize_hits
import tqdm

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
    clusterer = clue.clusterer(dc=dc, rhoc=rhoc, dm=dm, ppbin=ppbin)

    # Lists to store results for all events
    res_cluster_ids = []
    res_cx = []
    res_cy = []
    res_cz = []

    for points in tqdm.tqdm(points_list, desc="Clustering events"):
        # 1. Run CLUE
        clusterer.read_data(points.T)
        clusterer.run_clue(backend=backend, verbose=False, block_size=1024)
        
        # 2. Extract IDs and Centroids
        # shape: (N_hits,)
        c_ids = clusterer.cluster_ids 
        # shape: (N_clusters, 4) - typically [x, y, z, energy]
        centroids = clusterer.cluster_centroids() 
        
        # 3. Vectorized Centroid Mapping (Replaces the slow 'for i' loop)
        # Initialize arrays with INF (noise value)
        n_points = len(c_ids)
        cx = np.full(n_points, float('inf'), dtype=np.float32)
        cy = np.full(n_points, float('inf'), dtype=np.float32)
        cz = np.full(n_points, float('inf'), dtype=np.float32)
        
        # Create a mask for valid clusters (assuming -1 is noise)
        valid_mask = c_ids != -1
        
        # If there are any valid clusters in this event
        if np.any(valid_mask):
            # Get the IDs of valid points
            valid_ids = c_ids[valid_mask]
            
            # Map centroids directly using Numpy fancy indexing
            # centroids array is indexed by the cluster ID
            cx[valid_mask] = centroids[valid_ids, 0]
            cy[valid_mask] = centroids[valid_ids, 1]
            cz[valid_mask] = centroids[valid_ids, 2]
            

        # 4. Append results
        res_cluster_ids.append(c_ids)
        res_cx.append(cx)
        res_cy.append(cy)
        res_cz.append(cz)

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