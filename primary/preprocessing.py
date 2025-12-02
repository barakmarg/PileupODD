import polars as pl
from sklearn.cluster import MeanShift
import numpy as np

def cast_parent_id_to_int64(df: pl.DataFrame) -> pl.DataFrame:
    """
    Casts the 'parent_id' column from List<Float> to List<Int64>.
    
    This is a Zero-Copy operation where possible, but if the data 
    was physically Float on disk, it creates a new Integer array in RAM.
    """
    return df.with_columns(
        # We specify the target type as a List containing Int64s
        pl.col("parent_id").cast(pl.List(pl.Int64))
    ) 



def add_orphan_mask(df: pl.DataFrame) -> pl.DataFrame:
    print("Computing parent existence mask...")

    df_indexed = df.with_row_index("tmp_event_idx")

    # 1. Build Lookup Table
    valid_ids_lookup = (
        df_indexed.select(["tmp_event_idx", "particle_id"])
        .explode("particle_id")
        .rename({"particle_id": "valid_pid"})
        # Ensure ID types match (Int64 vs Int64)
        .with_columns(pl.col("valid_pid").cast(pl.Int64))
        .unique()
        # --- THE FIX: Add a tracer column ---
        # We need this because 'valid_pid' gets dropped during the join.
        .with_columns(pl.lit(True).alias("found_in_event")) 
    )

    # 2. Flatten Parent IDs
    # (Assuming you already ran cast_parent_id_to_int64, so parent_id is Int64)
    parents_flat = (
        df_indexed.select(["tmp_event_idx", "parent_id"])
        .explode("parent_id")
        .with_row_index("original_order")
    )

    # 3. Join
    matched = parents_flat.join(
        valid_ids_lookup,
        left_on=["tmp_event_idx", "parent_id"],
        right_on=["tmp_event_idx", "valid_pid"],
        how="left"
    )

    # 4. Check the Tracer
    result_mask = (
        matched
        .sort("original_order")
        .with_columns(
            # If 'found_in_event' is Null, the join failed -> Parent Missing
            pl.col("found_in_event").is_null().alias("is_parent_missing")
        )
        .group_by("tmp_event_idx", maintain_order=True)
        .agg(pl.col("is_parent_missing"))
    )

    # 5. Merge back
    return (
        df_indexed
        .join(result_mask, on="tmp_event_idx", how="left")
        .drop("tmp_event_idx")
    )

def add_created_inside_calo_mask(particles: pl.DataFrame) -> pl.DataFrame:
    r_xy_sq_threshold = 1400 ** 2
    z_threshold = 3000

    # 2. Create the Mask Calculation Query
    # We use a separate LazyFrame to calculate masks. This ensures we don't 
    # explode the massive columns (particle_id, parents, etc.) in RAM.
    mask_query = (
        particles.lazy()
        .select(["event_id", "vx", "vy", "vz"]) # Project only what is needed
        .explode(["vx", "vy", "vz"])            # Flatten
        .select([
            pl.col("event_id"),
            (
                # Logic: (vx^2 + vy^2) > 1400^2  OR  |vz| > 3000
                ((pl.col("vx").pow(2) + pl.col("vy").pow(2)) > r_xy_sq_threshold)
                | 
                (pl.col("vz").abs() > z_threshold)
            ).alias("calo_geometry_mask")
        ])
        # IMPORTANT: maintain_order=True guarantees the mask list 
        # aligns perfectly index-by-index with particle_id list
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("calo_geometry_mask"))
    )

    # 3. Join back to original data and Collect
    # We use join(how="left") to attach the new column.
    result = (
        particles.lazy()
        .join(mask_query, on="event_id", how="left")
        # Handle cases where an event might have empty lists (join results in null)
        .with_columns(pl.col("calo_geometry_mask").fill_null([]))
        .collect(streaming=True)
    )

    return result

def run_meanshift(event_idx:int, calo_hits:pl.DataFrame, bandwidth:int =100)->pl.DataFrame:
    """
    x,y,z
    """
    calo_event = calo_hits[event_idx]
    coords = calo_event.select(["x", "y", "z"]).explode(['x','y','z']).to_numpy()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(coords)

    labels = ms.labels_.astype(np.int32)
    centers = ms.cluster_centers_
    cluster_sizes = np.bincount(labels)

    cluster_info = pl.DataFrame(
        {
            "cluster_id": [labels],
            "cluster_cx": [centers[labels, 0]],
            "cluster_cy": [centers[labels, 1]],
            "cluster_cz": [centers[labels, 2]],
        }
    )
    return calo_event.with_columns(cluster_info)

def cluster_purity(calo_hits_with_clusters:pl.DataFrame, ancestors:pl.DataFrame) -> pl.DataFrame:
    """
    Computes the purity of each cluster based on ultimate ancestors of contributing particles.
    """
    # Explode to align hits with clusters
    exploded = (calo_hits_with_clusters.select(['event_id','contrib_energies', 'contrib_particle_ids', 'cluster_id'])
 .explode(['contrib_energies', 'contrib_particle_ids', 'cluster_id'])
 .explode(['contrib_energies', 'contrib_particle_ids']).rename({'contrib_particle_ids':'particle_id','contrib_energies':'energy'})
 )

    # Join with ancestors to get ultimate ancestor IDs
    exploded = exploded.join(
        ancestors,
        left_on="particle_id",
        right_on="particle_id",
        how="left"
    )

    energy_gruped_by_cluster =(exploded.group_by('event_id','cluster_id', 'ultimate_ancestor_id')
                               .agg(pl.col('energy').sum().alias('total_energy_in_cluster'))
                               
                               )
    del exploded
    energy_by_ancestor = (
        energy_gruped_by_cluster.group_by('event_id', 'ultimate_ancestor_id')
        .agg(pl.col('total_energy_in_cluster').sum().alias('energy_by_ancestor'))
    
    )
    final = (energy_gruped_by_cluster.join(
        energy_by_ancestor,
        on=['event_id', 'ultimate_ancestor_id'],
        how='left')
        .rename({'ultimate_ancestor_id':'ultimate_ancestor_id', 
                 'cluster_id':'cluster_id',
                 'total_energy_in_cluster':'total_energy_deps_in_cluster',
                 'energy_by_ancestor':'total_energy_deps'})
        .with_columns(
            (pl.col('total_energy_deps_in_cluster') / pl.col('total_energy_deps')).alias('purity')
        )
    )
    del energy_gruped_by_cluster

    return final


import polars as pl

def map_to_ultimate_ancestor_with_inherited_mask(df: pl.DataFrame) -> pl.DataFrame:
    """
    1. Flattens data and computes geometry masks for all particles.
    2. Finds the ultimate ancestor for every particle.
    3. Assigns the ULTIMATE ANCESTOR'S mask to the descendant.
    """
    print("Preparing data & calculating masks...")

    # Thresholds (using squares to avoid sqrt cost)
    R_SQ_LIMIT = 1400 ** 2
    Z_LIMIT = 3000

    # -------------------------------------------------------------------------
    # STEP 1: Single Pass Preparation
    # Flatten, Compute Mask, Drop Floats immediately.
    # -------------------------------------------------------------------------
    base_state = (
        df.lazy()
        .with_row_index("event_idx")
        .select([
            pl.col("event_idx"),
            pl.col("particle_id").cast(pl.List(pl.Int64)),
            pl.col("parent_id").cast(pl.List(pl.Int64)),
            # Project coordinates only for calculation, then drop them
            pl.col("vx"), pl.col("vy"), pl.col("vz"), pl.col("is_parent_missing")
        ])
        .explode(["particle_id", "parent_id", "vx", "vy", "vz", "is_parent_missing"])
        .select([
            pl.col("event_idx"),
            pl.col("particle_id"),
            
            # Logic: If parent is null, it maps to self
            #pl.coalesce([pl.col("parent_id"), pl.col("particle_id")]).alias("target"),
            pl.when(pl.col("is_parent_missing"))
              .then(pl.col("particle_id"))
              .otherwise(pl.col("parent_id"))
              .alias("target"),
            # Logic: Compute Mask (True/False)
            (
                ((pl.col("vx").pow(2) + pl.col("vy").pow(2)) > R_SQ_LIMIT) |
                (pl.col("vz").abs() > Z_LIMIT)
            ).alias("geometry_mask")
        ])
        .unique(subset=["event_idx", "particle_id"])
        .collect() # Materialize lightweight table (Int64 + Bool only)
    )

    # -------------------------------------------------------------------------
    # STEP 2: Create Separate Lookup Tables
    # -------------------------------------------------------------------------
    
    # Table A: The Reference for Masks [event, particle_id, mask]
    # We park this in memory and don't touch it until the end.
    mask_lookup = base_state.select(["event_idx", "particle_id", "geometry_mask"])

    # Table B: The Active Lineage Map [event, node, target]
    # We only iterate on IDs. We do NOT carry the mask in the loop (saves RAM).
    lineage_map = base_state.select([
        pl.col("event_idx"), 
        pl.col("particle_id").alias("node"), 
        pl.col("target")
    ])

    # -------------------------------------------------------------------------
    # STEP 3: Pointer Jumping Loop (Lineage Tracing)
    # -------------------------------------------------------------------------
    iteration = 0
    while True:
        iteration += 1
        
        # Self-Join to find the next parent
        next_step = lineage_map.join(
            lineage_map,
            left_on=["event_idx", "target"], 
            right_on=["event_idx", "node"],
            how="left",
            suffix="_jump"
        )
        
        # Check convergence: Do we have any new ancestors?
        updates = next_step.filter(
            pl.col("target_jump").is_not_null() & 
            (pl.col("target_jump") != pl.col("target"))
        )
        
        if updates.height == 0:
            print(f"Converged after {iteration} iterations.")
            break
            
        # Apply updates
        lineage_map = next_step.select([
            pl.col("event_idx"),
            pl.col("node"),
            pl.coalesce([pl.col("target_jump"), pl.col("target")]).alias("target")
        ])

    # -------------------------------------------------------------------------
    # STEP 4: Final Join (Retrieve Ancestor's Mask)
    # -------------------------------------------------------------------------
    # We join the final lineage (A -> Ancestor) with the mask lookup (Ancestor -> Mask)
    
    result = lineage_map.join(
        mask_lookup,
        left_on=["event_idx", "target"],       # target is the ultimate ancestor
        right_on=["event_idx", "particle_id"], # lookup mask by ID
        how="left"
    ).select([
        pl.col("event_idx"),
        pl.col("node").alias("particle_id"),
        pl.col("target").alias("ultimate_ancestor_id"),
        pl.col("geometry_mask").alias("ancestor_created_inside_calo")
    ])

    return result