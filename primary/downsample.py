import polars as pl

def lower_resolution(data: pl.DataFrame, factor: int):
    pass

voxel_config = pl.DataFrame({
        "detector": [9, 10, 11, 12, 13, 14],
        "v_size": [25.0, 60.0, 25.0, 60.0, 60.0, 60.0]
    }).with_columns([
        pl.col("detector").cast(pl.UInt8),
        pl.col("v_size").cast(pl.Float32) # Use Float32 for speed
    ])

def voxelize_hits(calo_hits: pl.DataFrame) -> pl.DataFrame:
    # 1. Define Configuration
    #    Using a DataFrame for the join is cleaner for Polars optimization engine


    # 2. Optimized Voxelization Query
    voxelized_hits = (
        calo_hits.lazy()
        # A. Filter (Optimization: reduce data volume before exploding)
        .select(["event_id", "detector", "x", "y", "z", "total_energy"])
        # B. Explode: Flatten the structure
        .explode([
            "detector", "total_energy", "x", "y", "z", 
        
        ])
        
        # C. Attach Voxel Size
        #    'broadcast=True' hints to Polars that voxel_config is a tiny lookup table
        .join(voxel_config.lazy(), on="detector", how="inner") 
        
        # D. Calculate Integer Voxel Indices (The "Grid")
        #    We convert spatial coordinates to integer indices immediately.
        #    (Int32 grouping is 2x-4x faster than Float64 grouping)
        .with_columns([
            (pl.col("x") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_x"),
            (pl.col("y") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_y"),
            (pl.col("z") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_z"),
        ])
        # E. Aggregation: Reduce the data
        #    We DO NOT touch x, y, or z here. We only sum energy and merge lists.
        .group_by(["event_id", "detector", "idx_x", "idx_y", "idx_z"])
        .agg([
            pl.col("total_energy").sum(),
            pl.col('x').mean().alias('x'), # Keep average x for reference (not used in grouping)
            pl.col('y').mean().alias('y'), # Keep average y for reference (not used in grouping)
            pl.col('z').mean().alias('z'), # Keep average z for reference (not used in grouping)
            #pl.col("contrib_particle_ids").flatten(),
            #pl.col("contrib_energies").flatten(),
            pl.col("v_size").first(), # Keep reference to size for reconstruction
            pl.col("x").len().alias("hit_count") # Count hits in the voxel for potential downsampling
        ])
        
        # F. Reconstruction: Calculate Voxel Center
        #    Formula: (Index + 0.5) * Size
        .group_by(["event_id"])
        .agg('*')
        .sort('event_id')
        .collect()
    )
    return voxelized_hits
