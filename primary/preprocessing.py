from typing import List
import polars as pl
from sklearn.cluster import MeanShift
import numpy as np
from primary.calibration import CALIBRATION
import awkward as ak
import fastjet


import polars as pl
import numpy as np
import fastjet
from primary.calibration import CALIBRATION

def add_cluster_labels(calo_hits: pl.DataFrame, R: float = 0.4) -> pl.DataFrame:
    """
    Robust version of clustering.
    Uses 'global_index' to guarantee that FastJet results merge back 
    to the correct Polars rows, regardless of sorting or shuffling.
    """
 
    # -------------------------------------------------------------------------
    # 1. Explode & Prepare Flat Data (with Global Index)
    # -------------------------------------------------------------------------
    # We only select the columns needed for clustering to keep memory usage low.
    flat_hits = (
        calo_hits.lazy()
        .select(['event_id', 'x', 'y', 'z', 'total_energy', 'detector'])
        # Explode the parallel lists to get 1 row per cell
        .explode(['x', 'y', 'z', 'total_energy', 'detector'])
        
        # CRITICAL: Create a stable Global ID immediately after exploding.
        # This records the exact order of cells as they appeared in the lists.
        .with_row_index("global_id")
        
        # Apply Calibration
        .join(CALIBRATION.lazy(), on='detector', how="left")
        .with_columns((pl.col('total_energy') * pl.col('calib_factor').fill_null(1.0)).alias('E'))
        
        # Calculate Geometry (Massless assumption)
        .with_columns([
            (pl.col('E') * pl.col('x') / (pl.col('x').pow(2) + pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt()).alias('px'),
            (pl.col('E') * pl.col('y') / (pl.col('x').pow(2) + pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt()).alias('py'),
            (pl.col('E') * pl.col('z') / (pl.col('x').pow(2) + pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt()).alias('pz')
        ])
        .collect()
    )

    # -------------------------------------------------------------------------
    # 2. Extract Numpy Arrays for Fast C++ Processing
    # -------------------------------------------------------------------------
    # We sort by event_id locally to make the clustering loop efficient,
    # but we rely strictly on 'global_id' to store results.
    
    df_numpy = flat_hits.sort("event_id").select(["event_id", "global_id", "px", "py", "pz", "E"])
    
    event_ids = df_numpy["event_id"].to_numpy()
    global_ids = df_numpy["global_id"].to_numpy()
    px = df_numpy["px"].to_numpy()
    py = df_numpy["py"].to_numpy()
    pz = df_numpy["pz"].to_numpy()
    E  = df_numpy["E"].to_numpy()

    # The result container, indexed by global_id
    num_total_cells = len(flat_hits)
    result_array = np.full(num_total_cells, -1, dtype=np.int64)

    # -------------------------------------------------------------------------
    # 3. FastJet Loop (Event by Event)
    # -------------------------------------------------------------------------
    unique_events, split_indices = np.unique(event_ids, return_index=True)
    split_indices = split_indices[1:] # remove 0
    
    # Create views for each event
    events_px = np.split(px, split_indices)
    events_py = np.split(py, split_indices)
    events_pz = np.split(pz, split_indices)
    events_E  = np.split(E,  split_indices)
    events_gid = np.split(global_ids, split_indices)

    jet_def = fastjet.JetDefinition(fastjet.kt_algorithm, R)
    
    print(f"Clustering {len(unique_events)} events...")

    for i, event in enumerate(unique_events):
        local_px = events_px[i]
        local_py = events_py[i]
        local_pz = events_pz[i]
        local_E  = events_E[i]
        local_gid = events_gid[i]

        # Convert to PseudoJets with User Index
        particles_pj = []
        for j in range(len(local_px)):
            pj = fastjet.PseudoJet(local_px[j], local_py[j], local_pz[j], local_E[j])
            pj.set_user_index(int(local_gid[j])) # Embed Global ID
            particles_pj.append(pj)

        # Run Clustering
        # inclusive_jets with ptmin=0.0 ensures EVERY cell gets a cluster ID
        cs = fastjet.ClusterSequence(particles_pj, jet_def)
        partitions = cs.inclusive_jets(ptmin=0.0)

        # Map results back to the master array
        for cluster_id, partition in enumerate(partitions):
            for c in partition.constituents():
                gid = c.user_index()
                result_array[gid] = cluster_id

    # -------------------------------------------------------------------------
    # 4. Re-Assemble Nested Lists
    # -------------------------------------------------------------------------
    
    # Create a small DF with the computed labels
    labels_df = pl.DataFrame({
        "event_id": event_ids,
        "global_id": global_ids,
        "cluster_id": result_array
    })

    # Aggregation Step:
    # 1. Sort by global_id. This effectively undoes the event sorting done for the loop
    #    and restores the EXACT order the cells had inside the lists originally.
    # 2. Group by event_id.
    # 3. Aggregate cluster_ids into a list.
    
    cluster_lists = (
        labels_df.lazy()
        .sort("global_id") 
        .group_by("event_id", maintain_order=True) # maintain_order ensures stability
        .agg(pl.col("cluster_id")) # This creates list[i64]
        .collect()
    )

    # -------------------------------------------------------------------------
    # 5. Join Back to Original Data
    # -------------------------------------------------------------------------
    # We simply attach the new column to the original DataFrame
    return calo_hits.join(
        cluster_lists,
        on="event_id",
        how="left"
    )
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
    # if is_parent_missing exist in df, drop it
    if "is_parent_missing" in df.columns:
        df = df.drop("is_parent_missing")
    
    # 1. Build Lookup Table
    valid_ids_lookup = (
        df.lazy()
        .select(["event_id", "particle_id"])
        .explode("particle_id")
        .rename({"particle_id": "valid_pid"})
        # Ensure ID types match (Int64 vs Int64)
        .with_columns(pl.col("valid_pid").cast(pl.Int64))
        .unique()
        # We need this because 'valid_pid' gets dropped during the join.
        .with_columns(pl.lit(True).alias("found_in_event")) 
    )

    # 2. Flatten Parent IDs
    # (Assuming you already ran cast_parent_id_to_int64, so parent_id is Int64)
    parents_flat = (
        df.lazy()
        .select(["event_id", "parent_id"])
        .explode("parent_id")
        .with_row_index("original_order")
    )

    # 3. Join
    matched = parents_flat.join(
        valid_ids_lookup,
        left_on=["event_id", "parent_id"],
        right_on=["event_id", "valid_pid"],
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
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("is_parent_missing"))
    )

    # 5. Merge back
    return (
        df.lazy()
        .join(result_mask, on="event_id", how="left")
        .collect(streaming=True)
    )

def add_eta_and_phi(particles: pl.DataFrame) -> pl.DataFrame:
    """
    Adds 'eta', 'phi', 'pt' with order preservation and numerical safety.
    """
    calculations = (
        particles.lazy()
        .select(["event_id", "px", "py", "pz"])
        .explode(["px", "py", "pz"])
        # FIX 1: Capture the exact original order of particles
        .with_row_index("particle_order")
        .with_columns(
            (pl.col("px").pow(2) + pl.col("py").pow(2)).sqrt().alias("pt"),
            pl.arctan2(pl.col("py"), pl.col("px")).alias("phi")
        )
        .with_columns(
            pl.arctan2(pl.col("pt"), pl.col("pz")).alias("theta")
        )
        .with_columns(
            # Standard eta calculation
            (-((pl.col("theta") / 2).tan().log()))
            # Optional: Handle numerical instability where theta=0 (beam pipe)
            .fill_nan(0.0) 
            .alias("eta")
        )
        # FIX 2: Sort by the captured index to ensure lists are rebuilt in order
        .sort("particle_order")
        .group_by("event_id", maintain_order=True)
        .agg([
            pl.col("eta"),
            pl.col("phi"),
            pl.col("pt")
        ])
    )

    return (
        particles.lazy()
        .join(calculations, on="event_id", how="left")
        .with_columns([
            pl.col("eta").fill_null([]),
            pl.col("phi").fill_null([]),
            pl.col("pt").fill_null([])
        ])
        .collect(streaming=True)
    )


def add_created_inside_calo_mask(particles: pl.DataFrame) -> pl.DataFrame:
    """
    Adds 'created_inside_calo' mask.
    Fixes:
    1. Squaring the threshold (1080 -> 1080**2).
    2. Preserves list order using row indexing.
    """
    
    # Define constants clearly to avoid math errors
    R_SQUARED_THRESHOLD = 1080 ** 2  # <--- FIX: Squared
    Z_THRESHOLD = 3030

    mask_query = (
        particles.lazy()
        .select(["event_id", "vx", "vy", "vz"])
        .explode(["vx", "vy", "vz"])
        # FIX 1: Capture global order immediately after explode
        .with_row_index("global_order")
        .select([
            pl.col("event_id"),
            pl.col("global_order"),
            (
                # Logic: Is the particle created OUTSIDE the tracker volume?
                # ~( (r^2 < thresh) AND (|z| < thresh) )
                ~(
                    ((pl.col("vx").pow(2) + pl.col("vy").pow(2)) < R_SQUARED_THRESHOLD) & 
                    (pl.col("vz").abs() < Z_THRESHOLD)
                )
            ).alias("created_inside_calo")
        ])
        # FIX 2: Sort by the captured index to restore original order before grouping
        .sort("global_order")
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("created_inside_calo"))
    )

    return (
        particles.lazy()
        .join(mask_query, on="event_id", how="left")
        .with_columns(pl.col("created_inside_calo").fill_null([]))
        .collect(streaming=True)
    )

def add_particle_have_track_mask(particles: pl.DataFrame, tracks: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a boolean mask 'has_track' with explicit type casting and order preservation.
    """
    # 1. Prepare Tracks: Cast ID to Int64 and deduplicate
    tracked_particles = (
        tracks.lazy()
        .select(["event_id", "majority_particle_id"])
        .explode("majority_particle_id")
        .with_columns(pl.col("majority_particle_id").cast(pl.Int64))
        .unique()
        .with_columns(pl.lit(True).alias("has_track"))
    )

    # 2. Prepare Particles: Explode -> Index -> Join -> Sort -> Group
    return (
        particles.lazy()
        .select(["event_id", "particle_id"])
        .explode("particle_id")
        .with_columns(pl.col("particle_id").cast(pl.Int64))
        # FIX STEP 1: Capture the original order after exploding but before joining
        .with_row_index("original_order") 
        .join(
            tracked_particles,
            left_on=["event_id", "particle_id"],
            right_on=["event_id", "majority_particle_id"],
            how="left"
        )
        .with_columns(pl.col("has_track").fill_null(False))
        # FIX STEP 2: Restore the original order before aggregating
        .sort("original_order")
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("has_track"))
        .join(
            particles.lazy(),
            on="event_id",
            how="inner"
        )
        .collect(streaming=True)
    )

def get_particles_id_parent_of_inside_calo_particles_mask(particles: pl.DataFrame) -> pl.DataFrame:
    df=     (
    particles.lazy()
    .select(['particle_id', 'parent_id', 'event_id','created_inside_calo'])
    .explode('created_inside_calo', 'parent_id', 'particle_id')
    .filter(pl.col('created_inside_calo'))
    .join
    (
        (particles.lazy()
        .select(['particle_id', 'event_id','created_inside_calo'])
        .explode('created_inside_calo', 'particle_id')
        .filter(~pl.col('created_inside_calo'))
        .rename({'particle_id':'outer_particle_id'})),

        left_on=['parent_id', 'event_id'],
        right_on=['outer_particle_id', 'event_id'],
        how='inner'   
    )
    .select(['parent_id', 'event_id']).unique()
    .rename({'parent_id':'particle_id'})
    .with_columns(pl.lit(True).alias('enter_calo'))
)
    return (
        particles.lazy()
        .select(["event_id", "particle_id"])
        .explode("particle_id")
        .join(
            df,
            left_on=["event_id", "particle_id"],
            right_on=["event_id", "particle_id"],
            how="left"
        )
        .with_columns(pl.col("enter_calo").fill_null(False))
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("enter_calo"))
        .join(
            particles.lazy(),
            on="event_id",
            how="inner"
        )
        .collect(streaming=True)
    )
    
def map_calo_depositors_to_first_outside_ancestor(
    particles: pl.DataFrame, 
    calo_hits: pl.DataFrame
) -> pl.DataFrame:
    print("Step 1: Building the Static Lookup Table (The Census)...")
    
    # -------------------------------------------------------------------------
    # 1. The Lookup Table (Static)
    # -------------------------------------------------------------------------
    # We must flatten the full particle list once to allow looking up parents.
    # We define the "Next Step" for every particle here.
    
    lookup_table = (
        particles.lazy()
        .select([
            pl.col("event_id"),
            pl.col("particle_id"),
            pl.col("parent_id"),
            pl.col("is_parent_missing"),
            pl.col("created_inside_calo")
        ])
        .explode(["particle_id", "parent_id", "is_parent_missing", "created_inside_calo"])
        .with_columns([
            pl.col("particle_id").cast(pl.Int64),
            pl.col("parent_id").cast(pl.Int64)
        ])
        .select([
            pl.col("event_id"),
            pl.col("particle_id").alias("node"), # Key
            
            # THE NAVIGATION LOGIC:
            # 1. If I am OUTSIDE, I point to Myself (Anchor/Stop).
            # 2. If I am INSIDE, I point to my Parent.
            # 3. If Parent missing, I point to Null.
            pl.when(pl.col("created_inside_calo").not_())
              .then(pl.col("particle_id"))
              .when(pl.col("is_parent_missing"))
              .then(None)
              .otherwise(pl.col("parent_id"))
              .alias("next_hop")                 # Value
        ])
        # Materialize this once. It acts as a hash map in the join.
        .collect(streaming=True)
    )

    print("Step 2: initializing Active Paths for Depositors only...")

    # -------------------------------------------------------------------------
    # 2. The Active Paths (Dynamic)
    # -------------------------------------------------------------------------
    # We extract the depositors and immediately find their first hop.
    # This reduces the dataframe size from ~Millions (all particles) 
    # to ~Thousands (only depositors).
    
    depositors_list = (
        calo_hits.lazy()
        .select(['event_id', 'contrib_particle_ids'])
        .explode('contrib_particle_ids')
        .explode('contrib_particle_ids') # Double explode if list[list]
        .rename({'contrib_particle_ids': 'particle_id'})
        .unique(subset=['event_id', 'particle_id'])
        .select([
            pl.col('event_id'),
            pl.col('particle_id').cast(pl.Int64)
        ])
    )

    # Initialize the active trace by joining depositors with the lookup table
    active_paths = (
        depositors_list.join(
            lookup_table.lazy(), 
            left_on=["event_id", "particle_id"], 
            right_on=["event_id", "node"],
            how="left"
        )
        .rename({"next_hop": "target"}) # 'target' is the current ancestor candidate
        .collect(streaming=True)
    )
    
    # -------------------------------------------------------------------------
    # 3. Pointer Jumping Loop (on Small Data)
    # -------------------------------------------------------------------------
    # We iterate only on 'active_paths' (small), 
    # looking up against 'lookup_table' (large, but static).
    
    iteration = 0
    while True:
        iteration += 1
        
        # Check: Where does my current target point to?
        next_step = active_paths.join(
            lookup_table,
            left_on=["event_id", "target"], 
            right_on=["event_id", "node"],
            how="left",
            suffix="_jump"
        )
        
        # Calculate changes
        # Logic: If 'target' points to 'next_hop' and they are different, we advance.
        # If 'target' points to itself (it's Outside), we stop updating.
        
        # Filter for rows that actually need moving
        updates = next_step.filter(
            pl.col("next_hop").is_not_null() & 
            (pl.col("next_hop") != pl.col("target"))
        )
        
        if updates.height == 0:
            print(f"Converged after {iteration} iterations on {active_paths.height} depositors.")
            break
            
        # Update the active paths
        # We take next_step (which has the jumped values) as the new state
        active_paths = next_step.select([
            pl.col("event_id"),
            pl.col("particle_id"),
            # If the jump returns null (dead end), keep the old target or become null?
            # Based on logic: If next_hop is null, parent is missing.
            # If next_hop is valid, take it.
            pl.col("next_hop").alias("target")
        ])

    # -------------------------------------------------------------------------
    # 4. Result
    # -------------------------------------------------------------------------
    return active_paths.rename({"target": "ancestor_outside_calo_id"})


def map_calo_depositors_to_first_outside_ancestorv2(
    particles: pl.DataFrame, 
    calo_hits: pl.DataFrame
) -> pl.DataFrame:
    print("Step 1: Building the Static Lookup Table (The Census)...")
    
    # -------------------------------------------------------------------------
    # 1. The Lookup Table (Static)
    # -------------------------------------------------------------------------
    # We must flatten the full particle list once to allow looking up parents.
    # We define the "Next Step" for every particle here.
    
    lookup_table = (
        particles.lazy()
        .select([
            pl.col("event_id"),
            pl.col("particle_id"),
            pl.col("parent_id"),
            pl.col("is_parent_missing"),
            pl.col("created_inside_calo")
        ])
        .explode(["particle_id", "parent_id", "is_parent_missing", "created_inside_calo"])
        .with_columns([
            pl.col("particle_id").cast(pl.Int64),
            pl.col("parent_id").cast(pl.Int64)
        ])
        .select([
            pl.col("event_id"),
            pl.col("particle_id").alias("node"), # Key
            
            # THE NAVIGATION LOGIC:
            # 1. If I am OUTSIDE, I point to Myself (Anchor/Stop).
            # 2. If I am INSIDE, I point to my Parent.
            # 3. If Parent missing, I point to Null.
            pl.when(pl.col("created_inside_calo").not_())
              .then(pl.col("particle_id"))
              .when(pl.col("is_parent_missing"))
              .then(None)
              .otherwise(pl.col("parent_id"))
              .alias("next_hop")                 # Value
        ])
        # Materialize this once. It acts as a hash map in the join.
        .collect(streaming=True)
    )

    print("Step 2: initializing Active Paths for Depositors only...")

    # -------------------------------------------------------------------------
    # 2. The Active Paths (Dynamic)
    # -------------------------------------------------------------------------
    # We extract the depositors and immediately find their first hop.
    # This reduces the dataframe size from ~Millions (all particles) 
    # to ~Thousands (only depositors).
    
    depositors_list = (
        calo_hits.lazy()
        .select(['event_id', 'contrib_particle_ids'])
        .explode('contrib_particle_ids')
        .explode('contrib_particle_ids') # Double explode if list[list]
        .rename({'contrib_particle_ids': 'particle_id'})
        .unique(subset=['event_id', 'particle_id'])
        .select([
            pl.col('event_id'),
            pl.col('particle_id').cast(pl.Int64)
        ])
    )

    # particles created in calo also
    dep2 = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'created_inside_calo'])
        .explode(['particle_id', 'created_inside_calo'])
        .filter(pl.col('created_inside_calo'))
        .select([
            pl.col('event_id'),
            pl.col('particle_id').cast(pl.Int64)
        ])
    )
    depositors_list = pl.concat([depositors_list, dep2]).unique(subset=['event_id', 'particle_id'])
    # Initialize the active trace by joining depositors with the lookup table
    active_paths = (
        depositors_list.join(
            lookup_table.lazy(), 
            left_on=["event_id", "particle_id"], 
            right_on=["event_id", "node"],
            how="left"
        )
        .rename({"next_hop": "target"}) # 'target' is the current ancestor candidate
        .collect(streaming=True)
    )
    
    # -------------------------------------------------------------------------
    # 3. Pointer Jumping Loop (on Small Data)
    # -------------------------------------------------------------------------
    # We iterate only on 'active_paths' (small), 
    # looking up against 'lookup_table' (large, but static).
    
    iteration = 0
    while True:
        iteration += 1
        
        # Check: Where does my current target point to?
        next_step = active_paths.join(
            lookup_table,
            left_on=["event_id", "target"], 
            right_on=["event_id", "node"],
            how="left",
            suffix="_jump"
        )
        
        # Calculate changes
        # Logic: If 'target' points to 'next_hop' and they are different, we advance.
        # If 'target' points to itself (it's Outside), we stop updating.
        
        # Filter for rows that actually need moving
        updates = next_step.filter(
            pl.col("next_hop").is_not_null() & 
            (pl.col("next_hop") != pl.col("target"))
        )
        
        if updates.height == 0:
            print(f"Converged after {iteration} iterations on {active_paths.height} depositors.")
            break
            
        # Update the active paths
        # We take next_step (which has the jumped values) as the new state
        active_paths = next_step.select([
            pl.col("event_id"),
            pl.col("particle_id"),
            # If the jump returns null (dead end), keep the old target or become null?
            # Based on logic: If next_hop is null, parent is missing.
            # If next_hop is valid, take it.
            pl.col("next_hop").alias("target")
        ])

    # -------------------------------------------------------------------------
    # 4. Result
    # -------------------------------------------------------------------------
    return active_paths.rename({"target": "ancestor_outside_calo_id"})

def get_particles_id_parent_of_inside_calo_particles_maskv3(particles: pl.DataFrame, calo_hits: pl.DataFrame) -> pl.DataFrame:
    """
    Identifies particles that are ancestors of calorimeter hits (entering the calo).
    """
    
    # 1. Prepare the Flags (Right side of the join)
    # Ensure IDs are Int64 to match the main dataframe safely
    combined_flags = (
        map_calo_depositors_to_first_outside_ancestorv2(particles, calo_hits)
        .lazy() # Ensure we work lazily if the helper returns eager
        .select(['event_id', 'ancestor_outside_calo_id'])
        .unique()
        .rename({'ancestor_outside_calo_id': 'particle_id'})
        .with_columns([
            pl.lit(True).alias('enter_calo'),
            pl.col('particle_id').cast(pl.Int64)
        ])
    )

    # 2. Attach Flags to Particles (Left side)
    return (
        particles.lazy()
        .select(["event_id", "particle_id"])
        .explode("particle_id")
        .with_columns(pl.col("particle_id").cast(pl.Int64))
        
        .with_row_index("global_order")
        
        .join(
            combined_flags,
            on=["event_id", "particle_id"],
            how="left"
        )
        .with_columns(pl.col("enter_calo").fill_null(False))
        
        # FIX 2: Sort by the captured index to restore order before grouping
        .sort("global_order")
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("enter_calo"))
        
        # 3. Join back to original data
        .join(
            particles.lazy(),
            on="event_id",
            how="inner"
        )
        .collect(streaming=True)
    )

def get_particles_id_parent_of_inside_calo_particles_maskv2(particles: pl.DataFrame, calo_hits: pl.DataFrame) -> pl.DataFrame:
    # Define "Outside Particles" query once
    # We need event_id and particle_id for particles where created_inside_calo is False
    outside_particles_lazy = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'created_inside_calo'])
        .explode(['particle_id', 'created_inside_calo'])
        .filter(~pl.col('created_inside_calo'))
        .select(['event_id', 'particle_id'])
    )

    # 1. Parents (outside) of Children (inside)
    df = (
        particles.lazy()
        .select(['event_id', 'parent_id', 'created_inside_calo'])
        .explode(['parent_id', 'created_inside_calo'])
        .filter(pl.col('created_inside_calo')) # Children inside
        .select(['event_id', 'parent_id'])
        .join(
            outside_particles_lazy.rename({'particle_id': 'outer_particle_id'}),
            left_on=['event_id', 'parent_id'],
            right_on=['event_id', 'outer_particle_id'],
            how='inner'
        )
        .select(['event_id', 'parent_id'])
        .unique()
        .rename({'parent_id': 'particle_id'})
        .with_columns(pl.lit(True).alias('enter_calo'))
    )
    
    # 2. Particles (outside) that hit Calo
    df2 = (
        calo_hits.lazy()
        .select(['event_id', 'contrib_particle_ids'])
        .explode('contrib_particle_ids')
        .explode('contrib_particle_ids')
        .rename({'contrib_particle_ids': 'particle_id'})
        .with_columns(pl.col('particle_id').cast(pl.Int64))
        .unique() 
        .join(
            outside_particles_lazy,
            on=['event_id', 'particle_id'],
            how='inner'
        )
        .select(['event_id', 'particle_id'])
        .with_columns(pl.lit(True).alias('enter_calo'))
    )

    # Merge both dataframes
    combined_flags = pl.concat([df, df2]).unique()

    # Join back to original structure
    return (
        particles.lazy()
        .select(["event_id", "particle_id"])
        .explode("particle_id")
        .join(
            combined_flags,
            on=["event_id", "particle_id"],
            how="left"
        )
        .with_columns(pl.col("enter_calo").fill_null(False))
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("enter_calo"))
        .join(
            particles.lazy(),
            on="event_id",
            how="inner"
        )
        .collect(streaming=True)
    )

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
    exploded = (
                calo_hits_with_clusters.lazy().select(['event_id','contrib_energies', 'contrib_particle_ids', 'cluster_id', 'detector'])
        .explode(['contrib_energies', 'contrib_particle_ids', 'cluster_id', 'detector'])
        .explode(['contrib_energies', 'contrib_particle_ids']).rename({'contrib_particle_ids':'particle_id','contrib_energies':'energy'})
        .join(
                CALIBRATION.select(['detector', 'calib_factor']),
                on='detector',
        )
        .with_columns((pl.col('energy') * pl.col('calib_factor')).alias('energy'))
        .drop('calib_factor')
        .drop('detector')
 )

    # Join with ancestors to get ultimate ancestor IDs
    exploded = exploded.join(
        ancestors.lazy(),
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
    ).collect(streaming=True)

    return final


def particle_energy_calo_deposits_ratio(
    calo_hits: pl.DataFrame, 
    ancestors: pl.DataFrame, 
    particles: pl.DataFrame
) -> pl.DataFrame:
    """
    Computes purity using Lazy execution and Streaming.
    
    Optimizations:
    1. No intermediate .collect(): Data flows through without holding huge tables in RAM.
    2. Streaming=True: Processes data in chunks (batch-wise).
    3. Composite Joins: Joins on [event_id, particle_id] to ensure correctness.
    4. Early Aggregation: Sums energies immediately after mapping to ancestors.
    """
    
    # 1. PREPARE HITS (Lazy)
    # We flatten the nested structure but DO NOT materialize it.
    hits_lazy = (
        calo_hits.lazy()
        .select(['event_id', 'contrib_energies', 'contrib_particle_ids', 'detector'])
        # Double explode implies list[list] structure. 
        # Polars optimizes sequential explodes in lazy mode.
        .explode(['contrib_energies', 'contrib_particle_ids', 'detector'])
        .explode(['contrib_energies', 'contrib_particle_ids'])
        .rename({
            'contrib_particle_ids': 'particle_id',
            'contrib_energies': 'energy'
        })
        .join(
            CALIBRATION.lazy().select(['detector', 'calib_factor']),
            on='detector',
            how='left'
        )
        .with_columns((pl.col('energy') * pl.col('calib_factor')).alias('energy'))
        .drop('calib_factor')
        .drop('detector')
    )

    # 2. PREPARE ANCESTORS (Lazy)
    # Ensure we have the mapping keys ready
    ancestors_lazy = ancestors.lazy().select(['event_id', 'particle_id', 'ultimate_ancestor_id'])

    # 3. PREPARE DENOMINATOR (Total Particle Energy)
    # Flatten particles to get the reference energy for the denominator
    particles_lazy = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'energy'])
        .explode(['particle_id', 'energy'])
        .rename({'energy': 'total_particle_energy'})
    )

    # 4. EXECUTE PIPELINE
    # This entire block is a single query plan.
    final_query = (
        hits_lazy
        # Step A: Map Hit-Particles to their Ultimate Ancestors
        # We join on event_id AND particle_id to avoid cross-event collisions
        .join(
            ancestors_lazy,
            on=['event_id', 'particle_id'],
            how='right'
        )
        .with_columns(pl.col('energy').fill_null(0.0))

        # Optimization: Drop rows where ancestor lookup failed (optional, but saves RAM)
      
        
        # Step B: Aggregate Numerator (Energy in Calo per Ancestor)
        # This reduces the row count massively (from #hits to #ancestors)
        .group_by(['event_id', 'ultimate_ancestor_id'])
        .agg(
            pl.col('energy').sum().alias('total_energy_in_calo')
        )
        
        # Step C: Join with Denominator (Total Energy of that Ancestor)
        # Note: We join ancestor_id (from hits) to particle_id (from particles table)
        .join(
            particles_lazy,
            left_on=['event_id', 'ultimate_ancestor_id'],
            right_on=['event_id', 'particle_id'],
            how='left'
        )
        
        # Step D: Calculate Purity
        .with_columns(
            (pl.col('total_energy_in_calo') / pl.col('total_particle_energy')).alias('purity')
        )
    )

    # 5. COLLECT WITH STREAMING
    # This is the only time RAM is heavily used, but streaming manages it in batches.
    return final_query.collect(streaming=True)

def cluster_purity(calo_hits_with_clusters: pl.DataFrame, ancestors: pl.DataFrame) -> pl.DataFrame:
    """
    Computes the purity/efficiency of each cluster based on ultimate ancestors.
    Optimized for memory using lazy execution, strict column selection, and window functions.
    """
    
    ancestors_lazy = (
        ancestors.lazy()
        .select(['event_id', 'src_particle_id', 'target_particle_id'])
        .rename({
            'src_particle_id': 'particle_id', 
            'target_particle_id': 'ultimate_ancestor_id'
        })
        .with_columns(pl.col("particle_id").cast(pl.Int64))
    )

    # 2. Prepare Calibration (Lazy)
    calib_lazy = (
        CALIBRATION.lazy()
        .select(['detector', 'calib_factor'])
        # Handle cases where a detector might be missing from the map (default to 1.0)
    )

    return (
        calo_hits_with_clusters.lazy()
        # A. Select required columns (Include 'detector' for calibration)
        .select(['event_id', 'contrib_energies', 'contrib_particle_ids', 'cluster_id', 'detector'])
        
        # B. Explode Level 1: Cells
        # We need to align detector ID with the lists of energies
        .explode(['contrib_energies', 'contrib_particle_ids', 'cluster_id', 'detector'])
        
        # C. Join Calibration (Cell Level)
        # This is more efficient than joining after the second explode
        .join(
            calib_lazy,
            on="detector",
            how="left"
        )

        # D. Explode Level 2: Contributors
        .explode(['contrib_energies', 'contrib_particle_ids'])
        
        # E. Apply Calibration & Type Cast
        .with_columns([
            (pl.col('contrib_energies') * pl.col('calib_factor')).alias('energy'),
            pl.col('contrib_particle_ids').cast(pl.Int64).alias('particle_id')
        ])
        
        # F. Drop heavy columns immediately
        .select(['event_id', 'cluster_id', 'particle_id', 'energy'])
        
        # G. Join with Ancestors (Strict on Event + Particle)
        .join(
            ancestors_lazy,
            on=["event_id", "particle_id"],
            how="left"
        )

        # H. Aggregation (Sum Energies)
        .group_by(['event_id', 'cluster_id', 'ultimate_ancestor_id'])
        .agg(
            pl.col('energy').sum().alias('total_energy_deps_in_cluster')
        )

        # I. Window Function for Denominator (Efficiency)
        # Calculates: "Total Calibrated Energy of this Ancestor in the Event"
        .with_columns(
            pl.col('total_energy_deps_in_cluster')
            .sum()
            .over(['event_id', 'ultimate_ancestor_id'])
            .alias('total_energy_deps')
        )
        
        # J. Purity Calculation
        .select([
            pl.col('event_id'),
            pl.col('cluster_id'),
            pl.col('ultimate_ancestor_id'),
            pl.col('total_energy_deps_in_cluster'),
            pl.col('total_energy_deps'),
            (pl.col('total_energy_deps_in_cluster') / pl.col('total_energy_deps')).alias('purity')
        ])
        .collect(streaming=True)
    )

def number_of_particles_per_cluster(calo_hits_with_clusters: pl.DataFrame, ancestors: pl.DataFrame, cut_off_percent: float = 0.05) -> pl.DataFrame:
    """
    #particles / cluster
    Computes the purity/efficiency of each cluster based on ultimate ancestors.
    Optimized for memory using lazy execution, strict column selection, and window functions.
    """
    
    ancestors_lazy = (
        ancestors.lazy()
        .select(['event_id', 'src_particle_id', 'target_particle_id'])
        .rename({
            'src_particle_id': 'particle_id', 
            'target_particle_id': 'ultimate_ancestor_id'
        })
        .with_columns(pl.col("particle_id").cast(pl.Int64))
    )

    # 2. Prepare Calibration (Lazy)
    calib_lazy = (
        CALIBRATION.lazy()
        .select(['detector', 'calib_factor'])
        # Handle cases where a detector might be missing from the map (default to 1.0)
    )

    return (
        calo_hits_with_clusters.lazy()
        # A. Select required columns (Include 'detector' for calibration)
        .select(['event_id', 'contrib_energies', 'contrib_particle_ids', 'cluster_id', 'detector'])
        
        # B. Explode Level 1: Cells
        # We need to align detector ID with the lists of energies
        .explode(['contrib_energies', 'contrib_particle_ids', 'cluster_id', 'detector'])
        
        # C. Join Calibration (Cell Level)
        # This is more efficient than joining after the second explode
        .join(
            calib_lazy,
            on="detector",
            how="left"
        )

        # D. Explode Level 2: Contributors
        .explode(['contrib_energies', 'contrib_particle_ids'])
        
        # E. Apply Calibration & Type Cast
        .with_columns([
            (pl.col('contrib_energies') * pl.col('calib_factor')).alias('energy'),
            pl.col('contrib_particle_ids').cast(pl.Int64).alias('particle_id')
        ])
        
        # F. Drop heavy columns immediately
        .select(['event_id', 'cluster_id', 'particle_id', 'energy'])
        
        # G. Join with Ancestors (Strict on Event + Particle)
        .join(
            ancestors_lazy,
            on=["event_id", "particle_id"],
            how="left"
        )

        # H. Aggregation (Sum Energies)
        .group_by(['event_id', 'cluster_id', 'ultimate_ancestor_id'])
        .agg(
            pl.col('energy').sum().alias('total_particle_energy_deps_in_cluster')
        )

        .with_columns(
            pl.col('total_particle_energy_deps_in_cluster').sum().over(['event_id', 'cluster_id']).alias('cluster_total_energy')
            
        )
        .filter(pl.col('total_particle_energy_deps_in_cluster') / pl.col('cluster_total_energy') > cut_off_percent)
        .group_by(['event_id', 'cluster_id'])
        .agg(
            pl.col('ultimate_ancestor_id').count().alias('num_contributing_ancestors'),
            pl.col('total_particle_energy_deps_in_cluster').max().alias('max_ancestor_energy_deps_in_cluster'),
            pl.col('cluster_total_energy').max().alias('cluster_total_energy')
        )
        # J. Purity Calculation
        .select([
            pl.col('event_id'),
            pl.col('cluster_id'),
            pl.col('num_contributing_ancestors'),
            pl.col('max_ancestor_energy_deps_in_cluster'),
            pl.col('cluster_total_energy')        ])
        .collect(streaming=True)
    )

import polars as pl

def particle_purity_by_class(
    calo_hits: pl.DataFrame, 
    ancestors: pl.DataFrame, 
    particles: pl.DataFrame,
    pdg_classes: List[List[int]]
) -> pl.DataFrame:
    """
    Computes purity using Lazy execution and Streaming.
    
    Optimizations:
    1. No intermediate .collect(): Data flows through without holding huge tables in RAM.
    2. Streaming=True: Processes data in chunks (batch-wise).
    3. Composite Joins: Joins on [event_id, particle_id] to ensure correctness.
    4. Early Aggregation: Sums energies immediately after mapping to ancestors.
    """
    pdg_classes_df = pl.DataFrame({
    "class_id": list(range(len(pdg_classes))),
    "pdg_ids": pdg_classes
})
    # 1. PREPARE HITS (Lazy)
    # We flatten the nested structure but DO NOT materialize it.
    hits_lazy = (
        calo_hits.lazy()
        .select(['event_id', 'contrib_energies', 'contrib_particle_ids', 'detector'])
        # Double explode implies list[list] structure. 
        # Polars optimizes sequential explodes in lazy mode.
        .explode(['contrib_energies', 'contrib_particle_ids', 'detector'])
        .explode(['contrib_energies', 'contrib_particle_ids'])
        .rename({
            'contrib_particle_ids': 'particle_id',
            'contrib_energies': 'energy'
        })
        .join(
            CALIBRATION.lazy().select(['detector', 'calib_factor']),
            on='detector',
            how='left'
        )
        .with_columns((pl.col('energy') * pl.col('calib_factor')).alias('energy'))
        .drop('calib_factor')
        .drop('detector')
    )

    # 2. PREPARE ANCESTORS (Lazy)
    # Ensure we have the mapping keys ready
    ancestors_lazy = ancestors.lazy().select(['event_id', 'particle_id', 'ultimate_ancestor_id'])

    # 3. PREPARE DENOMINATOR (Total Particle Energy)
    # Flatten particles to get the reference energy for the denominator
    particles_lazy = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'energy', 'pdg_id'])
        .explode(['particle_id', 'energy', 'pdg_id'])
        .join(
            (
            pdg_classes_df.lazy()
            .explode('pdg_ids')
           .rename({'pdg_ids':'pdg_id'})
            ),
            left_on='pdg_id',
            right_on='pdg_id',
            how='left'
        )
        .rename({'energy': 'total_particle_energy'})
        .with_columns(pl.col("class_id").fill_null(-1))

    )

    # 4. EXECUTE PIPELINE
    # This entire block is a single query plan.
    final_query = (
        hits_lazy
        # Step A: Map Hit-Particles to their Ultimate Ancestors
        # We join on event_id AND particle_id to avoid cross-event collisions
        .join(
            ancestors_lazy,
            on=['event_id', 'particle_id'],
            how='right'
        )
        .with_columns(pl.col('energy').fill_null(0.0))

        # Optimization: Drop rows where ancestor lookup failed (optional, but saves RAM)
      
        
        # Step B: Aggregate Numerator (Energy in Calo per Ancestor)
        # This reduces the row count massively (from #hits to #ancestors)
        .group_by(['event_id', 'ultimate_ancestor_id'])
        .agg(
            pl.col('energy').sum().alias('total_energy_in_calo')
        )
        
        # Step C: Join with Denominator (Total Energy of that Ancestor)
        # Note: We join ancestor_id (from hits) to particle_id (from particles table)
        .join(
            particles_lazy,
            left_on=['event_id', 'ultimate_ancestor_id'],
            right_on=['event_id', 'particle_id'],
            how='left'
        )
        
        # Step D: Calculate Purity
        .with_columns(
            (pl.col('total_energy_in_calo') / pl.col('total_particle_energy')).alias('purity')
        )
    )

    # 5. COLLECT WITH STREAMING
    # This is the only time RAM is heavily used, but streaming manages it in batches.
    return final_query.collect(streaming=True)



def get_mask_confusion_matrix(df: pl.DataFrame, mask_a: str, mask_b: str, is_lazy: bool = False) -> pl.DataFrame:
    """
    Calculates the confusion matrix between two boolean list columns.
    Memory efficient: Projects and explodes only the relevant columns.
    """
    print(f"Comparing '{mask_a}' vs '{mask_b}'...")
    if not is_lazy:
        df2 = df.lazy()
    else:
        df2 = df
    stats = (
        df2
        # 1. Select only the two columns to compare (saves RAM)
        .select([pl.col(mask_a), pl.col(mask_b)])
        # 2. Explode to flat boolean arrays
        .explode([mask_a, mask_b])
        .select([
            (pl.col(mask_a) & pl.col(mask_b)).alias("both_true"),
            (pl.col(mask_a) & ~pl.col(mask_b)).alias("a_only"),
            (~pl.col(mask_a) & pl.col(mask_b)).alias("b_only"),
            (~pl.col(mask_a) & ~pl.col(mask_b)).alias("both_false")
        ])
        .sum() # Sum boolean columns (True=1, False=0)
        .collect(streaming=True)
    )

    # Extract values
    both_true = stats["both_true"][0]
    a_only = stats["a_only"][0]
    b_only = stats["b_only"][0]
    both_false = stats["both_false"][0]

    # Print Report
    print(f"\n--- Comparison Report: {mask_a} vs {mask_b} ---")
    print(f"Intersection (Both True): {both_true:,}")
    print(f"Only in {mask_a}:       {a_only:,}")
    print(f"Only in {mask_b}:       {b_only:,}")
    print(f"Both False:             {both_false:,}")
    print(f"Both equal (True+False),   {both_true + both_false:,},percentage: {(both_true + both_false) / (both_true + a_only + b_only + both_false) * 100:.2f}%")
    print(f"Both true matchinf=g (without both False, both_true / (both_true + a_only + b_only) ,   {both_true:,},percentage: {both_true / (both_true + a_only + b_only) * 100:.2f}%")
    print("-" * 30)
    

    return stats


def child_is_primary_and_parent_exist(particles: pl.DataFrame, head=20) -> pl.DataFrame:
    return (
    particles.lazy()
    .select('primary','event_id', 'is_parent_missing', 'pdg_id', 'parent_id') # this is A
    .explode(['primary', 'is_parent_missing', 'pdg_id', 'parent_id'])
    .filter((pl.col('primary') & ~pl.col('is_parent_missing')))
    .select('pdg_id','event_id', 'parent_id')
    .join(
            (
                particles.lazy()
                .select('particle_id', 'pdg_id', 'event_id')
                .rename({'particle_id': 'particle_id', 'pdg_id': 'parent_pdg_id', 'event_id': 'event_id'})
                .explode('particle_id', 'parent_pdg_id')),
        left_on=['parent_id', 'event_id'],
        right_on=['particle_id', 'event_id'],
        how='left',
    )

    .group_by('pdg_id', 'parent_pdg_id')
    .len()
    .sort('len', descending=True)
    .head(head)
    .collect(streaming=True) ) 

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
        .select([
            pl.col("event_id"),
            pl.col("particle_id").cast(pl.List(pl.Int64)),
            pl.col("parent_id").cast(pl.List(pl.Int64)),
            # Project coordinates only for calculation, then drop them
            pl.col("vx"), pl.col("vy"), pl.col("vz"), pl.col("is_parent_missing")
        ])
        .explode(["particle_id", "parent_id", "vx", "vy", "vz", "is_parent_missing"])
        .select([
            pl.col("event_id"),
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
        .unique(subset=["event_id", "particle_id"])
        .collect() # Materialize lightweight table (Int64 + Bool only)
    )

    # -------------------------------------------------------------------------
    # STEP 2: Create Separate Lookup Tables
    # -------------------------------------------------------------------------
    
    # Table A: The Reference for Masks [event, particle_id, mask]
    # We park this in memory and don't touch it until the end.
    mask_lookup = base_state.select(["event_id", "particle_id", "geometry_mask"])

    # Table B: The Active Lineage Map [event, node, target]
    # We only iterate on IDs. We do NOT carry the mask in the loop (saves RAM).
    lineage_map = base_state.select([
        pl.col("event_id"), 
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
            left_on=["event_id", "target"], 
            right_on=["event_id", "node"],
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
            pl.col("event_id"),
            pl.col("node"),
            pl.coalesce([pl.col("target_jump"), pl.col("target")]).alias("target")
        ])

    # -------------------------------------------------------------------------
    # STEP 4: Final Join (Retrieve Ancestor's Mask)
    # -------------------------------------------------------------------------
    # We join the final lineage (A -> Ancestor) with the mask lookup (Ancestor -> Mask)
    
    result = lineage_map.join(
        mask_lookup,
        left_on=["event_id", "target"],       # target is the ultimate ancestor
        right_on=["event_id", "particle_id"], # lookup mask by ID
        how="left"
    ).select([
        pl.col("event_id"),
        pl.col("node").alias("particle_id"),
        pl.col("target").alias("ultimate_ancestor_id"),
        pl.col("geometry_mask").alias("ancestor_created_inside_calo")
    ])

    return result



def map_to_nearest_ancestor_with_track(particles: pl.DataFrame) -> pl.DataFrame:
    print("Preparing lineage map...")

    # -------------------------------------------------------------------------
    # STEP 1: Flatten and Initialize (The "Hop 0" State)
    # -------------------------------------------------------------------------
    # 1. Select the relevant list columns.
    # 2. Explode them together to unnest the event structure.
    # 3. Cast types to Int64 to ensure matching (parent is i64, particle is u64).
    
    flat_particles = (
        particles.lazy()
        .select([
            pl.col("event_id"),
            pl.col("particle_id"),
            pl.col("parent_id"),
            pl.col("has_track"),
            pl.col("is_parent_missing")
        ])
        .explode(["particle_id", "parent_id", "has_track", "is_parent_missing"])
        .with_columns([
            pl.col("particle_id").cast(pl.Int64),
            pl.col("parent_id").cast(pl.Int64)
        ])
    )

    # Define the initial targets:
    # - If has_track: Point to SELF (I am the ancestor).
    # - Else: Point to PARENT.
    # - If parent is missing: Point to NULL.
    lineage_map = (
        flat_particles
        .select([
            pl.col("event_id"),
            pl.col("particle_id").alias("node"), # Current Node
            
            pl.when(pl.col("has_track"))
            .then(pl.col("particle_id"))
            .when(pl.col("is_parent_missing"))
            .then(None)
            .otherwise(pl.col("parent_id"))
            .alias("target")                     # Where I'm looking
        ])
        .collect(streaming=True)
    )

    # We iterate on this map.
    current_state = lineage_map.clone()

    # -------------------------------------------------------------------------
    # STEP 2: Pointer Jumping Loop
    # -------------------------------------------------------------------------
    iteration = 0
    while True:
        iteration += 1
        
        # Self-Join: "If I point to Target, where does Target point?"
        next_step = current_state.join(
            current_state,
            left_on=["event_id", "target"], 
            right_on=["event_id", "node"],
            how="left",
            suffix="_jump"
        )
        
        # LOGIC UPDATE:
        # 1. new_target comes from the join ('target_jump').
        # 2. If 'target_jump' is Null, it means my 'target' (parent) does not exist 
        #    in the table (dead end). I should stop looking and become Null.
        # 3. If I was already pointing to myself (track found), the join succeeds 
        #    (I find myself), so I stay pointing to myself.
        
        next_step = next_step.select([
            pl.col("event_id"),
            pl.col("node"),
            pl.col("target_jump").alias("new_target"),
            pl.col("target").alias("old_target")
        ])

        # Check Convergence:
        # We continue if any valid path is still updating.
        # (new_target != old_target)
        
        # Note on Nulls:
        # If old_target was 100, and 100 is missing, new_target becomes Null.
        # 100 != Null is Null (filtered out). We need to handle this explicitly 
        # if we want to detect the change from "Dangling ID" to "Null".
        # However, for simple convergence, we usually just check if we found a NEW valid ID.
        
        changes = next_step.filter(
            pl.col("new_target").is_not_null() & 
            (pl.col("new_target") != pl.col("old_target"))
        )
        
        if changes.height == 0:
            print(f"Converged after {iteration} iterations.")
            # Final update to ensure dead-ends are Null
            current_state = next_step.select([
                pl.col("event_id"), 
                pl.col("node"), 
                pl.col("new_target").alias("target")
            ])
            break
            
        # Apply updates
        current_state = next_step.select([
            pl.col("event_id"),
            pl.col("node"),
            # If the jump failed (Null), it means dead end -> propagate Null.
            # If jump succeeded, take the new target.
            pl.col("new_target").alias("target")
        ])

    # -------------------------------------------------------------------------
    # STEP 3: Final Formatting
    # -------------------------------------------------------------------------
    return current_state.rename({
        "node": "particle_id", 
        "target": "ancestor_with_track_id"
    }).filter(pl.col("ancestor_with_track_id").is_not_null())


def set_target_particles_mask(
    particles: pl.DataFrame, 
    ) -> pl.DataFrame:
    """
    Adds a boolean mask 'is_target_particle' to the particles DataFrame.
    A particle is a target if:
    1. It has a track OR it enters the calorimeter.
    2. AND it does not have an ancestor with a track (unless it is the track itself).
    """
    # 0. Get Lineage (External calculation)
    # Ensure this function returns a DataFrame with [event_id, particle_id, ancestor_with_track_id]
    particles_with_track_linage = map_to_nearest_ancestor_with_track(particles)
    
    # 1. Identify Target Particles (Flat List)
    target_particles = (
        particles.lazy()
        .select(["event_id", "particle_id", "enter_calo", "has_track"])
        .explode(["particle_id", "enter_calo", "has_track"])
        .with_columns(pl.col("particle_id").cast(pl.Int64)) # Safety cast
        
        # Condition 1: Enter Calo OR Has Track
        .filter(pl.col("enter_calo") | pl.col("has_track"))
        
        .join(
            particles_with_track_linage.lazy(),
            on=["event_id", "particle_id"],
            how="left"
        )
        
        # Condition 2: No ancestor with track OR is the track itself
        # FIX: Removed the 'True' that was bypassing this check
        .filter(
            pl.col("ancestor_with_track_id").is_null() | pl.col('has_track')
        )
        .select(['event_id', 'particle_id'])
        .unique()
        .with_columns(pl.lit(True).alias('is_target_particle'))
    )

    # 2. Join back to original data efficiently
    return (
        particles.lazy()
        .select(["event_id", "particle_id"])
        .explode("particle_id")
        .with_columns(pl.col("particle_id").cast(pl.Int64)) # Safety cast
        
        # FIX 1: Capture global order
        .with_row_index("global_order")
        
        .join(
            target_particles,
            on=["event_id", "particle_id"],
            how="left"
        )
        .with_columns(pl.col("is_target_particle").fill_null(False))
        
        # FIX 2: Restore order before grouping
        .sort("global_order")
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("is_target_particle"))
        
        .join(
            particles.lazy(),
            on="event_id",
            how="inner"
        )
        .collect(streaming=True)
    )


def backtrack_to_target(
    particles: pl.DataFrame, 
    src_df: pl.DataFrame, 
    target_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Backtracks from src_df particles until it finds an ancestor present in target_df.
    
    Args:
        particles: The full lineage info (event_id, particle_id, parent_id).
        src_df: Where to start (event_id, particle_id).
        target_df: Where to stop (event_id, particle_id).
    
    Returns:
        DataFrame: [event_id, src_particle_id, target_particle_id]
    """
    
    print("Step 1: Preparing the Lookup Map (The World + Stop Signs)...")
    
    # 1. Flatten the world (particles)
    # We need to know everyone's parent.
    flat_particles = (
        particles.lazy()
        .select([
            pl.col("event_id"),
            pl.col("particle_id"),
            pl.col("parent_id"),
            pl.col("is_parent_missing")
        ])
        .explode(["particle_id", "parent_id", "is_parent_missing"])
        .with_columns([
            pl.col("particle_id").cast(pl.Int64),
            pl.col("parent_id").cast(pl.Int64)
        ])
    )

    # 2. Identify Stop Signs (Targets)
    # We need to know which particles are "Targets".
    # We do a semi-join or simple join to mark them.
    targets_marked = (
        target_df.lazy()
        .select([
            pl.col("event_id"), 
            pl.col("particle_id").cast(pl.Int64)
        ])
        .with_columns(pl.lit(True).alias("is_target"))
    )

    # 3. Create the Lookup Table
    # Node -> Next Hop
    lookup_table = (
        flat_particles
        .join(
            targets_marked, 
            on=["event_id", "particle_id"], 
            how="left"
        )
        .select([
            pl.col("event_id"),
            pl.col("particle_id").alias("node"),
            
            # --- THE NAVIGATION LOGIC ---
            pl.when(pl.col("is_target"))
                # If I am a target, I am the destination. Point to Self.
                .then(pl.col("particle_id"))
            .when(pl.col("is_parent_missing"))
                # Dead end
                .then(None)
            .otherwise(
                # Keep searching backwards
                pl.col("parent_id")
            ).alias("next_hop")
        ])
        .collect(streaming=True)
    )

    print("Step 2: Initializing Active Paths from Source...")
    
    # Prepare the walkers starting at src_df
    active_paths = (
        src_df.lazy()
        .select([
            pl.col("event_id"),
            pl.col("particle_id").cast(pl.Int64).alias("src_particle_id"),
            # Initial pointer is the particle itself
            pl.col("particle_id").cast(pl.Int64).alias("current_ptr")
        ])
        .collect(streaming=True)
    )

    print("Step 3: Backtracking Loop...")
    
    iteration = 0
    while True:
        iteration += 1
        
        # Look up the next hop for the current pointer
        next_step = active_paths.join(
            lookup_table,
            left_on=["event_id", "current_ptr"],
            right_on=["event_id", "node"],
            how="left",
            suffix="_jump"
        )
        
        # Check Convergence:
        # We stop if no particle changes its pointer.
        # (Meaning everyone is either pointing to themselves (Target) or Null (Dead End))
        updates = next_step.filter(
            pl.col("next_hop").is_not_null() & 
            (pl.col("next_hop") != pl.col("current_ptr"))
        )
        
        if updates.height == 0:
            print(f"Converged after {iteration} iterations.")
            # Final state update
            active_paths = next_step.select([
                pl.col("event_id"),
                pl.col("src_particle_id"),
                pl.col("next_hop").alias("current_ptr")
            ])
            break
            
        # Apply the jump
        active_paths = next_step.select([
            pl.col("event_id"),
            pl.col("src_particle_id"),
            pl.col("next_hop").alias("current_ptr")
        ])

    print("Step 4: Final validation...")
    
    # It is possible the loop ended because we hit a "Dead End" (Null),
    # NOT a target. We must filter out those cases.
    # We do this by ensuring the result actually exists in the target_df.
    
    # We can rely on the fact that targets point to themselves. 
    # But explicitly checking against target_df is safer/cleaner API.
    
    result = (
        active_paths.lazy()
        .rename({"current_ptr": "target_particle_id"})
        # Inner join with target_df ensures the ID we found is valid
        .join(
            target_df.lazy().select(
                pl.col("event_id"), 
                pl.col("particle_id").cast(pl.Int64).alias("target_particle_id")
            ),
            on=["event_id", "target_particle_id"],
            how="left"
        )
        .collect(streaming=True)
    )

    return result