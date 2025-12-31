from typing import Dict, List
import polars as pl
from sklearn.cluster import MeanShift
import numpy as np
from primary.calibration import CALIBRATION
import awkward as ak
import numpy as np
import polars as pl
from sklearn.cluster import MeanShift
from concurrent.futures import ProcessPoolExecutor
import os
import time
import fastjet
import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl
import numpy as np
import fastjet
from primary.calibration import CALIBRATION

def _fastjet_worker(payload):
    """
    Independent worker that processes a chunk of events.
    Returns two flat arrays: (global_ids, cluster_ids)
    """
    chunk_px, chunk_py, chunk_pz, chunk_E, chunk_gids, R = payload
    
    # Pre-allocate lists for speed
    out_gids = []
    out_cids = []
    
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    
    # Loop over the subset of events assigned to this worker
    for i in range(len(chunk_px)):
        px, py, pz, E, gids = chunk_px[i], chunk_py[i], chunk_pz[i], chunk_E[i], chunk_gids[i]
        
        # 1. Create PseudoJets (Python Loop - unavoidable overhead)
        particles_pj = []
        for j in range(len(px)):
            pj = fastjet.PseudoJet(px[j], py[j], pz[j], E[j])
            pj.set_user_index(int(gids[j])) 
            particles_pj.append(pj)

        # 2. Run FastJet (C++ Speed)
        cs = fastjet.ClusterSequence(particles_pj, jet_def)
        partitions = cs.inclusive_jets(ptmin=0.0)

        # 3. Extract results
        for cluster_id, partition in enumerate(partitions):
            for c in partition.constituents():
                # We simply collect pairs of (GlobalID, ClusterID)
                out_gids.append(c.user_index())
                out_cids.append(cluster_id)

    return np.array(out_gids, dtype=np.int64), np.array(out_cids, dtype=np.int64)

# -------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------
def add_cluster_labels_fastjet(calo_hits: pl.DataFrame, R: float = 0.4) -> pl.DataFrame:
    """
    Parallelized Robust Clustering using all available cores.
    """
    
    # -------------------------------------------------------------------------
    # 1. Explode & Prepare Flat Data (Same as original)
    # -------------------------------------------------------------------------
    print("Preparing data...")
    flat_hits = (
        calo_hits.lazy()
        .select(['event_id', 'x', 'y', 'z', 'total_energy', 'detector'])
        .explode(['x', 'y', 'z', 'total_energy', 'detector'])
        .with_row_index("global_id")
        .join(CALIBRATION.lazy(), on='detector', how="left")
        .with_columns((pl.col('total_energy') * pl.col('calib_factor').fill_null(1.0)).alias('E'))
        .with_columns([
            (pl.col('E') * pl.col('x') / (pl.col('x').pow(2) + pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt()).alias('px'),
            (pl.col('E') * pl.col('y') / (pl.col('x').pow(2) + pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt()).alias('py'),
            (pl.col('E') * pl.col('z') / (pl.col('x').pow(2) + pl.col('y').pow(2) + pl.col('z').pow(2)).sqrt()).alias('pz')
        ])
        .collect()
    )

    # -------------------------------------------------------------------------
    # 2. Prepare Data for Multiprocessing
    # -------------------------------------------------------------------------
    # Sort by event_id so we can slice arrays cleanly
    df_numpy = flat_hits.sort("event_id").select(["event_id", "global_id", "px", "py", "pz", "E"])
    
    event_ids = df_numpy["event_id"].to_numpy()
    
    # Identify event boundaries
    unique_events, split_indices = np.unique(event_ids, return_index=True)
    split_indices = split_indices[1:] # Skip 0
    
    # Split big arrays into list of arrays (one per event)
    # This is fast in Numpy
    events_px  = np.split(df_numpy["px"].to_numpy(), split_indices)
    events_py  = np.split(df_numpy["py"].to_numpy(), split_indices)
    events_pz  = np.split(df_numpy["pz"].to_numpy(), split_indices)
    events_E   = np.split(df_numpy["E"].to_numpy(),  split_indices)
    events_gid = np.split(df_numpy["global_id"].to_numpy(), split_indices)

    # -------------------------------------------------------------------------
    # 3. Parallel Execution
    # -------------------------------------------------------------------------
    num_cores = os.cpu_count()
    # If on a shared node, you might want to limit this, e.g., max(1, num_cores - 1)
    print(f"Distributing {len(unique_events)} events across {num_cores} cores...")

    # Chunk the events list for the workers
    # np.array_split handles uneven division automatically
    chunk_indices = np.array_split(np.arange(len(unique_events)), num_cores)
    
    payloads = []
    for idx_arr in chunk_indices:
        if len(idx_arr) == 0: continue
        start, end = idx_arr[0], idx_arr[-1] + 1
        # Create a payload tuple for each worker
        payloads.append((
            events_px[start:end],
            events_py[start:end],
            events_pz[start:end],
            events_E[start:end],
            events_gid[start:end],
            R
        ))

    results_gids = []
    results_cids = []

    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # map returns results in order of submission
        for gids, cids in executor.map(_fastjet_worker, payloads):
            results_gids.append(gids)
            results_cids.append(cids)

    # Concatenate all worker results
    all_gids = np.concatenate(results_gids)
    all_cids = np.concatenate(results_cids)

    # -------------------------------------------------------------------------
    # 4. Re-Assemble & Join
    # -------------------------------------------------------------------------
    print("Aggregating results...")
    
    # Create the mapping DataFrame
    labels_df = pl.DataFrame({
        "global_id": all_gids,
        "cluster_id": all_cids
    })

    # Since we have the global_id, we can join explicitly or sort.
    # Sorting by global_id restores the original exploded order.
    # Grouping by event_id aggregates the list back.
    
    # Note: We need the original event_ids associated with global_ids. 
    # The safest way is to join back to a lightweight version of the flat_hits.
    
    cluster_lists = (
        flat_hits.lazy()
        .select(["event_id", "global_id"]) # Minimal columns
        .join(labels_df.lazy(), on="global_id", how="left")
        .sort("global_id") # Critical to match original list order
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("cluster_id"))
        .collect()
    )

    return calo_hits.join(cluster_lists, on="event_id", how="left")


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

def add_eta_and_phi_and_pt(particles: pl.DataFrame) -> pl.DataFrame:
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
        map_calo_depositors_to_first_outside_ancestor(particles, calo_hits)
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

def _meanshift_worker_optimized(payload):
    """
    Worker function.
    Payload contains a LIST of events (arrays). 
    This allows processing 1 huge event or 50 small events in one call.
    """
    event_xs, event_ys, event_zs, event_gids, bandwidth = payload
    
    out_gids = []
    out_cids = []
    out_cx = []
    out_cy = []
    out_cz = []
    
    # We use n_jobs=1 here because we are parallelizing OVER events.
    # Using n_jobs=-1 inside a process pool usually causes oversubscription/deadlocks.
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True, n_jobs=1)
    
    for i in range(len(event_xs)):
        # Skip empty
        if len(event_xs[i]) == 0:
            continue
            
        # 1. Prepare Data
        X = np.column_stack((event_xs[i], event_ys[i], event_zs[i])).astype(np.float32)
        
        # 2. Run MeanShift
        ms.fit(X)
        labels = ms.labels_
        centers = ms.cluster_centers_
        
        # 3. Broadcast Centers
        assigned_centers = centers[labels]
        
        # 4. Collect
        out_gids.extend(event_gids[i])
        out_cids.extend(labels)
        out_cx.extend(assigned_centers[:, 0])
        out_cy.extend(assigned_centers[:, 1])
        out_cz.extend(assigned_centers[:, 2])

    return (
        np.array(out_gids, dtype=np.int64),
        np.array(out_cids, dtype=np.int32),
        np.array(out_cx, dtype=np.float32),
        np.array(out_cy, dtype=np.float32),
        np.array(out_cz, dtype=np.float32)
    )

def add_ms_cluster_labels(calo_hits: pl.DataFrame, bandwidth: float = 60.0) -> pl.DataFrame:
    print(f"--- Starting Optimized 3D Clustering (MeanShift, bw={bandwidth}) ---")
    t0 = time.time()

    # -------------------------------------------------------------------------
    # A. Flatten Data & Assign Global IDs
    # -------------------------------------------------------------------------
    # We keep track of event_id to sort/group later
    flat_hits = (
        calo_hits.lazy()
        .select(['event_id', 'x', 'y', 'z'])
        .explode(['x', 'y', 'z'])
        .with_row_index("global_id")
        .collect()
    )

    # Convert to Numpy for efficient splitting
    # Sort by event_id ensures we can use np.split efficiently
    df_numpy = flat_hits.sort("event_id")
    
    all_x = df_numpy["x"].to_numpy()
    all_y = df_numpy["y"].to_numpy()
    all_z = df_numpy["z"].to_numpy()
    all_gid = df_numpy["global_id"].to_numpy()
    all_eid = df_numpy["event_id"].to_numpy()

    # Find boundaries of events
    unique_events, split_indices = np.unique(all_eid, return_index=True)
    split_indices = split_indices[1:] # remove 0

    # Split into list of arrays per event
    events_x = np.split(all_x, split_indices)
    events_y = np.split(all_y, split_indices)
    events_z = np.split(all_z, split_indices)
    events_gid = np.split(all_gid, split_indices)
    
    num_events = len(events_x)
    print(f"Data prepared: {num_events} events. Time: {time.time()-t0:.2f}s")

    # -------------------------------------------------------------------------
    # B. Intelligent Scheduling (LPT - Longest Processing Time First)
    # -------------------------------------------------------------------------
    # 1. Calculate weights (number of points). MeanShift is roughly O(N^2) or O(N log N)
    # We sort by length descending. This solves the "tail" problem.
    lengths = np.array([len(x) for x in events_x])
    
    # Get indices that would sort the array from Largest -> Smallest
    sorted_indices = np.argsort(lengths)[::-1]
    
    # -------------------------------------------------------------------------
    # C. Dynamic Batching
    # -------------------------------------------------------------------------
    # Goal: Huge events get their own task. Tiny events are batched to reduce overhead.
    # Target batch size (in number of points)
    TARGET_BATCH_POINTS = 2000 
    
    payloads = []
    current_batch_indices = []
    current_batch_size = 0
    
    for idx in sorted_indices:
        n_points = lengths[idx]
        
        # If adding this event exceeds target, push current batch first
        # (Unless current batch is empty, then we must take the big one)
        if current_batch_indices and (current_batch_size + n_points > TARGET_BATCH_POINTS):
            # Finalize previous batch
            batch_x = [events_x[i] for i in current_batch_indices]
            batch_y = [events_y[i] for i in current_batch_indices]
            batch_z = [events_z[i] for i in current_batch_indices]
            batch_g = [events_gid[i] for i in current_batch_indices]
            payloads.append((batch_x, batch_y, batch_z, batch_g, bandwidth))
            
            # Reset
            current_batch_indices = []
            current_batch_size = 0

        # Add current event to batch
        current_batch_indices.append(idx)
        current_batch_size += n_points
        
        # If this single event is huge (larger than target), push immediately
        # This ensures specific cores work on this one massive event
        if current_batch_size >= TARGET_BATCH_POINTS:
            batch_x = [events_x[i] for i in current_batch_indices]
            batch_y = [events_y[i] for i in current_batch_indices]
            batch_z = [events_z[i] for i in current_batch_indices]
            batch_g = [events_gid[i] for i in current_batch_indices]
            payloads.append((batch_x, batch_y, batch_z, batch_g, bandwidth))
            
            current_batch_indices = []
            current_batch_size = 0

    # Flush remaining
    if current_batch_indices:
        batch_x = [events_x[i] for i in current_batch_indices]
        batch_y = [events_y[i] for i in current_batch_indices]
        batch_z = [events_z[i] for i in current_batch_indices]
        batch_g = [events_gid[i] for i in current_batch_indices]
        payloads.append((batch_x, batch_y, batch_z, batch_g, bandwidth))

    print(f"Workload optimized: {num_events} events merged into {len(payloads)} tasks.")
    print(f"Largest batch processing first (LPT scheduling).")

    # -------------------------------------------------------------------------
    # D. Parallel Execution
    # -------------------------------------------------------------------------
    num_cores = os.cpu_count()
    res_gids, res_cids, res_cx, res_cy, res_cz = [], [], [], [], []

    # ProcessPoolExecutor naturally handles load balancing if tasks are granular enough
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # We simply map over the payloads. Since payloads are sorted Large->Small,
        # the pool starts with the heavy lifting immediately.
        for r_gid, r_cid, r_cx, r_cy, r_cz in executor.map(_meanshift_worker_optimized, payloads):
            res_gids.append(r_gid)
            res_cids.append(r_cid)
            res_cx.append(r_cx)
            res_cy.append(r_cy)
            res_cz.append(r_cz)

    # -------------------------------------------------------------------------
    # E. Re-Assemble
    # -------------------------------------------------------------------------
    print("Aggregating results...")
    
    # Concatenate all results (order will be scrambled due to sorting/batching)
    # But global_id is preserved, which is our key.
    labels_df = pl.DataFrame({
        "global_id": np.concatenate(res_gids),
        "cluster_id": np.concatenate(res_cids),
        "cluster_cx": np.concatenate(res_cx),
        "cluster_cy": np.concatenate(res_cy),
        "cluster_cz": np.concatenate(res_cz),
    })

    # Join back using global_id
    cluster_lists = (
        flat_hits.lazy()
        .select(["event_id", "global_id"])
        .join(labels_df.lazy(), on="global_id", how="left")
        .sort("global_id") # Critical to maintain list order matching original x,y,z
        .group_by("event_id", maintain_order=True)
        .agg([
            pl.col("cluster_id"),
            pl.col("cluster_cx"),
            pl.col("cluster_cy"),
            pl.col("cluster_cz")
        ])
        .collect()
    )
    
    final_df = calo_hits.join(cluster_lists, on="event_id", how="left")
    
    print(f"Done. Total time: {time.time()-t0:.2f}s")
    return final_df

def cluster_purity(calo_hits_with_clusters:pl.DataFrame, ancestors:pl.DataFrame) -> pl.DataFrame:
    """
    Computes the particle deposited energy ratio in clusters
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

def number_of_particles_per_cluster(calo_hits_with_clusters: pl.DataFrame, ancestors: pl.DataFrame, particles: pl.DataFrame, cut_off_percent: float = 0.05, pt_cut: float = 1.0, eta_cut: float = 3.0) -> pl.DataFrame:
    """
    #particles / cluster
    Computes the number of contributing particles per cluster based on ultimate ancestors.
    Optimized for memory using lazy execution, strict column selection, and window functions.
    
    Args:
        calo_hits_with_clusters: DataFrame with calorimeter hits and cluster information.
        ancestors: DataFrame with particle ancestry information.
        particles: DataFrame with particle properties (pt, eta).
        cut_off_percent: Cutoff percentage for particle contribution filtering (default: 0.05).
        pt_cut: Transverse momentum cut in GeV (default: 1.0).
        eta_cut: Pseudorapidity cut (default: 3.0).
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

    # Prepare Calibration (Lazy)
    calib_lazy = (
        CALIBRATION.lazy()
        .select(['detector', 'calib_factor'])
        # Handle cases where a detector might be missing from the map (default to 1.0)
    )

    # Prepare particles with cuts (Lazy)
    particles_filtered = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'pt', 'eta', 'is_target_particle'])
        .explode(['particle_id', 'pt', 'eta', 'is_target_particle'])
        .with_columns(pl.col('particle_id').cast(pl.Int64))
        .filter((pl.col('pt') > pt_cut) & (pl.col('eta').abs() < eta_cut) & pl.col('is_target_particle') )
        .select(['event_id', 'particle_id'])
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
        # G. Filter by pt and eta cuts
        .join(
            particles_filtered,
            left_on=['event_id', 'ultimate_ancestor_id'],
            right_on=["event_id", "particle_id"],
            how='inner'
        )

        # I. Aggregation (Sum Energies)
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
        # J. Final selection
        .select([
            pl.col('event_id'),
            pl.col('cluster_id'),
            pl.col('num_contributing_ancestors'),
            pl.col('max_ancestor_energy_deps_in_cluster'),
            pl.col('cluster_total_energy')        ])
        .collect(streaming=True)
    )


def number_of_clusters_per_particle(calo_hits_with_clusters: pl.DataFrame, ancestors: pl.DataFrame, particles: pl.DataFrame, cut_off_percent: float = 0.05, pt_cut:float = 1, eta_cut: float=3) -> pl.DataFrame:
    """
    #cluster / particles 
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

    particles_with_clusters = (
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
        .filter((pl.col('total_particle_energy_deps_in_cluster') / pl.col('cluster_total_energy')) > cut_off_percent)
        .group_by(['event_id', 'ultimate_ancestor_id'])
        .agg(
            pl.col('cluster_id').count().alias('num_contributing_clusters'),
            pl.col('total_particle_energy_deps_in_cluster').max().alias('max_ancestor_energy_deps_in_cluster')
        )
        .select([
            pl.col('event_id'),
            pl.col('ultimate_ancestor_id'),
            pl.col('num_contributing_clusters'),
            pl.col('max_ancestor_energy_deps_in_cluster')       ])
        
    )

    return (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_target_particle', 'pt', 'eta'])
        .explode(['particle_id', 'is_target_particle', 'pt', 'eta'])
        .filter((pl.col('is_target_particle'))
                & (pl.col('pt') > pt_cut)
                & (pl.col('eta').abs() < eta_cut))
        .with_columns(pl.col('particle_id').cast(pl.Int64).alias('ultimate_ancestor_id'))
        .join(
            particles_with_clusters,
            on=['event_id', 'ultimate_ancestor_id'],
            how='left'
        )
        .fill_null(0)
    ).collect(streaming=True)

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

def set_target_particles_maskv2(
    particles: pl.DataFrame, 
    eta_cut: float = 3.0,
    pt_cut: float = 1.0
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
    almost_target_particles = (
        particles.lazy()
        .select(["event_id", "particle_id", "enter_calo", "has_track", 'pt', 'eta', 'pdg_id'])
        .explode(["particle_id", "enter_calo", "has_track", 'pt', 'eta', 'pdg_id'])
        .with_columns(pl.col("particle_id").cast(pl.Int64)) # Safety cast
        # Exclude neutrinos from target particles
        .filter((pl.col('pdg_id').abs() != 12) & (pl.col('pdg_id').abs() != 14) & (pl.col('pdg_id').abs() != 16))
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
        .filter((pl.col('pt') > pt_cut) & (pl.col('eta').abs() < eta_cut))
        .select(['event_id', 'particle_id', 'has_track'])
        .unique()
    ).collect(streaming=True)
    back_track_p = (almost_target_particles
    .select(['event_id', 'particle_id', 'has_track'])
    .filter(~pl.col('has_track'))
    )
    back_tracked = backtrack_to_target_roots(
        particles,
        back_track_p.select(['event_id', 'particle_id']),
        back_track_p.select(['event_id', 'particle_id'])
    )

    target_particles = pl.union([almost_target_particles.filter(pl.col('has_track')).select(['event_id', 'particle_id']).unique(),
        back_tracked
        .select(['event_id', 'target_particle_id'])
        .rename({'target_particle_id':'particle_id'})
        .unique()]).with_columns(pl.lit(True).alias('is_target_particle'))
    
    # 2. Join back to original data efficiently
    return (
        particles.lazy()
        .select(["event_id", "particle_id"])
        .explode("particle_id")
        .with_columns(pl.col("particle_id").cast(pl.Int64)) # Safety cast
        
        # FIX 1: Capture global order
        .with_row_index("global_order")
        
        .join(
            target_particles.lazy(),
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

import polars as pl

def backtrack_to_target_roots(
    particles: pl.DataFrame, 
    src_df: pl.DataFrame, 
    target_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Backtracks from src_df particles to find the 'greatest ancestor' 
    (root) that exists within target_df.
    
    Logic:
    - If a particle is in target_df but its parent is NOT, it is a Root (Stop).
    - If a particle is in target_df and its parent is ALSO in target_df, Keep Going.
    - If a particle is not in target_df, Keep Going (trying to find the target layer).
    
    Args:
        particles: The full lineage info (event_id, particle_id, parent_id).
                   Expected to contain Lists if one row per event.
        src_df: Where to start (event_id, particle_id).
        target_df: The subset defining the 'valid' area.
    
    Returns:
        DataFrame: [event_id, src_particle_id, target_particle_id]
    """
    
    print("Step 1: Preparing the Lookup Map (Context aware)...")
    
    # 1. Flatten the world
    # We must explode the columns because 'particles' likely contains lists (1 row per event)
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

    # 2. Identify Targets
    # We just need the list of IDs that count as "Target Territory"
    targets_list = (
        target_df.lazy()
        .select([
            pl.col("event_id"), 
            pl.col("particle_id").cast(pl.Int64)
        ])
        .with_columns(pl.lit(True).alias("in_target"))
    )

    # 3. Create the Lookup Table
    # We need to know if 'Self' is in target AND if 'Parent' is in target.
    lookup_table = (
        flat_particles
        # Join 1: Check if I am in target
        .join(
            targets_list, 
            on=["event_id", "particle_id"], 
            how="left"
        )
        .rename({"in_target": "self_in_target"})
        # Join 2: Check if my Parent is in target
        .join(
            targets_list,
            left_on=["event_id", "parent_id"],
            right_on=["event_id", "particle_id"],
            how="left"
        )
        .rename({"in_target": "parent_in_target"})
        .select([
            pl.col("event_id"),
            pl.col("particle_id").alias("node"),
            
            # --- THE NEW NAVIGATION LOGIC ---
            pl.when(
                # STOP CONDITION:
                # I am in the target group, BUT my parent is not (or doesn't exist).
                # This makes me the "Greatest Ancestor" in the specific dataframe.
                pl.col("self_in_target") & 
                (pl.col("parent_in_target").is_null()) # is_null implies false here due to left join
            )
                .then(pl.col("particle_id"))
                
            .when(pl.col("is_parent_missing"))
                # DEAD END:
                # I am not a root target (failed check above), and I have nowhere to go.
                .then(None)
                
            .otherwise(
                # CONTINUE:
                # Either I am not in target (swim up), 
                # OR I am in target and my parent is too (swim up).
                pl.col("parent_id")
            ).alias("next_hop")
        ])
        .collect(streaming=True)
    )

    print("Step 2: Initializing Active Paths from Source...")
    
    active_paths = (
        src_df.lazy()
        .select([
            pl.col("event_id"),
            pl.col("particle_id").cast(pl.Int64).alias("src_particle_id"),
            pl.col("particle_id").cast(pl.Int64).alias("current_ptr")
        ])
        .collect(streaming=True)
    )

    print("Step 3: Backtracking Loop...")
    
    iteration = 0
    while True:
        iteration += 1
        
        next_step = active_paths.join(
            lookup_table,
            left_on=["event_id", "current_ptr"],
            right_on=["event_id", "node"],
            how="left",
            suffix="_jump"
        )
        
        # Convergence Check:
        # Stop if everyone has settled (next_hop == current_ptr) OR everyone hit a dead end (next_hop is null)
        updates = next_step.filter(
            pl.col("next_hop").is_not_null() & 
            (pl.col("next_hop") != pl.col("current_ptr"))
        )
        
        if updates.height == 0:
            print(f"Converged after {iteration} iterations.")
            active_paths = next_step.select([
                pl.col("event_id"),
                pl.col("src_particle_id"),
                pl.col("next_hop").alias("current_ptr")
            ])
            break
            
        active_paths = next_step.select([
            pl.col("event_id"),
            pl.col("src_particle_id"),
            pl.col("next_hop").alias("current_ptr")
        ])

    print("Step 4: Final validation...")
    
    # We filter to ensure the particle we stopped at is actually in the target_df.
    # (Removes cases where we swam all the way up to a Dead End without hitting a Target Root).
    
    result = (
        active_paths.lazy()
        .rename({"current_ptr": "target_particle_id"})
        .join(
            target_df.lazy().select([
                pl.col("event_id"), 
                pl.col("particle_id").cast(pl.Int64).alias("target_particle_id")
            ]),
            on=["event_id", "target_particle_id"],
            how="inner" # Inner join keeps only valid found roots
        )
        .collect(streaming=True)
    )

    return result

def get_particle_direct_children(particles: pl.DataFrame, event_id: int, particle_id: int) -> pl.DataFrame:
    """
    Returns the direct children of a given particle in a specific event.
    """
    return (
        particles.lazy()
        .select([
            pl.col("event_id"),
            pl.col("particle_id"),
            pl.col("parent_id"),
            pl.col("pdg_id"),
            pl.col("energy"),
            pl.col('vx'), pl.col('vy'), pl.col('vz')
        ])
        .explode(["particle_id", "parent_id", "pdg_id", "energy", 'vx', 'vy', 'vz'])
        .filter(
            (pl.col("event_id") == event_id) &
            (pl.col("parent_id") == particle_id)
        )
        .select(["particle_id", 'event_id', 'pdg_id', 'energy', 'vx', 'vy', 'vz'])
        .collect(streaming=True)
    )

def calculate_extrapolated_features_polars(tracks: pl.DataFrame, B_field=3.0, R_cal_mm=1080.0, Z_cal_mm=3030.0):
    """
    Efficient Polars implementation of track extrapolation + kinematic features.
    
    Adds:
      - track_tanlambda: The slope in the R-Z plane (cot(theta))
      - track_omega: The signed curvature (charge / Radius) [1/mm]
      
    Guarantees:
      - Strict Float32 typing (no panics)
      - Output list order matches input list order exactly
    """
    # 0. Define strict Float32 constants
    f32 = pl.Float32
    alpha = pl.lit(0.0003 * B_field, dtype=f32)
    R_cal = pl.lit(R_cal_mm, dtype=f32)
    Z_cal = pl.lit(Z_cal_mm, dtype=f32)
    v_one = pl.lit(1.0, dtype=f32)
    v_two = pl.lit(2.0, dtype=f32)
    v_epsilon = pl.lit(1e-9, dtype=f32)

    # 1. Start Lazy & Create Index BEFORE Explode
    q = tracks.lazy().with_row_index("event_idx")

    # 2. Select & Explode
    # Flatten lists to apply vectorized math on all particles at once.
    # Order is preserved here implicitly.
    calc_q = q.select(["event_idx", "phi", "theta", "qop", "z0"]) \
              .explode(["phi", "theta", "qop", "z0"])

    # 3. Vectorized Physics Calculations
    calc_q = calc_q.with_columns([
        # Safe qop: avoid division by zero
        pl.when(pl.col("qop").abs() < v_epsilon)
          .then(v_epsilon * pl.col("qop").sign())
          .otherwise(pl.col("qop"))
          .alias("qop_safe")
    ]).with_columns([
        # --- Basic Kinematics ---
        (v_one / pl.col("qop_safe")).abs().alias("p"),
        pl.col("qop_safe").sign().alias("charge"),
        (v_one / pl.col("theta").tan()).alias("cot_theta"),
        
        # --- Regular Eta Calculation ---
        # eta = -ln(tan(theta / 2))
        (-v_one * (pl.col("theta") / v_two).tan().log()).cast(f32).alias("eta")
    ]).with_columns([
        (pl.col("p") * pl.col("theta").sin()).alias("pt")
    ]).with_columns([
        # Radius of Curvature
        (pl.col("pt") / alpha).alias("R_curv")
    ]).with_columns([
        # --- Track Parameters ---
        pl.col("cot_theta").alias("track_tanlambda"),
        
        # Omega: Signed Curvature (charge / Radius)
        (pl.col("charge") / pl.col("R_curv")).fill_nan(0.0).alias("track_omega"),
        
        # --- Extrapolation Logic Starts ---
        (R_cal / (v_two * pl.col("R_curv")))
            .clip(pl.lit(-1.0, dtype=f32), v_one)
            .alias("sin_arg")
    ]).with_columns([
        (v_two * pl.col("sin_arg").arcsin()).alias("delta_phi_barrel")
    ]).with_columns([
        (pl.col("R_curv") * pl.col("delta_phi_barrel")).alias("S_arc_barrel")
    ]).with_columns([
        (pl.col("z0") + pl.col("S_arc_barrel") * pl.col("cot_theta")).alias("z_out_barrel")
    ])

    # 4. Endcap Logic
    calc_q = calc_q.with_columns([
        (pl.col("z_out_barrel").abs() > Z_cal).alias("hits_endcap")
    ]).with_columns([
        (Z_cal * pl.col("z_out_barrel").sign()).alias("z_final_ec"),
        (pl.col("z_out_barrel") - pl.col("z0")).alias("dz_full")
    ]).with_columns([
        (pl.col("z_final_ec") - pl.col("z0")).alias("dz_target")
    ]).with_columns([
        (pl.col("dz_target") / pl.col("dz_full")).alias("ratio")
    ]).with_columns([
        (pl.col("S_arc_barrel") * pl.col("ratio")).alias("S_arc_ec")
    ]).with_columns([
        (pl.col("S_arc_ec") / pl.col("R_curv")).alias("delta_phi_ec")
    ]).with_columns([
        (v_two * pl.col("R_curv") * (pl.col("delta_phi_ec") / v_two).sin()).alias("R_final_ec")
    ])

    # 5. Merge Barrel and Endcap
    calc_q = calc_q.with_columns([
        pl.when(pl.col("hits_endcap"))
          .then(pl.col("z_final_ec"))
          .otherwise(pl.col("z_out_barrel"))
          .alias("z_final"),
        pl.when(pl.col("hits_endcap"))
          .then(pl.col("R_final_ec"))
          .otherwise(R_cal)
          .alias("R_final"),
        pl.when(pl.col("hits_endcap"))
          .then(pl.col("delta_phi_ec"))
          .otherwise(pl.col("delta_phi_barrel"))
          .alias("delta_phi_final")
    ])

    # 6. Final Coordinate Calculation (Int features)
    calc_q = calc_q.with_columns([
        (pl.col("phi") - (pl.col("charge") * pl.col("delta_phi_final"))).alias("phi_raw")
    ]).with_columns([
        pl.arctan2(pl.col("phi_raw").sin(), pl.col("phi_raw").cos()).cast(f32).alias("phi_int"),
        pl.arctan2(pl.col("R_final"), pl.col("z_final")).alias("theta_int")
    ]).with_columns([
        (-v_one * (pl.col("theta_int") / v_two).tan().log()).cast(f32).alias("eta_int")
    ])

    # 7. Implode: Group back to nested lists
    # maintain_order=True guarantees alignment with input lists (e.g. hit_ids)
    results = calc_q.group_by("event_idx", maintain_order=True) \
                    .agg([
                        pl.col("phi_int"),
                        pl.col("eta_int"),
                        pl.col("track_tanlambda"),
                        pl.col("track_omega"),
                        pl.col("pt"),
                        pl.col("eta")  # <--- Added here
                    ])

    # 8. Join back
    return tracks.lazy().with_row_index("event_idx") \
                 .join(results, on="event_idx", how="left") \
                 .drop("event_idx") \
                 .collect()

def preprocess_for_model(particles: pl.DataFrame, tracks: pl.DataFrame, calo_hits: pl.DataFrame, num_of_events: int=-1, eta_cut: float=2.5, pt_cut: float=1.0) -> Dict[str,pl.DataFrame]:
    """
    Aggregates the number of cells per cluster.
    """
    if num_of_events >= 0:
        particles = particles.filter(pl.col("event_id") <num_of_events)
        tracks = tracks.filter(pl.col("event_id") <num_of_events)
        calo_hits = calo_hits.filter(pl.col("event_id") <num_of_events)

    particles = add_orphan_mask(particles)
    particles = add_created_inside_calo_mask(particles)
    particles = add_particle_have_track_mask(particles, tracks)
    particles = add_eta_and_phi_and_pt(particles)
    particles = get_particles_id_parent_of_inside_calo_particles_maskv3(particles, calo_hits)
    particles = set_target_particles_maskv2(particles, eta_cut=eta_cut, pt_cut=pt_cut)
    # apply cuts, filter out tracks related to non target particles

    calo_hits = add_ms_cluster_labels(calo_hits, bandwidth=120.0)

    # Target particle caloremeter calo clusters deposits ---------

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

    target_particles = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_target_particle', 'pdg_id',
                  'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt',
                  'charge','mass', 'has_track'])
        .explode( 'particle_id', 'is_target_particle', 'pdg_id',
                  'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt',
                  'charge','mass', 'has_track')
        .filter(pl.col('is_target_particle'))
        .sort('event_id')
        .with_row_index("global_order")
        .sort('global_order')
        .drop('is_target_particle', 'global_order')
        .group_by('event_id', maintain_order=True)
        .agg('*')
        .collect(streaming=True)
    )

    # 1. OPTIMIZATION: Pre-compute 'is_hcal'
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
        # Deduplicate
        .group_by(['event_id', 'cluster_id'])
        .agg([
            pl.col('cluster_cx').first(),
            pl.col('cluster_cy').first(),
            pl.col('cluster_cz').first(),
        ])
        # Calculate Angles once per cluster
        .with_columns([
            # FIX: Use pl.arctan2(y, x) instead of y.arctan2(x)
            pl.arctan2(pl.col('cluster_cy'), pl.col('cluster_cx')).alias('cluster_phi'),
            
            # Cluster Eta: arcsinh(z / r_perp)
            (pl.col('cluster_cz') / (pl.col('cluster_cx').pow(2) + pl.col('cluster_cy').pow(2)).sqrt())
            .arcsinh()
            .alias('cluster_eta'),
            # rho
            (pl.col('cluster_cx').pow(2) + pl.col('cluster_cy').pow(2)).sqrt().alias('cluster_rho'),
        ])
        .drop(['cluster_cx', 'cluster_cy', 'cluster_cz'])
    )

    # --- BRANCH B: HIT PHYSICS & TOPOLOGY ---
    physics_df = (
        calo_hits.lazy()
        .select(['event_id', 'cluster_id', 'detector', 'total_energy', 'x', 'y', 'z'])
        .explode(['cluster_id', 'detector', 'total_energy', 'x', 'y', 'z'])
        
        # Join Calibration
        .join(calib_optimized, on='detector', how='left')
        
        # Vectorized Math (Hit Level)
        .with_columns([
            (pl.col('total_energy') * pl.col('calib_factor')).alias('cal_E'),
            (pl.col('x').pow(2) + pl.col('y').pow(2)).sqrt().alias('hit_rho') 
        ])
        .with_columns([
            # Hit Eta
            (pl.col('z') / pl.col('hit_rho')).arcsinh().alias('hit_eta'),
            # FIX: Use pl.arctan2(y, x) here as well
            pl.arctan2(pl.col('y'), pl.col('x')).alias('hit_phi'),
        ])
        
        # Aggregation
        .group_by(['event_id', 'cluster_id'])
        .agg([
            pl.col('cal_E').sum().alias('total_cluster_energy'),
            pl.col('cal_E').filter(pl.col('is_hcal')).sum().alias('hcal_energy'),
            
            # Topological Widths
            pl.col('hit_eta').std().fill_null(0.0).alias('sigma_eta'),
            pl.col('hit_phi').std().fill_null(0.0).alias('sigma_phi'),
            pl.col('hit_rho').std().fill_null(0.0).alias('sigma_rho'),
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
            .alias('hcal_fraction')
        )
        .sort(['event_id', 'cluster_id'])
        .group_by('event_id', maintain_order=True)
        .agg(pl.all())
        .collect(streaming=True)
    )
    target_particles_idx =  (
        target_particles.lazy()
        .select(['event_id', 'particle_id'])
        .explode('particle_id')
        .with_row_index('particle_idx')
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('particle_id'),
            (pl.col('particle_idx') - pl.col('particle_idx').min()).alias('particle_idx')
        ])
        .explode(['particle_id', 'particle_idx'])
        .collect()
    )
    cluster_to_cluster_idx = (
        calo_clusters.lazy()
        .select(['event_id', 'cluster_id'])
        .explode('cluster_id')
        .with_row_index('cluster_idx')
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('cluster_id'),
            (pl.col('cluster_idx') - pl.col('cluster_idx').min()).alias('cluster_idx')
        ])
        .explode(['cluster_id', 'cluster_idx'])
        .collect()
    )
    

    points_to_target = backtrack_to_target(particles=particles,
                       src_df=depositors_list,
                       target_df=target_particles_idx.select(['event_id', 'particle_id']))
    target_particles_deps = cluster_purity(calo_hits_with_clusters=calo_hits, ancestors=points_to_target)
    target_particles_deps_aggrigated = (target_particles_deps.lazy()
                                        .select(['event_id', 'cluster_id', 'ultimate_ancestor_id', 'total_energy_deps_in_cluster'])
                                        .rename({'ultimate_ancestor_id':'particle_id'})
                                        .join(
                                            target_particles_idx.lazy(),
                                            on=['event_id', 'particle_id'],
                                            how='left'
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
    
    # ----------------------------------------------
    tracks = calculate_extrapolated_features_polars(tracks)
    # change particle id to particle idx
    tracks_mappings = (
        tracks.lazy()
        .select(['event_id', 'majority_particle_id'])
        .with_columns(
            # FIX: Use 'int_ranges' (plural) to generate a range for every row
            local_order=pl.int_ranges(
                start=0,
                end=pl.col('majority_particle_id').list.len(), 
                dtype=pl.UInt32
            )
        )
        .explode(['majority_particle_id', 'local_order'])
        .rename({'majority_particle_id': 'particle_id'})
        .join(
            target_particles_idx.lazy(),
            on=['event_id', 'particle_id'],
            how='inner'
        )
        .group_by('event_id')
        .agg(
            pl.col('particle_idx').sort_by('local_order')
        )
    )

    # 2. Apply to original tracks
    tracks = (
        tracks.lazy()
        .drop('majority_particle_id') 
        .join(
            tracks_mappings, 
            on='event_id', 
            how='inner'
        )
        .collect(streaming=True)
    )
    # ----------------------------------------------
    return {
        "target_particles": target_particles,
        "calo_clusters": calo_clusters,
        "tracks": tracks,
        "target_particles_deps": target_particles_deps_aggrigated
    }



import polars as pl

