"""Particle-level preprocessing primitives.

Ported verbatim from ``primary/preprocessing.py`` on the ``master`` branch,
pruned to the 14 functions that the dataset pipeline actually reaches. The
1512 lines of unreachable code on master (alternative fastjet/MeanShift
clustering, the v2/v3 mask variants, purity/counting diagnostics) are not
carried over; with them go the ``torch``, ``fastjet``, ``awkward`` and
``sklearn.cluster`` dependencies.

Function bodies are unmodified so that output stays bit-identical to the
original scripts -- see ``tests/test_equivalence_frames.py``.

Rough call order in the pipeline (see :mod:`colliderml_pflow.pipeline`):

Stage A, every source
    :func:`calculate_extrapolated_features_polars`, :func:`add_eta_and_phi_and_pt`

Stage A, primary source only
    :func:`add_orphan_mask`, :func:`add_created_inside_calo_mask`,
    :func:`add_particle_have_track_mask`,
    :func:`get_particles_id_parent_of_inside_calo_particles_maskv3`,
    :func:`set_target_particles_maskv4`

Stage C, shared tail
    :func:`backtrack_to_target`, :func:`cluster_contrib_energy`,
    :func:`cluster_vertex_primary_deps`, :func:`cluster_purity`

The remaining three (:func:`map_calo_depositors_to_first_outside_ancestor`,
:func:`map_to_nearest_ancestor_with_track`, :func:`backtrack_to_target_roots`)
are internal helpers called by the functions above.
"""

import polars as pl

from colliderml_pflow.calibration import CALIBRATION
from colliderml_pflow.pdg import unstable_pdg_ids_df



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
        # make sure it is inside particles df (pileup stuff)
        .join(
            particles.lazy().select(['event_id', 'particle_id']).explode('particle_id').with_columns(pl.col('particle_id')),
            left_on=['event_id', 'particle_id'],
            right_on=['event_id', 'particle_id'],
            how='inner'
        )
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


def cluster_contrib_energy(calo_hits_with_clusters: pl.DataFrame) -> pl.DataFrame:
    """
    Per-(event, cluster, contributing_particle) calibrated energy.

    Performs the heavy double-explode of `contrib_particle_ids` and
    `contrib_energies` once, applies the cell-level calibration via
    CALIBRATION, then immediately collapses to per-(event, cluster, particle)
    sums so the result is far smaller than the per-contribution flat form.

    Both `cluster_purity` and `cluster_vertex_primary_deps` consume this so
    the explode runs only once.

    Returns:
        Flat DataFrame: event_id, cluster_id, particle_id (Int64), cal_E (Float32, GeV).
    """
    calib_lazy = CALIBRATION.lazy().select(['detector', 'calib_factor'])

    return (
        calo_hits_with_clusters.lazy()
        .select(['event_id', 'cluster_id', 'detector',
                 'contrib_particle_ids', 'contrib_energies'])
        # Level-1 explode: per cell
        .explode(['cluster_id', 'detector',
                  'contrib_particle_ids', 'contrib_energies'])
        .filter(pl.col('cluster_id') >= 0)
        # Calibration at cell level (cheaper than after the second explode)
        .join(calib_lazy, on='detector', how='left')
        # Level-2 explode: per contributor
        .explode(['contrib_particle_ids', 'contrib_energies'])
        .with_columns([
            (pl.col('contrib_energies') * pl.col('calib_factor'))
                .cast(pl.Float32).alias('cal_E'),
            pl.col('contrib_particle_ids').cast(pl.Int64).alias('particle_id'),
        ])
        # Reduce immediately — sum cal_E per (event, cluster, particle)
        .group_by(['event_id', 'cluster_id', 'particle_id'])
        .agg(pl.col('cal_E').sum().alias('cal_E'))
        .collect(streaming=True)
    )


def cluster_purity(contrib_energy: pl.DataFrame, ancestors: pl.DataFrame) -> pl.DataFrame:
    """
    Computes the purity/efficiency of each cluster based on ultimate ancestors.

    Consumes the shared `cluster_contrib_energy` intermediate so the heavy
    double-explode is not repeated.
    """
    ancestors_lazy = (
        ancestors.lazy()
        .select(['event_id', 'src_particle_id', 'target_particle_id'])
        .rename({
            'src_particle_id': 'particle_id',
            'target_particle_id': 'ultimate_ancestor_id',
        })
        .with_columns(pl.col('particle_id').cast(pl.Int64))
    )

    return (
        contrib_energy.lazy()
        .join(ancestors_lazy, on=['event_id', 'particle_id'], how='left')
        .group_by(['event_id', 'cluster_id', 'ultimate_ancestor_id'])
        .agg(pl.col('cal_E').sum().alias('total_energy_deps_in_cluster'))
        .with_columns(
            pl.col('total_energy_deps_in_cluster')
            .sum()
            .over(['event_id', 'ultimate_ancestor_id'])
            .alias('total_energy_deps')
        )
        .select([
            pl.col('event_id'),
            pl.col('cluster_id'),
            pl.col('ultimate_ancestor_id'),
            pl.col('total_energy_deps_in_cluster'),
            pl.col('total_energy_deps'),
            (pl.col('total_energy_deps_in_cluster') / pl.col('total_energy_deps'))
            .alias('purity'),
        ])
        .collect(streaming=True)
    )


def cluster_vertex_primary_deps(
    contrib_energy: pl.DataFrame,
    pid_to_vertex: pl.DataFrame,
    cluster_to_cluster_idx: pl.DataFrame,
) -> pl.DataFrame:
    """
    Per-(event, cluster) calibrated energy aggregated by primary vertex.

    Args:
        contrib_energy: output of `cluster_contrib_energy`.
        pid_to_vertex:  flat DataFrame [event_id, particle_id (Int64),
                        vertex_primary (UInt16)].
        cluster_to_cluster_idx: flat DataFrame [event_id, cluster_id, cluster_idx]
                        matching the cluster ordering used in calo_clusters.

    Returns:
        One row per event with columns:
            vertex_primary_indices   List[List[UInt16]]
            vertex_primary_energies  List[List[Float32]]   # GeV, calibrated
        Outer list position i corresponds to cluster i in calo_clusters
        (same cluster ordering via cluster_to_cluster_idx).
        Inner lists are sorted by vertex_primary so the two lists pair
        unambiguously: vp_indices[i][j] deposited vp_energies[i][j] GeV
        in cluster i.
    """
    return (
        contrib_energy.lazy()
        .join(pid_to_vertex.lazy(),
              on=['event_id', 'particle_id'], how='left')
        # Sum calibrated energy per (event, cluster, vertex)
        .group_by(['event_id', 'cluster_id', 'vertex_primary'])
        .agg(pl.col('cal_E').sum().alias('vertex_energy'))
        # Map cluster_id -> cluster_idx so order matches calo_clusters
        .join(cluster_to_cluster_idx.lazy(),
              on=['event_id', 'cluster_id'], how='left')
        .drop('cluster_id')
        # Inner agg: per (event, cluster) -> two parallel lists
        .sort(['event_id', 'cluster_idx', 'vertex_primary'])
        .group_by(['event_id', 'cluster_idx'], maintain_order=True)
        .agg([
            pl.col('vertex_primary').alias('vertex_primary_indices'),
            pl.col('vertex_energy').cast(pl.Float32).alias('vertex_primary_energies'),
        ])
        # Outer agg: per event -> list of clusters, ordered by cluster_idx
        .sort(['event_id', 'cluster_idx'])
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('vertex_primary_indices'),
            pl.col('vertex_primary_energies'),
        ])
        .collect(streaming=True)
    )


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


def set_target_particles_maskv4(
    particles: pl.DataFrame, 
    tracks: pl.DataFrame,
    truth_eta_cut: float = 3.0,
    truth_pt_cut: float = 1.0,
    target_pt_cut: float = 0.2
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
        .select(['event_id', 'particle_id', 'has_track', 'pdg_id'])
        .unique()
    ).collect(streaming=True)

    # Now we proceed of removing non-stable particle that appear to be in target because they have low energetic children that are not saved
    # in truth records, and the odd guys attribute the caloremeter dep to the non stable particle
    # Logic - filter just those unstable ones
    # drawback - if they have all decendants < 100 Mev, it will bw ignored completetly.
    unstables_series = unstable_pdg_ids_df.select('pdg_id').to_series()
    almost_target_particles = almost_target_particles.filter(
        (pl.col('has_track')) | (~pl.col('pdg_id').is_in(unstables_series))
    )
    # --------------------------------
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
    # Now apply cuts!
    # Accept target if  truth ancestor has pt>pt_cut and target particle pt>pt_cut/3
    # BUT if it's tracked particle than filtering logic should be applied based on track info also
    particle_roots_no_parents = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_parent_missing'])
        .explode('particle_id','is_parent_missing')
        .filter(pl.col('is_parent_missing') 
                )
        .select(['event_id', 'particle_id'])
    ).collect(streaming=True)

    mappings_target_to_truth =  backtrack_to_target(particles=particles, src_df=target_particles, target_df=particle_roots_no_parents).rename({'src_particle_id':'target_particle_id','target_particle_id':'truth_particle_id'}) 
    particles_with_pt_eta = particles.lazy().select(['event_id', 'particle_id', 'pt', 'eta']).explode(['particle_id', 'pt', 'eta'])

    target_particles =(particles_with_pt_eta.rename({'particle_id':'target_particle_id', 'pt':'pt_target', 'eta':'eta_target'})
                    .join(
                        mappings_target_to_truth.lazy(),
                        left_on=['event_id', 'target_particle_id'],
                        right_on=['event_id', 'target_particle_id'],
                        how='inner',
                    )
                    .join(
                        particles_with_pt_eta.rename({'particle_id':'truth_particle_id', 'pt':'pt_truth', 'eta':'eta_truth'})
                        ,
                        left_on=['event_id', 'truth_particle_id'],
                        right_on=['event_id', 'truth_particle_id'],
                        how='inner',
                    )
                    .join(
                        (tracks.lazy().select(['event_id', 'majority_particle_id', 'pt', 'eta']).rename({'majority_particle_id':'particle_id', 'pt':'track_pt', 'eta':'track_eta'})
                        .explode(['particle_id', 'track_pt', 'track_eta'])),
                        left_on=['event_id', 'target_particle_id'],
                        right_on=['event_id', 'particle_id'],
                        how='left'
                    )
                    .filter(
                        (((pl.col('pt_truth') > truth_pt_cut) & (pl.col('pt_target') > target_pt_cut) & (pl.col('eta_truth').abs() < truth_eta_cut)) |
                         
                         (pl.col('track_pt').is_not_null() & (pl.col('track_pt') > target_pt_cut) & (pl.col('track_eta').abs() < truth_eta_cut))
                         )
                    )
                    .drop('track_pt', 'track_eta')
                    .select(['event_id', 'target_particle_id'])
                    .unique()
                    .rename({'target_particle_id':'particle_id'})
                    .with_columns(pl.lit(True).alias('is_target_particle'))
                    ).collect(streaming=True)
    
    truth_particles = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_parent_missing', 'eta', 'pt', 'pdg_id'])
        .explode('particle_id','is_parent_missing', 'eta', 'pt', 'pdg_id')
        .filter(pl.col('is_parent_missing') 
                & (pl.col('pdg_id').abs() != 12) & (pl.col('pdg_id').abs() != 14) & (pl.col('pdg_id').abs() != 16)
                & (pl.col('eta').abs() < truth_eta_cut)
                & (pl.col('pt') > truth_pt_cut)
                )
        .select(['event_id', 'particle_id'])
        .with_columns(pl.lit(True).alias('is_truth_particle'))
    ).collect(streaming=True)
    # 2. Join back target to original data efficiently
    result = (
        particles.lazy()
        .select(["event_id", "particle_id"])
        .explode("particle_id")
        .with_columns(pl.col("particle_id").cast(pl.Int64)) # Safety cast
        
        # FIX 1: Capture global order
        .with_row_index("global_order")
        
        .join(
            truth_particles.lazy(),
            on=["event_id", "particle_id"],
            how="left"
        )
        .with_columns(pl.col("is_truth_particle").fill_null(False))
        
        # FIX 2: Restore order before grouping
        .sort("global_order")
        .group_by("event_id", maintain_order=True)
        .agg(pl.col("is_truth_particle"))
        
        .join(
            particles.lazy(),
            on="event_id",
            how="inner"
        )
        .collect(streaming=True)
    )

    result = (
        result.lazy()
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
            result.lazy(),
            on="event_id",
            how="inner"
        )
        .collect(streaming=True)
    )
    return result


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

