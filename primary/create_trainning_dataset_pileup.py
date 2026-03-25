from typing import Dict
import polars as pl
import yaml # type: ignore
import gc

from sklearn.model_selection import train_test_split
from primary.preprocessing import add_eta_and_phi_and_pt, add_eta_and_phi_and_pt, \
     add_orphan_mask, add_created_inside_calo_mask, add_particle_have_track_mask, set_target_particles_maskv4, get_particles_id_parent_of_inside_calo_particles_maskv3, \
    add_eta_and_phi_and_pt, backtrack_to_target, cluster_purity, calculate_extrapolated_features_polars
from primary.calibration import CALIBRATION
from primary.clue_clustering import clue_clustering



def split_train_val_test(datasets: Dict[str, pl.DataFrame], train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)->Dict[str, Dict[str, pl.DataFrame]]:
    """
    Splits the dataset into training, validation, and test sets based on the provided fractions.

    Args:
        dataset 
        train_frac (float): Fraction of data to be used for training.
        val_frac (float): Fraction of data to be used for validation.
        test_frac (float): Fraction of data to be used for testing.
        seed (int): Random seed for reproducibility.
    """
    event_ids = datasets['target_particles']['event_id'].unique().to_numpy()
    n_events = len(event_ids)
    train_ids, temp_ids = train_test_split(event_ids, train_size=train_frac, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_frac/(val_frac + test_frac), random_state=seed)

    # Map the split names to the generated ID arrays
    split_mapping = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    # Initialize the output structure
    results = {
        "train": {},
        "val": {},
        "test": {}
    }

    # Iterate through each specific dataset (particles, clusters, tracks, etc.)
    for key, df in datasets.items():
        # Apply the filters for each split (train/val/test)
        for split_name, ids in split_mapping.items():
            # Use Polars .is_in() to filter rows efficiently
            results[split_name][key] = df.filter(pl.col("event_id").is_in(ids))

    return results

def generate_normalization_yaml(data: Dict[str, pl.DataFrame]) -> str:
    """
    Generates a normalization config YAML string based on the provided data.
    """
    
    # Define the schema: (Dataframe Name, Column Name, Transform, Type)
    config_schema = {
        "eta":        {"df": "calo_clusters", "col": "cluster_eta", "transform": None, "type": "min_max_sym"},
        "rho":        {"df": "calo_clusters", "col": "cluster_rho", "transform": None, "type": "min_max_sym"},
        "e":          {"df": "calo_clusters", "col": "total_cluster_energy", "transform": "sqrt", "type": "min_max_sym"},
        "pt":         {"df": "tracks",        "col": "pt",          "transform": "sqrt", "type": "min_max_sym"},
        "cluster_pt": {"df": "calo_clusters", "col": None,         "transform": "sqrt", "type": "min_max_sym"},
        "sigma_eta":  {"df": "calo_clusters", "col": "sigma_eta",   "transform": "sqrt", "type": "std"},
        "sigma_phi":  {"df": "calo_clusters", "col": "sigma_phi",   "transform": "sqrt", "type": "std"},
        "sigma_rho":  {"df": "calo_clusters", "col": "sigma_rho",   "transform": "sqrt", "type": "std"},
        "d0":         {"df": "tracks",        "col": "d0",          "transform": None,   "type": "min_max_sym"},
        "z0":         {"df": "tracks",        "col": "z0",          "transform": None,   "type": "min_max_sym"},
        "tanlambda":  {"df": "tracks",        "col": "track_tanlambda", "transform": None, "type": "min_max_sym"},
        "omega":      {"df": "tracks",        "col": "track_omega", "transform": None,   "type": "std"},
    }
    
    yaml_config = {}
    
    for key, schema in config_schema.items():
        df_name = schema["df"]
        col_name = schema["col"]
        transform = schema["transform"]
        config_type = schema["type"]
        
        if df_name not in data:
            continue
            
        df = data[df_name]
        
        if key == "cluster_pt":
            # Special case: compute pt from energy and eta
            energy_series = df.select(pl.col("total_cluster_energy")).explode("total_cluster_energy").get_column("total_cluster_energy")
            eta_series = df.select(pl.col("cluster_eta")).explode("cluster_eta").get_column("cluster_eta")
            series = energy_series / eta_series.cosh()
        else:
            if col_name not in df.columns:
                continue

            try:
                # Check if column is List type and explode if so
                is_list = False
                dtype = df.schema[col_name]
                if isinstance(dtype, pl.List):
                    is_list = True
                
                if is_list:
                    series = df.select(pl.col(col_name).explode()).get_column(col_name)
                else:
                    series = df.get_column(col_name)
            except Exception:
                 raise ValueError(f"Column {col_name} not found in DataFrame {df_name}")

        
        if len(series) == 0:
            continue
            
        if transform == "sqrt":
            series = series.sqrt()
            
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        
        def smart_fmt(val):
             if abs(val) >= 10 and abs(round(val) - val) < 1e-3:
                  return int(round(val))
             return float(f"{val:.4f}")

        entry = {
            "type": config_type,
            "mean": float(f"{mean_val:.4f}"),
            "std": float(f"{std_val:.4f}"),
            "min": smart_fmt(min_val),
            "max": smart_fmt(max_val)
        }
        
        if transform:
            entry["fn"] = transform
            
        # Reconstruct to ensure order: type, fn (optional), mean, std, min, max
        ordered_entry = {}
        ordered_entry["type"] = entry["type"]
        if "fn" in entry:
            ordered_entry["fn"] = entry["fn"]
        ordered_entry["mean"] = entry["mean"]
        ordered_entry["std"] = entry["std"]
        ordered_entry["min"] = entry["min"]
        ordered_entry["max"] = entry["max"]
        
        yaml_config[key] = ordered_entry
        
    return yaml.dump(yaml_config, sort_keys=False, default_flow_style=False)


def generate_normalization_stats_sequential(data_dir: str) -> str:
    """
    Generates normalization stats sequentially from parquet files in a directory.
    Memory-efficient alternative to generate_normalization_yaml.
    """
    from pathlib import Path
    from tqdm import tqdm
    import numpy as np
    
    # Define the schema: (Dataframe Name, Column Name, Transform, Type)
    config_schema = {
        "eta":        {"df": "calo_clusters", "col": "cluster_eta", "transform": None, "type": "min_max_sym"},
        "rho":        {"df": "calo_clusters", "col": "cluster_rho", "transform": None, "type": "min_max_sym"},
        "e":          {"df": "calo_clusters", "col": "total_cluster_energy", "transform": "sqrt", "type": "min_max_sym"},
        "pt":         {"df": "tracks",        "col": "pt",          "transform": "sqrt", "type": "min_max_sym"},
        "cluster_pt": {"df": "calo_clusters", "col": None,         "transform": "sqrt", "type": "min_max_sym"},
        "sigma_eta":  {"df": "calo_clusters", "col": "sigma_eta",   "transform": "sqrt", "type": "std"},
        "sigma_phi":  {"df": "calo_clusters", "col": "sigma_phi",   "transform": "sqrt", "type": "std"},
        "sigma_rho":  {"df": "calo_clusters", "col": "sigma_rho",   "transform": "sqrt", "type": "std"},
        "d0":         {"df": "tracks",        "col": "d0",          "transform": None,   "type": "min_max_sym"},
        "z0":         {"df": "tracks",        "col": "z0",          "transform": None,   "type": "min_max_sym"},
        "tanlambda":  {"df": "tracks",        "col": "track_tanlambda", "transform": None, "type": "min_max_sym"},
        "omega":      {"df": "tracks",        "col": "track_omega", "transform": None,   "type": "std"},
    }
    
    # Initialize accumulators
    stats = {}
    for key in config_schema:
        stats[key] = {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float('inf'),
            "max": float('-inf')
        }

    path = Path(data_dir)
    # Search for track files to determine chunks
    # Assuming standard naming: tracks-{index}.parquet
    # We look for tracks-*.parquet as the driver for indices
    track_files = sorted(list(path.glob("tracks-*.parquet")))
    
    if not track_files:
        print(f"No tracks-*.parquet files found in {data_dir}. Cannot generate stats.")
        return ""

    indices = []
    for f in track_files:
        try:
            # Extract index between last '-' and '.'
            indices.append(f.name.split('-')[-1].split('.')[0])
        except Exception:
            pass

    # Process files sequentially
    for idx_str in tqdm(indices, desc="Computing Normalization Stats"):
        
        # Identify which dataframes we need for the schema
        required_dfs = set(v["df"] for v in config_schema.values())
        loaded_dfs = {}
        
        # Load necessary files for this index
        for df_name in required_dfs:
            fpath = path / f"{df_name}-{idx_str}.parquet"
            if fpath.exists():
                loaded_dfs[df_name] = pl.read_parquet(fpath)
            
        # Compute stats for each schema entry
        for key, schema in config_schema.items():
            df_name = schema["df"]
            col_name = schema["col"]
            transform = schema["transform"]
            
            if df_name not in loaded_dfs:
                continue
                
            df = loaded_dfs[df_name]
            
            series = None
            
            # Special handling for cluster_pt
            if key == "cluster_pt":
                if "total_cluster_energy" in df.columns and "cluster_eta" in df.columns:
                    e = df.select(pl.col("total_cluster_energy").explode()).get_column("total_cluster_energy")
                    eta = df.select(pl.col("cluster_eta").explode()).get_column("cluster_eta")
                    series = e / eta.cosh()
            else:
                if col_name not in df.columns:
                    continue
                
                # Check if list and explode
                dtype = df.schema[col_name]
                if isinstance(dtype, pl.List):
                    series = df.select(pl.col(col_name).explode()).get_column(col_name)
                else:
                    series = df.get_column(col_name)

            if series is None or len(series) == 0:
                continue
                
            if transform == "sqrt":
                series = series.sqrt()
            
            # Convert to numpy for accumulation
            arr = series.to_numpy()
            
            # Remove NaNs / Infs
            arr = arr[np.isfinite(arr)]
            
            if len(arr) == 0:
                continue
                
            n = len(arr)
            s = np.sum(arr)
            ss = np.sum(arr * arr)
            mn = np.min(arr)
            mx = np.max(arr)
            
            stats[key]["count"] += n
            stats[key]["sum"] += s
            stats[key]["sum_sq"] += ss
            if mn < stats[key]["min"]:
                stats[key]["min"] = mn
            if mx > stats[key]["max"]:
                stats[key]["max"] = mx

    # Finalize
    yaml_config = {}
    
    def smart_fmt(val):
         if isinstance(val, (int, float, np.number)):
            if abs(val) >= 10 and abs(round(val) - val) < 1e-3:
                  return int(round(val))
            return float(f"{val:.4f}")
         return val

    for key, schema in config_schema.items():
        s = stats[key]
        N = s["count"]
        
        if N == 0:
            continue
            
        mean_val = s["sum"] / N
        var_val = (s["sum_sq"] - (s["sum"]**2 / N)) / (N - 1) if N > 1 else 0.0
        if var_val < 0: var_val = 0.0
        std_val = np.sqrt(var_val)
        
        entry = {
            "type": schema["type"],
            "mean": float(f"{mean_val:.4f}"),
            "std": float(f"{std_val:.4f}"),
            "min": smart_fmt(s["min"]),
            "max": smart_fmt(s["max"])
        }
        
        if schema["transform"]:
            entry["fn"] = schema["transform"]
        
        # Order keys
        ordered = {}
        ordered["type"] = entry["type"]
        if "fn" in entry: ordered["fn"] = entry["fn"]
        ordered["mean"] = entry["mean"]
        ordered["std"] = entry["std"]
        ordered["min"] = entry["min"]
        ordered["max"] = entry["max"]
        
        yaml_config[key] = ordered

    return yaml.dump(yaml_config, sort_keys=False, default_flow_style=False)





def filter_orphans_and_reindex(
    target_particles: pl.DataFrame,
    target_particles_deps: pl.DataFrame,
    tracks: pl.DataFrame,
    cluster_to_cluster_idx: pl.DataFrame
) -> Dict[str, pl.DataFrame]:
    """
    Filters out orphan target particles (those with no tracks and no cluster deposits).
    Re-indexes particles after filtering and updates dependencies and tracks.
    Prints statistics about filtered orphans.
    """
    
    # 1. Calculate Initial Statistics
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

    # 2. Identify valid particles (those in deps or tracks)
    
    # Particles from deps
    ids_in_deps = (
        target_particles_deps.lazy()
        .select(['event_id', 'ultimate_ancestor_id'])
        .rename({'ultimate_ancestor_id': 'particle_id'})
        .filter(pl.col('particle_id').is_not_null())
        .unique()
    )
    
    # Particles from tracks
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

    # 3. Filter target_particles to remove orphans
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

    # 4. Calculate Final Statistics & Print
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
    
    # Avoid division by zero
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


    # 5. Create Mapping (particle_id -> particle_idx) based on filtered particles
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

    # 6. Update target_particles_deps with new particle_idx
    target_particles_deps_aggrigated = (target_particles_deps.lazy()
                                        .select(['event_id', 'cluster_id', 'ultimate_ancestor_id', 'total_energy_deps_in_cluster'])
                                        .rename({'ultimate_ancestor_id':'particle_id'})
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
    
    # 7. Update tracks : change particle id to particle idx
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
            particle_mapping,
            on=['event_id', 'particle_id'],
            how='left'
        )
        .with_columns(
            pl.col('particle_idx').fill_null(-1) # Mark tracks that lost their particle mapping (orphans) with -1
        )
        .group_by('event_id', maintain_order=True)
        .agg(
            pl.col('particle_idx').sort_by('local_order')
        )
    )

    # Apply to original tracks
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
        "tracks": tracks_updated
    }


def create_calo_clusters(calo_hits: pl.DataFrame) -> pl.DataFrame:
    """
    Computes cluster features (centroids, geometry, physics) from calorimeter hits.
    """
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
        .filter(pl.col('cluster_id')>=0)
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
    
    return calo_clusters


def preprocess_for_model(particles: pl.DataFrame, tracks: pl.DataFrame, calo_hits: pl.DataFrame,

                         num_of_events: int=-1,  truth_eta_cut: float=3.0, truth_pt_cut: float=1.0, target_pt_cut: float=0.3, clusters_cutoff: float=0.1) -> Dict[str,pl.DataFrame]:
    """
    Aggregates the number of cells per cluster.
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    print("\n[PREPROCESS START]")
    print(f"RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    if num_of_events >= 0:
        print(f"[FILTERING] Filtering to {num_of_events} events...")
        particles = particles.filter(pl.col("event_id") <num_of_events)
        tracks = tracks.filter(pl.col("event_id") <num_of_events)
        calo_hits = calo_hits.filter(pl.col("event_id") <num_of_events)
        print(f"[FILTERING DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # Cast to Float32
    print("[CASTING] Converting Float64 to Float32...")
    particles = particles.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32))
    ])
    tracks = tracks.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32))
    ])
    calo_hits = calo_hits.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32))
    ])
    print("[EXTRAPOLATED FEATURES] Calculating extrapolated track features...")
    tracks = calculate_extrapolated_features_polars(tracks)

    print(f"[EXTRAPOLATED FEATURES DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    print("[PROCESSING TRACKS] Filtering and processing tracks...")
    track_cols = [c for c in tracks.columns if c != 'event_id']
    tracks = (
        tracks.lazy()
        .with_columns(
            local_order=pl.int_ranges(
                start=0,
                end=pl.col('majority_particle_id').list.len(), 
                dtype=pl.UInt32
            )
        )
        .select(['event_id', 'local_order'] + track_cols)
        .explode(['local_order'] + track_cols)
        .filter(pl.col('pt') > truth_pt_cut)
        .filter(pl.col('eta').abs() < truth_eta_cut)
        .join(
            particles.lazy().select(['event_id', 'particle_id', 'vertex_primary']).explode('particle_id', 'vertex_primary'),
            left_on=['event_id', 'majority_particle_id'],
            right_on=['event_id', 'particle_id'],
            how='left'
        )
        .with_columns(pl.col('majority_particle_id').cast(pl.Int64))
        .sort(['event_id', 'local_order'])
        .group_by('event_id', maintain_order=True)
        .agg([pl.col(c) for c in track_cols]+ [pl.col('vertex_primary')])
        .sort('event_id')
        .collect(streaming=True)
    )

    particles = (particles.lazy().with_columns(
                # 1. Find the INDICES strictly inside list.eval()
                # (pl.element() == 1) creates a boolean mask
                # .arg_true() converts that mask to indices
                _indices = pl.col("vertex_primary").list.eval(
                    (pl.element() == 1).arg_true()
                )
            ).with_columns(
                # 2. Use those indices to pick elements from all other list columns
                pl.exclude("event_id", "_indices")
                .list.gather(pl.col("_indices"))
            ).drop("_indices").sort("event_id")
)
    particles_hard_scatter_ids=(particles.lazy()
    .select('event_id', 'particle_id')
    ).collect()

    # ----------------------------------------------

    print(f"[CASTING DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    print("[ORPHAN MASK] Adding orphan mask...")
    particles = add_orphan_mask(particles)
    print(f"[ORPHAN MASK DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    print("[CALO MASK] Adding created inside calo mask...")
    particles = add_created_inside_calo_mask(particles)
    print(f"[CALO MASK DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    print("[TRACK MASK] Adding particle have track mask...")
    particles = add_particle_have_track_mask(particles, tracks)
    print(f"[TRACK MASK DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    

    # ---------------------
    print("[ETA PHI PT] Adding eta, phi, pt...")
    particles = add_eta_and_phi_and_pt(particles)
    print(f"[ETA PHI PT DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    print("[PARENT MASK] Getting particles id parent of inside calo particles...")
    particles = get_particles_id_parent_of_inside_calo_particles_maskv3(particles, calo_hits)
    print(f"[PARENT MASK DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    print("[TARGET PARTICLES MASK] Setting target particles mask...")
    particles = set_target_particles_maskv4(particles, truth_eta_cut=truth_eta_cut, truth_pt_cut=truth_pt_cut, target_pt_cut=target_pt_cut, tracks=tracks)
    print(f"[TARGET PARTICLES MASK DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")


    print(f"[PROCESSING TRACKS DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    print("[CLUE CLUSTERING] Running CLUE clustering...")
    #calo_hits = clue_clustering(calo_hits, dc=75.88106168184893, rhoc=104.34315216716726, dm=87.0967630118376, ppbin=16)
    calo_hits = clue_clustering(calo_hits, dc=75.88106168184893, rhoc=104.34315216716726, dm=87.0967630118376, ppbin=16)
    gc.collect()
    # apply cutoff on calo hits, grouby by event_id and cluster_id to aggregate cell ids, if sum < 0.1 Gev drop the cells
    calo_hits = (
        calo_hits.lazy()
        .with_row_index('_event_idx_temp')
        .explode(pl.all().exclude(['event_id', '_event_idx_temp']))
        .join(CALIBRATION.lazy().select(['detector', 'calib_factor']), on='detector', how='left')
        .with_columns(
            (pl.col('total_energy') * pl.col('calib_factor')).alias('hit_energy_gev')
        )
        .with_columns(
            pl.col('hit_energy_gev').sum().over(['event_id', 'cluster_id']).alias('cluster_sum_energy')
        )
        .filter(pl.col('cluster_sum_energy') > clusters_cutoff)
        .filter(pl.col('cluster_id') >= 0) # Remove noise hits that were not clustered
        .drop(['calib_factor', 'hit_energy_gev', 'cluster_sum_energy'])
        .group_by(['_event_idx_temp', 'event_id'], maintain_order=True)
        .agg(pl.all().exclude(['_event_idx_temp', 'event_id']))
        .drop('_event_idx_temp')
        .collect(streaming=True)
    )
    print(f"[CLUE CLUSTERING DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    # Target particle caloremeter calo clusters deposits ---------
    print("[DEPOSITORS LIST] Creating depositors list...")
    depositors_list = (
        calo_hits.lazy()
        .select(['event_id', 'contrib_particle_ids'])
        .explode('contrib_particle_ids')
        .explode('contrib_particle_ids') # Double explode if list[list]
        .rename({'contrib_particle_ids': 'particle_id'})
        .unique(subset=['event_id', 'particle_id'])

        .join(
            particles_hard_scatter_ids.lazy().select(['event_id', 'particle_id']).explode('particle_id'),
            on=['event_id', 'particle_id'],
            how='inner'
        )
        .select([
            pl.col('event_id'),
            pl.col('particle_id').cast(pl.Int64)
        ])
    ).collect(streaming=True)
    print(f"[DEPOSITORS LIST DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    print("[TARGET PARTICLES AGG] Aggregating target particles...")
    target_particles = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_target_particle', 'pdg_id',
              'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt', 'has_track', 'vertex_primary'])
        .explode( 'particle_id', 'is_target_particle', 'pdg_id',
              'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt',
              'has_track', 'vertex_primary')
        .filter(pl.col('is_target_particle'))
        .sort('event_id')
        .with_row_index("global_order")
        .sort('global_order')
        .drop('is_target_particle', 'global_order')
        .group_by('event_id', maintain_order=True)
        .agg('*')
        .collect(streaming=True)
    )
    print(f"[TARGET PARTICLES AGG DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # OPTIMIZATION: Save particles for backtrack, then free large structure
    particles_for_backtrack = particles.lazy().select(           pl.col("event_id"),
            pl.col("particle_id"),
            pl.col("parent_id"),
            pl.col("is_parent_missing")).collect()
    del particles
    gc.collect()
    print(f"[PARTICLES FREED] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    print("[CREATE CALO CLUSTERS] Creating calo clusters...")
    calo_clusters = create_calo_clusters(calo_hits)
    print(f"[CREATE CALO CLUSTERS DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    print("[CLUSTER IDX MAPPING] Creating cluster to cluster index mapping...")
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
    print(f"[CLUSTER IDX MAPPING DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    

    print("[BACKTRACK TO TARGET] Backtracking particles to target...")
    points_to_target = backtrack_to_target(particles=particles_for_backtrack,
                       src_df=depositors_list,
                       target_df=target_particles.select(['event_id', 'particle_id']).explode('particle_id'))
    print(f"[BACKTRACK TO TARGET DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # OPTIMIZATION: Free intermediate structures
    del particles_for_backtrack, depositors_list
    gc.collect()
    print(f"[INTERMEDIATES FREED] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    print("[CLUSTER PURITY] Computing cluster purity...")
    target_particles_deps = cluster_purity(calo_hits_with_clusters=calo_hits, ancestors=points_to_target)
    print(f"[CLUSTER PURITY DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # OPTIMIZATION: Free calo_hits after cluster purity
    #del calo_hits
    gc.collect()
    print(f"[CALO HITS FREED] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")



    # Filter Orphans and Reindex -------------------
    print("[FILTER ORPHANS] Filtering orphan particles and reindexing...")
    filtered_data = filter_orphans_and_reindex(
        target_particles=target_particles,
        target_particles_deps=target_particles_deps,
        tracks=tracks,
        cluster_to_cluster_idx=cluster_to_cluster_idx
    )
    print(f"[FILTER ORPHANS DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print("[PREPROCESS COMPLETE]\n")

    return {
        "target_particles": filtered_data["target_particles"],
        "calo_clusters": calo_clusters,
        "tracks": filtered_data["tracks"],
        "target_particles_deps": filtered_data["target_particles_deps"], 
        "calo_hits": calo_hits,
    }





def add_vertex_info_to_target_particles(
    target_particles: pl.DataFrame,
    particles_raw: pl.DataFrame,
) -> pl.DataFrame:
    """
    Adds vx, vy, vz columns to an existing target_particles DataFrame by joining
    with the raw particles DataFrame (from HuggingFace) on (event_id, particle_id).

    Both DataFrames may have list-type or scalar particle_id columns; this function
    handles either format automatically.

    Args:
        target_particles: Grouped DataFrame (one row per event_id, list-type columns).
                          Must contain 'event_id', 'particle_id', 'particle_idx'.
        particles_raw:    Raw particles DataFrame with 'event_id', 'particle_id',
                          'vx', 'vy', 'vz' columns (grouped or flat).

    Returns:
        target_particles with additional columns: vx (List[Float32]),
        vy (List[Float32]), vz (List[Float32]).
    """
    # Flatten raw particles if they are grouped (list-type particle_id)
    vxyz_cols = ['event_id', 'particle_id', 'vx', 'vy', 'vz']
    raw_lf = particles_raw.lazy().select(vxyz_cols)
    if isinstance(particles_raw.schema['particle_id'], pl.List):
        raw_lf = raw_lf.explode(['particle_id', 'vx', 'vy', 'vz'])

    # Cast particle_id to match target_particles type (UInt64)
    raw_lf = raw_lf.with_columns(pl.col('particle_id').cast(pl.UInt64))

    # Explode target_particles to flat form, preserving sort order via particle_idx
    tp_flat = (
        target_particles.lazy()
        .select(['event_id', 'particle_id', 'particle_idx'])
        .explode(['particle_id', 'particle_idx'])
    )

    # Join, sort by particle_idx within each event, then re-aggregate as lists
    vxyz_grouped = (
        tp_flat
        .join(raw_lf, on=['event_id', 'particle_id'], how='left')
        .sort(['event_id', 'particle_idx'])
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('vx').cast(pl.Float32),
            pl.col('vy').cast(pl.Float32),
            pl.col('vz').cast(pl.Float32),
        ])
        .collect()
    )

    return target_particles.join(vxyz_grouped, on='event_id', how='left')


def add_vertex_info_to_tracks(
    tracks: pl.DataFrame,
    tracks_raw: pl.DataFrame,
    particles_raw: pl.DataFrame,
) -> pl.DataFrame:
    """
    Adds majority_particle_id, majority_particle_vx/vy/vz to processed tracks
    by chaining two joins:
      processed tracks (track_id) → raw HF tracks  (track_id → majority_particle_id)
                                  → raw HF particles (majority_particle_id → vx, vy, vz)

    Orphan tracks (no matching majority_particle_id) will have null vertex values.

    Args:
        tracks:       Processed tracks DataFrame (one row per event_id, list columns).
                      Must contain 'event_id', 'track_id'.
        tracks_raw:   Raw HF tracks DataFrame with 'event_id', 'track_id',
                      'majority_particle_id' (grouped or flat).
        particles_raw: Raw HF particles DataFrame with 'event_id', 'particle_id',
                       'vx', 'vy', 'vz' (grouped or flat).

    Returns:
        tracks with additional columns: majority_particle_id (List[Int64]),
        majority_particle_vx/vy/vz (List[Float32]),
        majority_particle_vertex_primary (List[UInt16]).
    """
    # --- Step 1: flatten raw tracks to get track_id -> majority_particle_id ---
    tr_lf = tracks_raw.lazy().select(['event_id', 'track_id', 'majority_particle_id'])
    if isinstance(tracks_raw.schema['track_id'], pl.List):
        tr_lf = tr_lf.explode(['track_id', 'majority_particle_id'])
    tr_lf = tr_lf.with_columns(
        pl.col('track_id').cast(pl.UInt16),
        pl.col('majority_particle_id').cast(pl.Int64),
    )

    # --- Step 2: flatten raw particles to get particle_id -> vx, vy, vz, vertex_primary ---
    p_lf = particles_raw.lazy().select(['event_id', 'particle_id', 'vx', 'vy', 'vz', 'vertex_primary'])
    if isinstance(particles_raw.schema['particle_id'], pl.List):
        p_lf = p_lf.explode(['particle_id', 'vx', 'vy', 'vz', 'vertex_primary'])
    p_lf = p_lf.with_columns(pl.col('particle_id').cast(pl.Int64))

    # track_id -> majority_particle_id -> vx, vy, vz, vertex_primary
    track_id_to_info = (
        tr_lf
        .join(p_lf, left_on=['event_id', 'majority_particle_id'], right_on=['event_id', 'particle_id'], how='left')
        .select(['event_id', 'track_id', 'majority_particle_id', 'vx', 'vy', 'vz', 'vertex_primary'])
    )

    # --- Step 3: explode processed tracks by track_id, join, re-aggregate ---
    result_grouped = (
        tracks.lazy()
        .select(['event_id', 'track_id'])
        .with_columns(
            local_order=pl.int_ranges(0, pl.col('track_id').list.len(), dtype=pl.UInt32)
        )
        .explode(['track_id', 'local_order'])
        .with_columns(pl.col('track_id').cast(pl.UInt16))
        .join(track_id_to_info, on=['event_id', 'track_id'], how='left')
        .sort(['event_id', 'local_order'])
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('majority_particle_id'),
            pl.col('vx').cast(pl.Float32).alias('majority_particle_vx'),
            pl.col('vy').cast(pl.Float32).alias('majority_particle_vy'),
            pl.col('vz').cast(pl.Float32).alias('majority_particle_vz'),
            pl.col('vertex_primary').alias('majority_particle_vertex_primary'),
        ])
        .collect()
    )

    return tracks.join(result_grouped, on='event_id', how='left')


def update_tracks_with_vertex_info(
    data_dir: str,
    event_name: str = "ttbar_pu200",
    number_of_hf_repo_files: int = 1000,
    overwrite: bool = False,
    file_indices=None,
) -> None:
    """
    For each tracks-*.parquet in data_dir, downloads the corresponding raw tracks
    and particles from HuggingFace, then adds majority_particle_id and
    majority_particle_vx/vy/vz. Files are overwritten in place.

    Args:
        data_dir:                 Directory containing tracks-*.parquet files.
        event_name:               HuggingFace dataset event name (e.g. 'ttbar_pu200').
        number_of_hf_repo_files:  Total number of parquet shards in the HF repo.
        overwrite:                If False (default), skip files already containing
                                  majority_particle_id.
        file_indices:             Optional iterable of integer file indices to process
                                  (e.g. range(0, 10) or [0, 5, 42]). If None, all
                                  tracks-*.parquet files in data_dir are processed.
    """
    from pathlib import Path
    from huggingface_hub import HfFileSystem
    import tqdm
    import gc

    fs = HfFileSystem()
    data_path = Path(data_dir)

    if file_indices is not None:
        allowed = {i for i in file_indices}
        track_files = sorted(
            f for f in data_path.glob("tracks-*.parquet")
            if int(f.stem.split('-')[-1]) in allowed
        )
    else:
        track_files = sorted(data_path.glob("tracks-*.parquet"))

    if not track_files:
        print(f"No tracks-*.parquet files found in {data_dir}")
        return

    print(f"Found {len(track_files)} tracks files to process.")

    for track_file in tqdm.tqdm(track_files, desc="Adding majority_particle_id/vx/vy/vz to tracks"):
        idx_str = track_file.stem.split('-')[-1]
        i = int(idx_str)

        tracks = pl.read_parquet(track_file)

        if not overwrite and 'majority_particle_id' in tracks.columns:
            continue

        hf_tracks_path = (
            f"datasets/CERN/ColliderML-Release-1/data/{event_name}_tracks/"
            f"train-{i:05d}-of-{number_of_hf_repo_files:05d}.parquet"
        )
        hf_particles_path = (
            f"datasets/CERN/ColliderML-Release-1/data/{event_name}_particles/"
            f"train-{i:05d}-of-{number_of_hf_repo_files:05d}.parquet"
        )

        if not fs.exists(hf_tracks_path):
            print(f"Warning: HF tracks not found: {hf_tracks_path}, skipping.")
            continue
        if not fs.exists(hf_particles_path):
            print(f"Warning: HF particles not found: {hf_particles_path}, skipping.")
            continue

        with fs.open(hf_tracks_path, "rb") as f:
            tracks_raw = pl.read_parquet(f, columns=['event_id', 'track_id', 'majority_particle_id'])

        with fs.open(hf_particles_path, "rb") as f:
            particles_raw = pl.read_parquet(f, columns=['event_id', 'particle_id', 'vx', 'vy', 'vz', 'vertex_primary'])

        updated = add_vertex_info_to_tracks(tracks, tracks_raw, particles_raw)
        updated.write_parquet(track_file)

        del tracks, tracks_raw, particles_raw, updated
        gc.collect()

    print("Done.")


def update_target_particles_with_vertex_info(
    data_dir: str,
    event_name: str = "ttbar_pu200",
    number_of_hf_repo_files: int = 1000,
    overwrite: bool = False,
) -> None:
    """
    Reads all target_particles-*.parquet files in data_dir, re-downloads the
    corresponding particles file from HuggingFace, adds vx/vy/vz columns, and
    overwrites the parquet files in place.

    Args:
        data_dir:                 Directory containing target_particles-*.parquet files.
        event_name:               HuggingFace dataset event name (e.g. 'ttbar_pu200').
        number_of_hf_repo_files:  Total number of parquet shards in the HF repo.
        overwrite:                If False (default), skip files that already have vx.
    """
    from pathlib import Path
    from huggingface_hub import HfFileSystem
    import tqdm
    import gc

    fs = HfFileSystem()
    data_path = Path(data_dir)

    tp_files = sorted(data_path.glob("target_particles-*.parquet"))
    if not tp_files:
        print(f"No target_particles-*.parquet files found in {data_dir}")
        return

    print(f"Found {len(tp_files)} target_particles files to process.")

    for tp_file in tqdm.tqdm(tp_files, desc="Adding vx/vy/vz to target_particles"):
        idx_str = tp_file.stem.split('-')[-1]
        i = int(idx_str)

        target_particles = pl.read_parquet(tp_file)

        if not overwrite and 'vx' in target_particles.columns:
            continue

        hf_path = (
            f"datasets/CERN/ColliderML-Release-1/data/{event_name}_particles/"
            f"train-{i:05d}-of-{number_of_hf_repo_files:05d}.parquet"
        )
        if not fs.exists(hf_path):
            print(f"Warning: HF file not found: {hf_path}, skipping.")
            continue

        with fs.open(hf_path, "rb") as f:
            particles_raw = pl.read_parquet(f, columns=['event_id', 'particle_id', 'vx', 'vy', 'vz'])

        updated = add_vertex_info_to_target_particles(target_particles, particles_raw)
        updated.write_parquet(tp_file)

        del target_particles, particles_raw, updated
        gc.collect()

    print("Done.")


def run_preprocessing_pipeline(r=None, event_name: str="ttbar_pu200", ):
    from huggingface_hub import HfFileSystem
    import polars as pl
    import tqdm
    import gc
    fs = HfFileSystem()
    if r is not None:
        number_of_files = r
    number_of_hf_repo_files = 1000
    for i in tqdm.tqdm(number_of_files):
        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_name}_particles/train-{i:05d}-of-{number_of_hf_repo_files:05d}.parquet"
        print(f"Processing file: {file_path}")
        if not fs.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_name}_particles/train-{i:05d}-of-{number_of_hf_repo_files:05d}.parquet"
        with fs.open(file_path, "rb") as f:
            particles = pl.read_parquet(f,columns=[    'event_id',
    'particle_id',
    'vertex_primary',
    'pdg_id',
    'energy',
    'px',
    'py',
    'pz',
    'vx',
    'vy',
    'vz',
    'parent_id',
])


        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_name}_calo_hits/train-{i:05d}-of-{number_of_hf_repo_files:05d}.parquet"
        with fs.open(file_path, "rb") as f:
            calo_hits = pl.read_parquet(f,columns=    ['event_id',
    'detector',
    'total_energy',
    'x',
    'y',
    'z',
    'contrib_particle_ids',
    'contrib_energies'
])

        file_path = f"datasets/CERN/ColliderML-Release-1/data/{event_name}_tracks/train-{i:05d}-of-{number_of_hf_repo_files:05d}.parquet"
        with fs.open(file_path, "rb") as f:
            tracks = pl.read_parquet(f)

        preprocessed_data = preprocess_for_model(particles=particles, tracks=tracks,
                                                  calo_hits=calo_hits, num_of_events=-1, 
                                                  truth_pt_cut=1, truth_eta_cut=3.0, target_pt_cut=0.3, clusters_cutoff=0.15)
        
        # write preprocessed data to local disk as parquets
        file_path_data = f"/storage/agrp/barakma/PileupODD/data/{event_name}"
        from pathlib import Path
        Path(file_path_data).mkdir(parents=True, exist_ok=True)
        for key, df in preprocessed_data.items():
            df.write_parquet(f"{file_path_data}/{key}-{i:05d}.parquet")
        
        # Free memory
        del particles, tracks, calo_hits, preprocessed_data
        gc.collect()

