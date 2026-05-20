from typing import Dict
import polars as pl
import numpy as np
import yaml # type: ignore
import gc

from sklearn.model_selection import train_test_split
from primary.preprocessing import (
    add_eta_and_phi_and_pt,
    add_orphan_mask,
    add_created_inside_calo_mask,
    add_particle_have_track_mask,
    set_target_particles_maskv4,
    get_particles_id_parent_of_inside_calo_particles_maskv3,
    backtrack_to_target,
    cluster_purity,
    cluster_contrib_energy,
    cluster_vertex_primary_deps,
    calculate_extrapolated_features_polars,
)
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
        "number_of_hits":  {"df": "calo_clusters", "col": "number_of_hits",  "transform": "sqrt", "type": "min_max_sym"},
        "energy_hits_std": {"df": "calo_clusters", "col": "energy_hits_std", "transform": "sqrt", "type": "std"},
        "max_hit_energy":  {"df": "calo_clusters", "col": "max_hit_energy",  "transform": "sqrt", "type": "min_max_sym"},
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


def generate_normalization_stats_sequential(data_dir: str, max_files: int = 40, kll_k: int = 200) -> str:
    """
    Generates normalization stats sequentially from parquet files in a directory.
    Memory-efficient alternative to generate_normalization_yaml.
    """
    from pathlib import Path
    from datasketches import kll_floats_sketch
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
        "number_of_hits":  {"df": "calo_clusters", "col": "number_of_hits",  "transform": None, "type": "min_max_sym"},
        "energy_hits_std": {"df": "calo_clusters", "col": "energy_hits_std", "transform": "sqrt", "type": "std"},
        "max_hit_energy":  {"df": "calo_clusters", "col": "max_hit_energy",  "transform": "sqrt", "type": "min_max_sym"},
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

    # One KLL sketch per feature for streaming quantile estimation.
    quantile_sketches = {
        key: kll_floats_sketch(kll_k)
        for key in config_schema
    }

    def _extract_values(df: pl.DataFrame, key: str, schema: Dict[str, str]) -> np.ndarray:
        """
        Extracts and transforms a single feature into a finite NumPy array.
        """
        col_name = schema["col"]
        transform = schema["transform"]

        series = None

        if key == "cluster_pt":
            if "total_cluster_energy" in df.columns and "cluster_eta" in df.columns:
                e = df.select(pl.col("total_cluster_energy").explode()).get_column("total_cluster_energy")
                eta = df.select(pl.col("cluster_eta").explode()).get_column("cluster_eta")
                series = e / eta.cosh()
        else:
            if col_name not in df.columns:
                return np.array([], dtype=np.float64)

            dtype = df.schema[col_name]
            if isinstance(dtype, pl.List):
                series = df.select(pl.col(col_name).explode()).get_column(col_name)
            else:
                series = df.get_column(col_name)

        if series is None or len(series) == 0:
            return np.array([], dtype=np.float64)

        if transform == "sqrt":
            series = series.sqrt()

        arr = series.to_numpy()
        arr = arr[np.isfinite(arr)]
        return arr

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

    if max_files > 0:
        indices = indices[:max_files]

    # Single pass: moments + min/max + KLL quantiles.
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
            
            if df_name not in loaded_dfs:
                continue
                
            df = loaded_dfs[df_name]
            
            arr = _extract_values(df, key, schema)
            if len(arr) == 0:
                continue

            # Update streaming quantile sketch.
            try:
                quantile_sketches[key].update(arr)
            except TypeError:
                for x in arr:
                    quantile_sketches[key].update(float(x))
                
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

        if s["max"] <= s["min"]:
            q25_val = float(s["min"])
            q75_val = float(s["max"])
            q95_val = float(s["max"])
            q99_val = float(s["max"])
        else:
            sketch = quantile_sketches.get(key)
            if sketch is None:
                q25_val = float(s["min"])
                q75_val = float(s["max"])
                q95_val = float(s["max"])
                q99_val = float(s["max"])
            else:
                # datasketches API differs slightly across builds.
                if hasattr(sketch, "is_empty"):
                    sketch_empty = bool(sketch.is_empty())
                elif hasattr(sketch, "n"):
                    sketch_empty = int(sketch.n) == 0
                else:
                    sketch_empty = False

                if sketch_empty:
                    q25_val = float(s["min"])
                    q75_val = float(s["max"])
                    q95_val = float(s["max"])
                    q99_val = float(s["max"])
                else:
                    q25_val = float(sketch.get_quantile(0.25))
                    q75_val = float(sketch.get_quantile(0.75))
                    q95_val = float(sketch.get_quantile(0.95))
                    q99_val = float(sketch.get_quantile(0.99))
        
        entry = {
            "type": schema["type"],
            "mean": float(f"{mean_val:.4f}"),
            "std": float(f"{std_val:.4f}"),
            "q25": smart_fmt(q25_val),
            "q75": smart_fmt(q75_val),
            "q95": smart_fmt(q95_val),
            "q99": smart_fmt(q99_val),
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
        ordered["q25"] = entry["q25"]
        ordered["q75"] = entry["q75"]
        ordered["q95"] = entry["q95"]
        ordered["q99"] = entry["q99"]
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
        .agg([
            pl.col('particle_idx').sort_by('local_order'),
            pl.col('particle_id').sort_by('local_order'),
        ])
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


def _preprocess_source(
    particles: pl.DataFrame,
    tracks: pl.DataFrame,
    calo_hits: pl.DataFrame,
    kind: str,
    num_of_events: int = -1,
    truth_eta_cut: float = 3.0,
    truth_pt_cut: float = 1.0,
    target_pt_cut: float = 0.3,
) -> Dict[str, pl.DataFrame]:
    """
    Per-source preprocessing (HS or PU), run independently before overlay.

    Shared (both):
      Float32 cast, extrapolated track features, track explode + pt/eta filter
      + vertex_primary join + group-back, particles eta/phi/pt, and
      track->particle info join for vx/vy/vz/particle_pt.

    HS-only:
      pre-filter particles_pid_to_vertex snapshot, hard-scatter filter, particle
      masks (orphan/calo/has_track), parent-of-inside-calo mask, target mask.

    PU particles are consumed only to add pt and to feed the track-particle
    info join, then dropped.
    """
    import psutil
    import os
    process = psutil.Process(os.getpid())
    tag = kind.upper()
    print(f"\n[{tag} PREPROCESS START] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # num_of_events is an HS-only debug limiter: shrinking the pileup pool too
    # would starve the Poisson sampler. Pileup keeps its full event pool.
    # Filter by the first N unique event_ids (files have global ids, not 0-based).
    if num_of_events >= 0 and kind == 'hs':
        first_n_ids = particles['event_id'].unique().sort()[:num_of_events]
        particles = particles.filter(pl.col('event_id').is_in(first_n_ids))
        tracks = tracks.filter(pl.col('event_id').is_in(first_n_ids))
        calo_hits = calo_hits.filter(pl.col('event_id').is_in(first_n_ids))

    # Float32 cast
    particles = particles.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32)),
    ])
    tracks = tracks.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32)),
    ])
    calo_hits = calo_hits.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.List(pl.Float64)).cast(pl.List(pl.Float32)),
    ])

    tracks = calculate_extrapolated_features_polars(tracks)
    print(f"[{tag} EXTRAPOLATED] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # Track filter + vertex_primary join + group-back
    track_cols = [c for c in tracks.columns if c != 'event_id']
    tracks = (
        tracks.lazy()
        .with_columns(
            local_order=pl.int_ranges(
                start=0,
                end=pl.col('majority_particle_id').list.len(),
                dtype=pl.UInt32,
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
            how='left',
        )
        .with_columns(pl.col('majority_particle_id').cast(pl.Int64))
        .sort(['event_id', 'local_order'])
        .group_by('event_id', maintain_order=True)
        .agg([pl.col(c) for c in track_cols] + [pl.col('vertex_primary')])
        .sort('event_id')
        .collect(streaming=True)
    )
    print(f"[{tag} TRACKS FILTERED] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    particles_pid_to_vertex = None
    particles_hard_scatter_ids = None

    if kind == 'hs':
        particles_pid_to_vertex = (
            particles.lazy()
            .select(['event_id', 'particle_id', 'vertex_primary'])
            .explode('particle_id', 'vertex_primary')
            .with_columns(
                pl.col('particle_id').cast(pl.Int64),
                pl.col('vertex_primary').cast(pl.UInt16),
            )
            .collect(streaming=True)
        )

        particles = (
            particles.lazy()
            .with_columns(
                _indices=pl.col('vertex_primary').list.eval(
                    (pl.element() == 1).arg_true()
                )
            )
            .with_columns(
                pl.exclude('event_id', '_indices').list.gather(pl.col('_indices'))
            )
            .drop('_indices')
            .sort('event_id')
        ).collect()

        particles_hard_scatter_ids = (
            particles.lazy().select('event_id', 'particle_id')
        ).collect()

        particles = add_orphan_mask(particles)
        particles = add_created_inside_calo_mask(particles)
        particles = add_particle_have_track_mask(particles, tracks)

    particles = add_eta_and_phi_and_pt(particles)
    print(f"[{tag} ETA PHI PT] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # Track <- particle info (vx, vy, vz, particle_pt)
    particle_info_lf = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'vx', 'vy', 'vz', 'pt'])
        .explode('particle_id', 'vx', 'vy', 'vz', 'pt')
        .rename({'pt': 'particle_pt'})
    )
    track_particle_cols = (
        tracks.lazy()
        .select(['event_id', 'majority_particle_id'])
        .with_columns(
            _local_order=pl.int_ranges(start=0, end=pl.col('majority_particle_id').list.len(), dtype=pl.UInt32)
        )
        .explode(['majority_particle_id', '_local_order'])
        .join(
            particle_info_lf,
            left_on=['event_id', 'majority_particle_id'],
            right_on=['event_id', 'particle_id'],
            how='left',
        )
        .sort(['event_id', '_local_order'])
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('vx').cast(pl.Float32),
            pl.col('vy').cast(pl.Float32),
            pl.col('vz').cast(pl.Float32),
            pl.col('particle_pt').cast(pl.Float32),
        ])
        .collect(streaming=True)
    )
    tracks = tracks.join(track_particle_cols, on='event_id', how='left')
    del track_particle_cols, particle_info_lf
    print(f"[{tag} TRACK<-PARTICLE INFO] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    out: Dict[str, pl.DataFrame] = {
        'tracks': tracks,
        'calo_hits': calo_hits,
    }
    if kind == 'hs':
        particles = get_particles_id_parent_of_inside_calo_particles_maskv3(particles, calo_hits)
        particles = set_target_particles_maskv4(
            particles,
            truth_eta_cut=truth_eta_cut,
            truth_pt_cut=truth_pt_cut,
            target_pt_cut=target_pt_cut,
            tracks=tracks,
        )
        out['particles'] = particles
        out['particles_pid_to_vertex'] = particles_pid_to_vertex
        out['particles_hard_scatter_ids'] = particles_hard_scatter_ids
        print(f"[{tag} MASKS DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    else:
        # Pileup particles are not part of the dataset; release them.
        # Keep a tiny event_ids snapshot first: particles is the canonical
        # source of "which PU vertices exist" — used by the sampler so that
        # vertices with no calo hits (invisible vertices) still get sampled
        # at their Poisson rate and contribute zero cells.
        out['particle_event_ids'] = (
            particles.lazy().select('event_id').unique(maintain_order=True).collect()
        )
        del particles
        gc.collect()
    return out


def _build_sample_map(hs_event_ids: np.ndarray, pu_event_ids: np.ndarray,
                       pileup_level: int, seed: int,
                       invisible_pu_prob: float = 0.0) -> pl.DataFrame:
    """
    Per HS event: N ~ Poisson(pileup_level), then choose N distinct pileup
    event_ids (no repeat within an HS event). Replacement allowed across HS
    events. Returns a DataFrame with columns hs_event_id (u32) and
    pu_event_id (list[u32]).

    If invisible_pu_prob > 0, each of the N draws is independently "invisible"
    (contributes nothing — simulates diffractive events missing the detector)
    with that probability. Equivalent to drawing K ~ Binomial(N, 1-p) and
    sampling K events from the pool — done that way for efficiency (no wasted
    sampling on rolls that would be discarded).
    """
    if not 0.0 <= invisible_pu_prob < 1.0:
        raise ValueError(f"invisible_pu_prob must be in [0, 1), got {invisible_pu_prob}")
    rng = np.random.default_rng(seed=seed)
    pool = pu_event_ids
    pool_size = len(pool)
    ns = rng.poisson(pileup_level, size=len(hs_event_ids))
    if invisible_pu_prob > 0.0:
        ns = rng.binomial(ns, 1.0 - invisible_pu_prob)
    ns = np.minimum(ns, pool_size).astype(np.int64)
    pu_per_hs = [rng.choice(pool, size=int(n), replace=False).astype(pool.dtype) for n in ns]
    return pl.DataFrame({
        'hs_event_id': hs_event_ids,
        'pu_event_id': pu_per_hs,
    })


def _overlay_calo_hits(
    hs_calo_hits: pl.DataFrame,
    pu_calo_hits: pl.DataFrame,
    sample_map_flat: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge HS + sampled pileup calo hits cell-by-cell on (event_id, detector,
    x, y, z) via full outer join. Pileup contribs are NOT carried: we only
    add pileup energy. HS contrib lists pass through untouched; pileup-only
    cells get empty contrib lists. Returns list-per-event frame suitable
    for clue_clustering.
    """
    # Pileup hits: select-explode-round, join sample_map (replicates per HS event), sum per cell.
    pu_cells = (
        pu_calo_hits.lazy()
        .select(['event_id', 'detector', 'total_energy', 'x', 'y', 'z'])
        .explode(['detector', 'total_energy', 'x', 'y', 'z'])
        # Drop the phantom null row that polars produces when an empty calo_hits
        # list is exploded. The PU event itself is still in the sampler pool
        # (enumerated from unique event_id of pu_calo_hits) so empty-calo
        # vertices still get sampled and just contribute zero cells, which
        # correctly imitates an "invisible" pileup vertex.
        .filter(pl.col('detector').is_not_null())
        .with_columns([
            pl.col('x').round(3),
            pl.col('y').round(3),
            pl.col('z').round(3),
        ])
    )
    pu_cell_energy = (
        sample_map_flat.lazy()
        .join(pu_cells, left_on='pu_event_id', right_on='event_id')
        .group_by([pl.col('hs_event_id').alias('event_id'), 'detector', 'x', 'y', 'z'])
        .agg(pl.col('total_energy').sum().alias('pu_energy'))
    )

    # HS hits: select-explode-round.
    hs_flat = (
        hs_calo_hits.lazy()
        .select(['event_id', 'detector', 'total_energy', 'x', 'y', 'z',
                 'contrib_particle_ids', 'contrib_energies'])
        .explode(['detector', 'total_energy', 'x', 'y', 'z',
                  'contrib_particle_ids', 'contrib_energies'])
        .with_columns([
            pl.col('x').round(3),
            pl.col('y').round(3),
            pl.col('z').round(3),
        ])
    )

    merged_flat = (
        hs_flat
        .join(pu_cell_energy,
              on=['event_id', 'detector', 'x', 'y', 'z'],
              how='full', coalesce=True)
        .with_columns([
            (pl.col('total_energy').fill_null(0.0) + pl.col('pu_energy').fill_null(0.0))
                .alias('total_energy'),
            pl.col('contrib_particle_ids').fill_null(pl.lit([], dtype=pl.List(pl.UInt64))),
            pl.col('contrib_energies').fill_null(pl.lit([], dtype=pl.List(pl.Float32))),
        ])
        .drop('pu_energy')
    )

    merged_calo_hits = (
        merged_flat
        .group_by('event_id', maintain_order=True)
        .agg(pl.all())
        .sort('event_id')
        .collect(streaming=True)
    )
    return merged_calo_hits


def _overlay_tracks(
    hs_tracks: pl.DataFrame,
    pu_tracks: pl.DataFrame,
    sample_map_flat: pl.DataFrame,
) -> pl.DataFrame:
    """
    Concatenate HS tracks and sampled pileup tracks per HS event, with a new
    `source_pileup_event_id` list column (null on HS rows, original pileup
    event_id on pileup rows). HS tracks come first in each per-event list,
    then pileup tracks.
    """
    hs_track_cols = [c for c in hs_tracks.columns if c != 'event_id']
    hs_flat = (
        hs_tracks.lazy()
        .explode(hs_track_cols)
        .with_columns(pl.lit(None, dtype=pl.UInt32).alias('source_pileup_event_id'))
    )

    pu_track_cols = [c for c in pu_tracks.columns if c != 'event_id']
    pu_flat = (
        pu_tracks.lazy()
        .explode(pu_track_cols)
        .rename({'event_id': 'pu_event_id'})
    )
    pu_overlaid = (
        sample_map_flat.lazy()
        .join(pu_flat, on='pu_event_id', how='inner')
        .rename({'hs_event_id': 'event_id',
                 'pu_event_id': 'source_pileup_event_id'})
    )

    final_cols = ['event_id'] + hs_track_cols + ['source_pileup_event_id']
    tracks = (
        pl.concat(
            [hs_flat.select(final_cols), pu_overlaid.select(final_cols)],
            how='vertical_relaxed',
        )
        .group_by('event_id', maintain_order=True)
        .agg(pl.all())
        .sort('event_id')
        .collect(streaming=True)
    )
    return tracks


def _run_overlay_and_aggregate(
    hs: Dict[str, pl.DataFrame],
    pu: Dict[str, pl.DataFrame],
    pileup_level: int,
    seed: int,
    clusters_cutoff: float,
    clue_backend: str,
    process,
    invisible_pu_prob: float = 0.0,
) -> Dict[str, pl.DataFrame]:
    """
    Overlay HS+PU calo hits, cluster, and run the full target/cluster
    aggregation. Consumes (and progressively `del`s) entries inside `hs`;
    leaves `pu` untouched (caller owns it — useful for chunked reuse).
    """
    # 1. Poisson sample map.
    hs_event_ids = hs['calo_hits']['event_id'].to_numpy()
    # Enumerate the PU pool from PARTICLES (canonical "vertex exists" set), so
    # vertices with no calo hits / no tracks still get sampled at their
    # Poisson rate — they correctly contribute zero cells / zero tracks
    # (imitates the real PU200 distribution where some vertices are invisible).
    # Fallback to calo_hits.event_id for callers that don't set particle_event_ids.
    if 'particle_event_ids' in pu:
        pu_event_ids = pu['particle_event_ids']['event_id'].to_numpy()
    else:
        pu_event_ids = pu['calo_hits']['event_id'].unique(maintain_order=True).to_numpy()
    sample_map = _build_sample_map(hs_event_ids, pu_event_ids, pileup_level, seed,
                                   invisible_pu_prob=invisible_pu_prob)
    sample_map_flat = sample_map.explode('pu_event_id')
    del sample_map
    print(f"[SAMPLE MAP] {len(sample_map_flat)} HS-PU pairs across {len(hs_event_ids)} HS events. "
          f"RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 2. Overlay calo hits.
    print("[OVERLAY CALO HITS] Merging HS and pileup hits per cell...")
    merged_calo_hits = _overlay_calo_hits(hs['calo_hits'], pu['calo_hits'], sample_map_flat)
    del hs['calo_hits']
    gc.collect()
    print(f"[OVERLAY CALO HITS DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 3. Overlay tracks.
    print("[OVERLAY TRACKS] Merging HS and pileup tracks...")
    tracks = _overlay_tracks(hs['tracks'], pu['tracks'], sample_map_flat)
    del hs['tracks'], sample_map_flat
    gc.collect()
    print(f"[OVERLAY TRACKS DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 4. CLUE clustering.
    print("[CLUE CLUSTERING] Running CLUE clustering on overlaid hits...")
    calo_hits = clue_clustering(merged_calo_hits, dc=75.88106168184893,
                                rhoc=104.34315216716726, dm=87.0967630118376, ppbin=16,
                                backend=clue_backend)
    del merged_calo_hits
    gc.collect()
    print(f"[CLUE CLUSTERING DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    # 5. Cluster energy cutoff filter.
    # Compute kept hit positions on a NARROW side-frame (3 hit cols + position),
    # then apply with `list.gather` to ALL wide list columns in one pass — no
    # wide explode of contrib_particle_ids / contrib_energies and no regroup.
    import time as _time
    _t0 = _time.time()
    keep_idx = (
        calo_hits.lazy()
        .select(['cluster_id', 'total_energy', 'detector'])
        .with_row_index('_rid')
        .with_columns(
            _pos=pl.int_ranges(0, pl.col('cluster_id').list.len(), dtype=pl.UInt32)
        )
        .explode(['cluster_id', 'total_energy', 'detector', '_pos'])
        .join(CALIBRATION.lazy().select(['detector', 'calib_factor']),
              on='detector', how='left')
        .with_columns(_cal_e=pl.col('total_energy') * pl.col('calib_factor'))
        .with_columns(_clu_sum=pl.col('_cal_e').sum().over(['_rid', 'cluster_id']))
        .filter((pl.col('_clu_sum') > clusters_cutoff) & (pl.col('cluster_id') >= 0))
        .group_by('_rid', maintain_order=True)
        .agg(_indices=pl.col('_pos').sort())
        .select(['_rid', '_indices'])
    )

    calo_hits = (
        calo_hits.lazy()
        .with_row_index('_rid')
        .join(keep_idx, on='_rid', how='left')
        .with_columns(
            pl.col('_indices').fill_null(pl.lit([], dtype=pl.List(pl.UInt32)))
        )
        .with_columns(
            pl.exclude('event_id', '_rid', '_indices').list.gather(pl.col('_indices'))
        )
        .drop(['_rid', '_indices'])
        .collect(streaming=True)
    )
    _dt = _time.time() - _t0
    print(f"[CLUE CLUSTERING DONE hits cut off] {_dt:.2f}s. "
          f"RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 6. depositors_list (HS particles only).
    # .pop() so the dict no longer holds these refs — `del` below truly frees them.
    particles = hs.pop('particles')
    particles_pid_to_vertex = hs.pop('particles_pid_to_vertex')
    particles_hard_scatter_ids = hs.pop('particles_hard_scatter_ids')

    print("[DEPOSITORS LIST] Creating depositors list...")
    depositors_list = (
        calo_hits.lazy()
        .select(['event_id', 'contrib_particle_ids'])
        .explode('contrib_particle_ids')
        .explode('contrib_particle_ids')
        .rename({'contrib_particle_ids': 'particle_id'})
        .unique(subset=['event_id', 'particle_id'])
        .join(
            particles_hard_scatter_ids.lazy().select(['event_id', 'particle_id']).explode('particle_id'),
            on=['event_id', 'particle_id'],
            how='inner',
        )
        .select([
            pl.col('event_id'),
            pl.col('particle_id').cast(pl.Int64),
        ])
    ).collect(streaming=True)
    del particles_hard_scatter_ids
    gc.collect()
    print(f"[DEPOSITORS LIST DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 7. Target particles aggregation.
    target_particles = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_target_particle', 'pdg_id',
                 'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt', 'has_track', 'vertex_primary',
                 'vx', 'vy', 'vz'])
        .explode('particle_id', 'is_target_particle', 'pdg_id',
                 'energy', 'eta', 'phi', 'px', 'py', 'pz', 'pt',
                 'has_track', 'vertex_primary', 'vx', 'vy', 'vz')
        .filter(pl.col('is_target_particle'))
        .sort('event_id')
        .with_row_index('global_order')
        .sort('global_order')
        .drop('is_target_particle', 'global_order')
        .group_by('event_id', maintain_order=True)
        .agg('*')
        .collect(streaming=True)
    )
    print(f"[TARGET PARTICLES AGG DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    particles_for_backtrack = (
        particles.lazy()
        .select(pl.col('event_id'), pl.col('particle_id'), pl.col('parent_id'), pl.col('is_parent_missing'))
        .collect()
    )
    del particles
    gc.collect()

    print("[CREATE CALO CLUSTERS] Creating calo clusters...")
    calo_clusters = create_calo_clusters(calo_hits)
    print(f"[CREATE CALO CLUSTERS DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    cluster_to_cluster_idx = (
        calo_clusters.lazy()
        .select(['event_id', 'cluster_id'])
        .explode('cluster_id')
        .with_row_index('cluster_idx')
        .group_by('event_id', maintain_order=True)
        .agg([
            pl.col('cluster_id'),
            (pl.col('cluster_idx') - pl.col('cluster_idx').min()).alias('cluster_idx'),
        ])
        .explode(['cluster_id', 'cluster_idx'])
        .collect()
    )

    points_to_target = backtrack_to_target(
        particles=particles_for_backtrack,
        src_df=depositors_list,
        target_df=target_particles.select(['event_id', 'particle_id']).explode('particle_id'),
    )
    del particles_for_backtrack, depositors_list
    gc.collect()

    contrib_energy = cluster_contrib_energy(calo_hits_with_clusters=calo_hits)
    del calo_hits
    gc.collect()

    cluster_vertex_deps = cluster_vertex_primary_deps(
        contrib_energy=contrib_energy,
        pid_to_vertex=particles_pid_to_vertex,
        cluster_to_cluster_idx=cluster_to_cluster_idx,
    )
    del particles_pid_to_vertex
    gc.collect()

    calo_clusters = calo_clusters.join(cluster_vertex_deps, on='event_id', how='left')
    del cluster_vertex_deps
    gc.collect()

    target_particles_deps = cluster_purity(
        contrib_energy=contrib_energy,
        ancestors=points_to_target,
    )
    del contrib_energy
    gc.collect()
    print(f"[CLUSTER PURITY DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    filtered_data = filter_orphans_and_reindex(
        target_particles=target_particles,
        target_particles_deps=target_particles_deps,
        tracks=tracks,
        cluster_to_cluster_idx=cluster_to_cluster_idx,
    )
    print(f"[FILTER ORPHANS DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    return {
        'target_particles': filtered_data['target_particles'],
        'calo_clusters': calo_clusters,
        'tracks': filtered_data['tracks'],
        'target_particles_deps': filtered_data['target_particles_deps'],
    }


def preprocess_for_model(
    hs_particles: pl.DataFrame,
    hs_tracks: pl.DataFrame,
    hs_calo_hits: pl.DataFrame,
    pu_particles: pl.DataFrame,
    pu_tracks: pl.DataFrame,
    pu_calo_hits: pl.DataFrame,
    pileup_level: int = 200,
    seed: int = 42,
    num_of_events: int = -1,
    truth_eta_cut: float = 3.0,
    truth_pt_cut: float = 1.0,
    target_pt_cut: float = 0.3,
    clusters_cutoff: float = 0.1,
    clue_backend: str = 'gpu cuda',
    chunk_size: int = -1,
    chunk_tmp_dir: str = "/storage/agrp/barakma/PileupODD/data/tmp",
    invisible_pu_prob: float = 0.0,
) -> Dict[str, pl.DataFrame]:
    """
    Build a synthetic PU<pileup_level> dataset by overlaying ~Poisson(pileup_level)
    pileup events on each PU0 hard-scatter event, then run clustering and the
    full target/cluster aggregation pipeline.

    clue_backend: backend passed to CLUEstering. Use 'gpu cuda' (default) for
    NVIDIA GPUs, or 'cpu serial' / 'cpu tbb' for CPU-only nodes. Example:
        preprocess_for_model(..., clue_backend='cpu serial')

    chunk_size: if > 0, process HS events in chunks of this size through the
    overlay->cluster->aggregate pipeline, then concatenate. The PU pool is
    NOT chunked (it's the shared sampling pool). Reduces peak RAM but slightly
    increases wall time. <=0 means no chunking (process all HS events at once).

    chunk_tmp_dir: parent directory under which a per-run temp dir is created
    for spilling chunk outputs to disk. Defaults to a path under PileupODD/data
    (large shared storage). Only used when chunk_size > 0.
    """
    import psutil
    import os
    process = psutil.Process(os.getpid())

    print("\n[PREPROCESS START]")
    print(f"RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # 1. Per-source preprocessing.
    hs = _preprocess_source(hs_particles, hs_tracks, hs_calo_hits, kind='hs',
                             num_of_events=num_of_events,
                             truth_eta_cut=truth_eta_cut,
                             truth_pt_cut=truth_pt_cut,
                             target_pt_cut=target_pt_cut)
    del hs_particles, hs_tracks
    gc.collect()

    pu = _preprocess_source(pu_particles, pu_tracks, pu_calo_hits, kind='pu',
                             num_of_events=num_of_events,
                             truth_eta_cut=truth_eta_cut,
                             truth_pt_cut=truth_pt_cut,
                             target_pt_cut=target_pt_cut)
    del pu_particles, pu_tracks
    gc.collect()
    print(f"[PER-SOURCE DONE] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    hs_event_ids_all = hs['calo_hits']['event_id'].to_numpy()
    n_hs = len(hs_event_ids_all)

    if chunk_size is None or chunk_size <= 0 or n_hs <= chunk_size:
        # Single pass (no chunking).
        result = _run_overlay_and_aggregate(hs, pu, pileup_level, seed,
                                            clusters_cutoff, clue_backend, process,
                                            invisible_pu_prob=invisible_pu_prob)
        del hs, pu
        gc.collect()
        print("[PREPROCESS COMPLETE]\n")
        return result

    # Chunked pass: slice HS by event_id, share PU pool across chunks.
    # Each chunk's 4 output frames are spilled to disk immediately so we don't
    # accumulate them in memory. Sources (hs, pu) are freed before reading back.
    import tempfile
    from pathlib import Path

    n_chunks = (n_hs + chunk_size - 1) // chunk_size
    print(f"[CHUNKING] Splitting {n_hs} HS events into {n_chunks} chunks of <= {chunk_size}")

    keys = ['target_particles', 'calo_clusters', 'tracks', 'target_particles_deps']

    Path(chunk_tmp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='clue_chunks_', dir=chunk_tmp_dir) as tmpdir:
        tmp = Path(tmpdir)
        for ci in range(n_chunks):
            chunk_ids_np = hs_event_ids_all[ci * chunk_size:(ci + 1) * chunk_size]
            chunk_ids = pl.Series('event_id', chunk_ids_np)
            print(f"\n[CHUNK {ci+1}/{n_chunks}] {len(chunk_ids_np)} HS events. "
                  f"RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

            hs_chunk = {
                'particles':                 hs['particles'].filter(pl.col('event_id').is_in(chunk_ids)),
                'tracks':                    hs['tracks'].filter(pl.col('event_id').is_in(chunk_ids)),
                'calo_hits':                 hs['calo_hits'].filter(pl.col('event_id').is_in(chunk_ids)),
                'particles_pid_to_vertex':   hs['particles_pid_to_vertex'].filter(pl.col('event_id').is_in(chunk_ids)),
                'particles_hard_scatter_ids':hs['particles_hard_scatter_ids'].filter(pl.col('event_id').is_in(chunk_ids)),
            }

            out = _run_overlay_and_aggregate(
                hs_chunk, pu,
                pileup_level=pileup_level,
                seed=seed + ci,  # distinct sampling per chunk
                clusters_cutoff=clusters_cutoff,
                clue_backend=clue_backend,
                process=process,
                invisible_pu_prob=invisible_pu_prob,
            )
            del hs_chunk
            gc.collect()

            # Spill this chunk's outputs to disk and free them from memory.
            for k in keys:
                out[k].write_parquet(tmp / f'chunk_{ci:04d}_{k}.parquet')
            del out
            gc.collect()
            print(f"[CHUNK {ci+1}/{n_chunks} SPILLED] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # All chunks processed: free per-source data BEFORE reading chunks back.
        del hs, pu
        gc.collect()
        print(f"\n[ALL CHUNKS DONE / SOURCES FREED] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # Read chunks per-key, concat, then delete files to release disk early.
        final: Dict[str, pl.DataFrame] = {}
        for k in keys:
            parts = sorted(tmp.glob(f'chunk_*_{k}.parquet'))
            final[k] = pl.concat([pl.read_parquet(p) for p in parts])
            for p in parts:
                p.unlink()
            gc.collect()
            print(f"[CONCAT {k}] {len(parts)} chunks merged. "
                  f"RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    print(f"[CHUNKS MERGED] RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print("[PREPROCESS COMPLETE]\n")
    return final





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
    vxyz_cols = ['event_id', 'particle_id', 'vx', 'vy', 'vz', 'pt']
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
            pl.col('pt').cast(pl.Float32),
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


def run_preprocessing_pipeline(
    r=None,
    event_name: str = "ttbar_pu0",
    pileup_level: int = 200,
    pu_event_name: str = "pileup_only_pu0",
    seed: int = 42,
    num_of_events: int = -1,
    clusters_cutoff: float = 0.15,
    pu_files_per_batch: int = 3,
    pu_indices=None,
    clue_backend: str = 'gpu cuda',
    chunk_size: int = -1,
    chunk_tmp_dir: str = "/storage/agrp/barakma/PileupODD/data/tmp",
    invisible_pu_prob: float = 0.0,
):
    """
    Synthetic PU<pileup_level>: load PU0 HS and pileup-only triplets from
    HuggingFace, overlay Poisson(pileup_level) pileup events on each HS event,
    cluster, aggregate, and write parquets to:
      /storage/agrp/barakma/PileupODD/data/{event_name}_overlay_pu{pileup_level}/

    Args:
      r: iterable of file indices to process (e.g. [0, 1, 2] or range(5)).
      event_name: HS dataset prefix (e.g. 'ttbar_pu0').
      pileup_level: mean of the Poisson sampler for pileup events per HS event.
      pu_event_name: pileup-only dataset prefix on HF.
      seed: RNG seed for the sample map.
      num_of_events: if >= 0, restrict HS events (pileup pool is never truncated).
      clusters_cutoff: drop clusters whose calibrated energy sum is below this (GeV).
      pu_files_per_batch: number of PU files concatenated into a shared pool per batch
                          of HS files.  Each HS file in the batch samples from the full
                          combined pool, giving better combinatorics across files.
                          Ignored when `pu_indices` is provided.
      pu_indices: explicit iterable of PU file indices to load into a single shared
                  pool used by ALL HS files in `r` (decouples PU pool from HS files).
                  When None (default), PU pool is derived from `r` in batches of
                  `pu_files_per_batch` (legacy behavior).
      clue_backend: backend for CLUEstering. Default 'gpu cuda' (NVIDIA GPU).
                    Use 'cpu serial' or 'cpu tbb' for CPU-only nodes. Example:
                        run_preprocessing_pipeline(r=[0], clue_backend='cpu serial')
      chunk_size: if > 0, process HS events of each file in chunks of this size
                  through the overlay->cluster->aggregate stages to cap peak RAM
                  (PU pool stays shared across chunks). <=0 disables chunking.
      invisible_pu_prob: per-PU-draw probability of contributing nothing
                  (simulates diffractive events missing the detector). Drawn
                  efficiently via Binomial — no wasted sampling on rolls that
                  would be discarded. Default 0.0 (no change to legacy behavior).
                  Reasonable value: 0.19.
    """
    from huggingface_hub import HfFileSystem
    from pathlib import Path
    import polars as pl
    import tqdm
    import gc
    import time

    fs = HfFileSystem()
    if r is None:
        raise ValueError("Must pass an iterable `r` of file indices to process.")
    number_of_hf_repo_files = 1000

    particle_cols = [
        'event_id', 'particle_id', 'vertex_primary', 'pdg_id',
        'energy', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'parent_id',
    ]
    calo_cols = [
        'event_id', 'detector', 'total_energy', 'x', 'y', 'z',
        'contrib_particle_ids', 'contrib_energies',
    ]

    out_dir = Path(f"/storage/agrp/barakma/PileupODD/data/{event_name}_overlay_pu{pileup_level}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _hf_path(prefix: str, kind: str, idx: int) -> str:
        return (
            f"datasets/CERN/ColliderML-Release-1/data/"
            f"{prefix}_{kind}/train-{idx:05d}-of-{number_of_hf_repo_files:05d}.parquet"
        )

    def _read(prefix: str, kind: str, idx: int, columns=None) -> pl.DataFrame:
        path = _hf_path(prefix, kind, idx)
        if not fs.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with fs.open(path, "rb") as f:
            return pl.read_parquet(f, columns=columns)

    def _load_pu_batch(file_indices) -> tuple:
        """Load and concatenate PU files, offsetting event_ids to be unique."""
        p_list, c_list, t_list = [], [], []
        offset = 0
        for idx in file_indices:
            p = _read(pu_event_name, 'particles', idx, columns=particle_cols)
            c = _read(pu_event_name, 'calo_hits', idx, columns=calo_cols)
            t = _read(pu_event_name, 'tracks', idx)
            # Offset so event_ids don't collide across files.
            max_eid = int(max(p['event_id'].max(), c['event_id'].max(), t['event_id'].max())) + 1
            p_list.append(p.with_columns(pl.col('event_id') + offset))
            c_list.append(c.with_columns(pl.col('event_id') + offset))
            t_list.append(t.with_columns(pl.col('event_id') + offset))
            offset += max_eid
        return pl.concat(p_list), pl.concat(c_list), pl.concat(t_list)

    r_list = list(r)
    if pu_indices is not None:
        batches = [(list(pu_indices), r_list)]
    else:
        batches = [
            (r_list[i:i + pu_files_per_batch], r_list[i:i + pu_files_per_batch])
            for i in range(0, len(r_list), pu_files_per_batch)
        ]

    for pu_batch, hs_batch in tqdm.tqdm(batches, desc="Batches"):
        print(f"\n=== Loading PU pool from files {pu_batch} ===")
        pu_particles, pu_calo_hits, pu_tracks = _load_pu_batch(pu_batch)
        n_pu = pu_calo_hits['event_id'].n_unique()
        print(f"    PU pool: {n_pu} unique pileup events from {len(pu_batch)} file(s).")

        for i in tqdm.tqdm(hs_batch, desc="HS files in batch", leave=False):
            print(f"\n=== File index {i:05d} ===")
            _t0 = time.perf_counter()

            hs_particles = _read(event_name, 'particles', i, columns=particle_cols)
            hs_calo_hits = _read(event_name, 'calo_hits', i, columns=calo_cols)
            hs_tracks = _read(event_name, 'tracks', i)

            preprocessed_data = preprocess_for_model(
                hs_particles=hs_particles, hs_tracks=hs_tracks, hs_calo_hits=hs_calo_hits,
                pu_particles=pu_particles, pu_tracks=pu_tracks, pu_calo_hits=pu_calo_hits,
                pileup_level=pileup_level,
                seed=seed + i,  # different sampling per file index
                num_of_events=num_of_events,
                truth_pt_cut=1, truth_eta_cut=3.0, target_pt_cut=0.3,
                clusters_cutoff=clusters_cutoff,
                clue_backend=clue_backend,
                chunk_size=chunk_size,
                chunk_tmp_dir=chunk_tmp_dir,
                invisible_pu_prob=invisible_pu_prob,
            )

            for key, df in preprocessed_data.items():
                df.write_parquet(out_dir / f"{key}-{i:05d}.parquet")

            del hs_particles, hs_tracks, hs_calo_hits, preprocessed_data
            gc.collect()
            _dt = time.perf_counter() - _t0
            print(f"=== File index {i:05d} done in {_dt:.1f} s ({_dt/60:.2f} min) ===")

        del pu_particles, pu_calo_hits, pu_tracks
        gc.collect()

