from typing import Dict
import polars as pl
import yaml # type: ignore

from sklearn.model_selection import train_test_split
"""
accepts datasets=
    return {
        "target_particles": target_particles,
        "calo_clusters": calo_clusters,
        "tracks": tracks,
        "target_particles_deps": target_particles_deps_aggrigated
    }
"""
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
