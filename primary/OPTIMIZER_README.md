# Multi-Event CLUE Optimizer

Optimizes **unified CLUE parameters** (dc, rhoc, dm, ppbin) across multiple events with **averaged scoring**.

Unlike the original single-event optimizer, this uses CLUEstering library with a single set of parameters that applies to all events together.

## Files

### 1. `multi_event_optimizer.py` (Main Library)
Core optimizer implementation with the `MultiEventOptimizer` class.

**Key Features:**
- ✅ Multi-event support: Optimizes across multiple events simultaneously
- ✅ Unified parameters: dc, rhoc, dm, ppbin (not separate ECAL/HCAL)
- ✅ Averaged scoring: Computes mean metric across all events for robust tuning
- ✅ Vectorized processing: Efficient data handling (from notebook approach)
- ✅ GPU-accelerated: Uses CLUEstering with CUDA backend
- ✅ Soft constraints: Penalty-based constraints on cluster count and metric values

**Usage:**
```python
from primary.multi_event_optimizer import run_multi_event_optimizer
import polars as pl

# Load your data
calo_hits = pl.read_parquet("path/to/calo_hits.parquet")

# Run optimizer
best_params, study = run_multi_event_optimizer(
    calo_hits=calo_hits,
    max_events=50,      # Use first 50 events
    n_trials=100,       # Run 100 Optuna trials
    seed=42
)

# Results
print(best_params)
# Output: {'dc': 85.2, 'rhoc': 42.5, 'dm': 156.3, 'ppbin': 18}
```

### 2. `run_multi_event_optimizer.py` (CLI Runner)
Command-line interface for convenient execution.

**Usage from Terminal:**
```bash
# Default: 50 events, 100 trials
python primary/run_multi_event_optimizer.py

# Custom parameters
python primary/run_multi_event_optimizer.py \
  --max-events 100 \
  --n-trials 200 \
  --num-files 2
```

**Usage in Jupyter:**
```python
from run_multi_event_optimizer import main

best_params, study = main(
    max_events=100,
    n_trials=200,
    num_files=2
)
```

## Parameters to Optimize

| Parameter | Range | Default | Meaning |
|-----------|-------|---------|---------|
| **dc** | 15.0 - 150.0 | 100.977 | Local density radius (mm) |
| **rhoc** | 0.0 - 100.0 | 50.0 | Local density threshold |
| **dm** | dc - 2×dc | 120.166 | Distance to nearest higher density |
| **ppbin** | 8 - 32 | 16 | Points per bin |

## Architecture

```
Input: Polars DataFrame with calo_hits
   ↓
_prepare_data()
   ├─ Calibrate energy
   ├─ Split by event
   └─ Create point arrays
   ↓
Optuna Optimization Loop:
   ├─ Sample parameters (Optuna TPE)
   ├─ _run_clustering() [All events sequentially]
   ├─ Compute cluster IDs and stats
   ├─ Compute metric & average scores  ← KEY: Averaged across all events
   ├─ Apply soft constraints
   └─ Return total objective
   ↓
Output: Best parameters (dict), Optuna study object
```

## Example: Complete Workflow

```python
# Setup
import sys
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')

from primary.multi_event_optimizer import run_multi_event_optimizer
import polars as pl
from huggingface_hub import HfFileSystem

# Load sample data
fs = HfFileSystem()
calo_hits_list = []
for i in range(1):
    path = f"datasets/CERN/ColliderML-Release-1/data/ttbar_pu200_calo_hits/train-{i:05d}-of-01000.parquet"
    with fs.open(path, "rb") as f:
        calo_hits_list.append(pl.read_parquet(f))
calo_hits = pl.concat(calo_hits_list)

# Run optimization on 50 events, 100 trials
best_params, study = run_multi_event_optimizer(
    calo_hits=calo_hits,
    max_events=50,
    n_trials=100,
    seed=42
)

# Inspect results
print(f"Best objective: {study.best_value}")
print(f"Best params: {best_params}")

# View optimization history with Optuna
import optuna
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

## Custom Metric Function (from Your Code)

Use your actual evaluation metric:

```python
from primary.preprocessing import number_of_clusters_per_particle

def evaluate(calo, calo_deps_mappings, particles_hard_scatter):
    """
    Metric: fraction of particles with 0 contributing clusters (lower is better).

    Args:
        calo: Calo dataframe with cluster assignments
        calo_deps_mappings: Particle dependency mappings
        particles_hard_scatter: Hard scatter particles

    Returns:
        float: Fraction of particles with 0 clusters (0-1, lower is better)
    """
    from primary.preprocessing import number_of_clusters_per_particle

    c = number_of_clusters_per_particle(
        calo_hits_with_clusters=calo,
        ancestors=calo_deps_mappings,
        particles=particles_hard_scatter,
        cut_off_percent=0.05,
        pt_cut=1.0,
        eta_cut=3.0
    )

    # Fraction of particles not in any cluster
    return len(c.filter(pl.col('num_contributing_clusters') == 0)) / len(c)

# Run with custom metric
best_params, study = run_multi_event_optimizer(
    calo_hits=calo_hits,
    max_events=50,
    n_trials=100,
    metric_fn=evaluate,
    calo_deps_mappings=calo_deps_mappings,
    particles_hard_scatter=particles_hard_scatter
)
```

**What it measures:**
- Fraction of hard-scatter particles with pT > 1 GeV and |η| < 3.0
- That have 0 energy contribution in any cluster
- **Lower = better** (fewer "missed" particles)

## Output

Returns tuple: `(best_params_dict, optuna_study)`

Example best_params:
```python
{
    'dc': 85.234,      # Local density radius
    'rhoc': 42.156,    # Local density threshold
    'dm': 156.789,     # Distance to nearest higher density
    'ppbin': 18        # Points per bin
}
```

## Notes

- Uses **CLUEstering** library with GPU CUDA backend
- Assumes calibration constants in `CALIBRATION` (from `primary.calibration`)
- **All events use the same parameters** (unified approach)
- **Metric direction: MINIMIZE**
  - Default noise ratio: lower is better (fewer hits marked as noise)
  - Custom metric: lower is better (fewer particles with 0 clusters)
  - If you want to maximize a metric, negate it: `return 1 - metric`
- Penalizes: excessive cluster count (>20k), high metric values (>0.5)
- Suitable for iterative tuning: run multiple times with different seeds
- Final evaluation prints statistics based on metric type

## Comparison: Single vs Multi-Event

| Aspect | Original | Multi-Event |
|--------|----------|------------|
| Events | Single (hardcoded) | Multiple (configurable) |
| Scoring | Single event | **Averaged across events** |
| Parameters | Separate ECAL/HCAL | **Unified (dc, rhoc, dm, ppbin)** |
| Data | Row loops | **Vectorized** |
| Library | Custom | **CLUEstering** |
| Output | DataFrame + study | **Parameters + study** |
| CLI Support | None | ✅ Yes |
