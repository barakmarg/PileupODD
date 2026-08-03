"""Feature normalization statistics for the written dataset.

Computes, in a single streaming pass over the produced parquet shards, the
per-feature moments and quantiles the model's input scaler needs. Quantiles
come from KLL sketches rather than exact sorting, so memory stays bounded no
matter how many shards are scanned.

Emits YAML in the layout the training code expects:

.. code-block:: yaml

    eta:
      type: min_max_sym
      mean: 0.0021
      std: 1.4832
      q25: -1.0
      ...

Ported from ``generate_normalization_stats_sequential`` on ``master``, with two
changes:

- the ``cluster_time`` entry is gone, matching this branch dropping that column
  from ``calo_clusters`` (see the README);
- ``master`` also carried an in-memory twin, ``generate_normalization_yaml``,
  whose only difference was applying a ``sqrt`` transform to ``number_of_hits``.
  The published statistics were produced by the streaming version, so that is
  the behaviour kept here and the divergent twin is not carried over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import yaml

#: Per-feature source and treatment.
#:
#: ``df``        which output table the values come from
#: ``col``       column name; ``None`` means the feature is derived (cluster_pt)
#: ``transform`` optional pre-transform applied before accumulating statistics
#: ``type``      scaling scheme the model applies at load time
CONFIG_SCHEMA: Dict[str, Dict] = {
    "eta":        {"df": "calo_clusters", "col": "cluster_eta", "transform": None, "type": "min_max_sym"},
    "rho":        {"df": "calo_clusters", "col": "cluster_rho", "transform": None, "type": "min_max_sym"},
    "e":          {"df": "calo_clusters", "col": "total_cluster_energy", "transform": "sqrt", "type": "min_max_sym"},
    "pt":         {"df": "tracks",        "col": "pt",          "transform": "sqrt", "type": "min_max_sym"},
    "cluster_pt": {"df": "calo_clusters", "col": None,          "transform": "sqrt", "type": "min_max_sym"},
    "sigma_eta":  {"df": "calo_clusters", "col": "sigma_eta",   "transform": "sqrt", "type": "std"},
    "sigma_phi":  {"df": "calo_clusters", "col": "sigma_phi",   "transform": "sqrt", "type": "std"},
    "sigma_rho":  {"df": "calo_clusters", "col": "sigma_rho",   "transform": "sqrt", "type": "std"},
    "d0":         {"df": "tracks",        "col": "d0",          "transform": None,   "type": "min_max_sym"},
    "z0":         {"df": "tracks",        "col": "z0",          "transform": None,   "type": "min_max_sym"},
    "tanlambda":  {"df": "tracks",        "col": "track_tanlambda", "transform": None, "type": "min_max_sym"},
    "omega":      {"df": "tracks",        "col": "track_omega", "transform": None,   "type": "std"},
    "number_of_hits":  {"df": "calo_clusters", "col": "number_of_hits",  "transform": None,   "type": "min_max_sym"},
    "energy_hits_std": {"df": "calo_clusters", "col": "energy_hits_std", "transform": "sqrt", "type": "std"},
    "max_hit_energy":  {"df": "calo_clusters", "col": "max_hit_energy",  "transform": "sqrt", "type": "min_max_sym"},
}


def _extract_values(df: pl.DataFrame, key: str, schema: Dict) -> np.ndarray:
    """Pull one feature out of a table as a flat, finite float array.

    Args:
        df: the source table, with per-event list columns.
        key: feature name; ``cluster_pt`` is derived rather than read.
        schema: that feature's :data:`CONFIG_SCHEMA` entry.

    Returns:
        Finite values only; empty if the column is absent.
    """
    col_name = schema["col"]
    transform = schema["transform"]
    series = None

    if key == "cluster_pt":
        # Not stored: transverse momentum from cluster energy and pseudorapidity.
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
    return arr[np.isfinite(arr)]


def _smart_fmt(val):
    """Round to 4 dp, or to a plain int when the value is large and near-integral."""
    if isinstance(val, (int, float, np.number)):
        if abs(val) >= 10 and abs(round(val) - val) < 1e-3:
            return int(round(val))
        return float(f"{val:.4f}")
    return val


def generate_normalization_stats(
    data_dir: str | Path,
    max_files: int = 40,
    kll_k: int = 200,
) -> str:
    """Accumulate normalization statistics over written shards.

    Args:
        data_dir: directory holding ``tracks-*.parquet`` and
            ``calo_clusters-*.parquet``. The ``tracks`` files drive shard
            discovery.
        max_files: cap on shards scanned; ``<= 0`` scans all of them. Statistics
            converge quickly, so a few tens of shards is normally enough.
        kll_k: KLL sketch accuracy parameter. Larger is more accurate and uses
            more memory.

    Returns:
        A YAML document, or ``""`` if no shards were found.
    """
    from datasketches import kll_floats_sketch
    from tqdm import tqdm

    stats = {
        key: {"count": 0, "sum": 0.0, "sum_sq": 0.0,
              "min": float('inf'), "max": float('-inf')}
        for key in CONFIG_SCHEMA
    }
    quantile_sketches = {key: kll_floats_sketch(kll_k) for key in CONFIG_SCHEMA}

    path = Path(data_dir)
    track_files = sorted(path.glob("tracks-*.parquet"))
    if not track_files:
        print(f"No tracks-*.parquet files found in {data_dir}. Cannot generate stats.")
        return ""

    indices: List[str] = []
    for f in track_files:
        try:
            indices.append(f.name.split('-')[-1].split('.')[0])
        except Exception:
            pass
    if max_files > 0:
        indices = indices[:max_files]

    required_dfs = {v["df"] for v in CONFIG_SCHEMA.values()}

    # Single pass: moments, min/max and KLL quantiles together.
    for idx_str in tqdm(indices, desc="Computing normalization stats"):
        loaded_dfs = {}
        for df_name in required_dfs:
            fpath = path / f"{df_name}-{idx_str}.parquet"
            if fpath.exists():
                loaded_dfs[df_name] = pl.read_parquet(fpath)

        for key, schema in CONFIG_SCHEMA.items():
            df = loaded_dfs.get(schema["df"])
            if df is None:
                continue
            arr = _extract_values(df, key, schema)
            if len(arr) == 0:
                continue

            try:
                quantile_sketches[key].update(arr)
            except TypeError:
                # Older datasketches builds take scalars only.
                for x in arr:
                    quantile_sketches[key].update(float(x))

            stats[key]["count"] += len(arr)
            stats[key]["sum"] += float(np.sum(arr))
            stats[key]["sum_sq"] += float(np.sum(arr * arr))
            stats[key]["min"] = min(stats[key]["min"], float(np.min(arr)))
            stats[key]["max"] = max(stats[key]["max"], float(np.max(arr)))

    yaml_config = {}
    for key, schema in CONFIG_SCHEMA.items():
        s = stats[key]
        n = s["count"]
        if n == 0:
            continue

        mean_val = s["sum"] / n
        var_val = (s["sum_sq"] - (s["sum"] ** 2 / n)) / (n - 1) if n > 1 else 0.0
        std_val = np.sqrt(max(var_val, 0.0))

        q25_val, q75_val, q95_val, q99_val = _quantiles(
            quantile_sketches.get(key), s["min"], s["max"])

        entry = {"type": schema["type"]}
        if schema["transform"]:
            entry["fn"] = schema["transform"]
        entry.update({
            "mean": float(f"{mean_val:.4f}"),
            "std": float(f"{std_val:.4f}"),
            "q25": _smart_fmt(q25_val),
            "q75": _smart_fmt(q75_val),
            "q95": _smart_fmt(q95_val),
            "q99": _smart_fmt(q99_val),
            "min": _smart_fmt(s["min"]),
            "max": _smart_fmt(s["max"]),
        })
        yaml_config[key] = entry

    return yaml.dump(yaml_config, sort_keys=False, default_flow_style=False)


def _quantiles(sketch, min_val: float, max_val: float):
    """Read q25/q75/q95/q99 from a sketch, falling back to min/max when degenerate."""
    fallback = (float(min_val), float(max_val), float(max_val), float(max_val))
    if max_val <= min_val or sketch is None:
        return fallback

    # The datasketches API varies slightly between builds.
    if hasattr(sketch, "is_empty"):
        empty = bool(sketch.is_empty())
    elif hasattr(sketch, "n"):
        empty = int(sketch.n) == 0
    else:
        empty = False
    if empty:
        return fallback

    return (
        float(sketch.get_quantile(0.25)),
        float(sketch.get_quantile(0.75)),
        float(sketch.get_quantile(0.95)),
        float(sketch.get_quantile(0.99)),
    )


def write_normalization_stats(
    data_dir: str | Path,
    output_path: Optional[str | Path] = None,
    max_files: int = 40,
    kll_k: int = 200,
) -> Path:
    """Compute statistics and write them next to the dataset.

    Args:
        data_dir: directory holding the written shards.
        output_path: destination. Defaults to
            ``<data_dir>/normalization_stats.yaml``.
        max_files: cap on shards scanned.
        kll_k: KLL sketch accuracy parameter.

    Returns:
        The path written.

    Raises:
        RuntimeError: if no shards were found in ``data_dir``.
    """
    text = generate_normalization_stats(data_dir, max_files=max_files, kll_k=kll_k)
    if not text:
        raise RuntimeError(f"no shards found in {data_dir}")
    target = Path(output_path) if output_path else Path(data_dir) / "normalization_stats.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"[NORM STATS] wrote {target}")
    return target
