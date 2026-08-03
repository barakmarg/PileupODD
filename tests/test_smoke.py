"""End-to-end runs of the real CLI, one per mode.

Exercises the whole path -- HuggingFace read, chunked spawn workers, clustering,
aggregation, parquet write, then the normalization and split subcommands -- on
four events, which takes seconds rather than hours.

These are integration tests: they check the pipeline runs and produces the
expected schema. Whether the *numbers* are right is what ``test_equivalence.py``
establishes.

Needs network access and a CUDA device.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

from colliderml_pflow.config import OUTPUT_KEYS

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"

#: Overlay must start from a pileup-free sample.
MODE_EVENT_NAME = {
    "hard_scatter": "ttbar_pu200",
    "all_vertices": "ttbar_pu200",
    "overlay": "ttbar_pu0",
}

EXPECTED_CLUSTER_COLUMNS = {
    "event_id", "cluster_id", "total_cluster_energy", "hcal_energy", "hcal_fraction",
    "sigma_eta", "sigma_phi", "sigma_rho", "number_of_hits", "energy_hits_std",
    "max_hit_energy", "cluster_phi", "cluster_eta", "cluster_rho",
    "vertex_primary_indices", "vertex_primary_energies",
}


def _cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-u", "-m", "colliderml_pflow", *args],
        cwd=cwd, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": __import__("os").environ["PATH"],
             "HOME": __import__("os").environ.get("HOME", "")},
    )
    if result.returncode != 0:
        pytest.fail(
            f"CLI {' '.join(args)} exited {result.returncode}\n"
            f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )
    return result


@pytest.mark.network
@pytest.mark.gpu
@pytest.mark.parametrize("mode", list(MODE_EVENT_NAME))
def test_preprocess_writes_all_tables(mode, tmp_path):
    """A full run must write all four tables with the expected schema."""
    out_dir = tmp_path / f"out_{mode}"
    _cli(
        "preprocess", "--config", str(SMOKE_CONFIG),
        "--set", f"mode={mode}",
        "--set", f"dataset.event_name={MODE_EVENT_NAME[mode]}",
        "--set", f"runtime.output_dir={out_dir}",
        "--set", f"runtime.tmp_dir={tmp_path / 'tmp'}",
        cwd=tmp_path,
    )

    for key in OUTPUT_KEYS:
        path = out_dir / f"{key}-00000.parquet"
        assert path.exists(), f"{key} was not written"
        frame = pl.read_parquet(path)
        assert frame.height > 0, f"{key} is empty"
        assert "event_id" in frame.columns
        # One row per event, everything else a per-event list.
        list_cols = [c for c in frame.columns
                     if c != "event_id" and isinstance(frame.schema[c], pl.List)]
        assert list_cols, f"{key} has no list columns -- unexpected layout"

    clusters = pl.read_parquet(out_dir / "calo_clusters-00000.parquet")
    assert set(clusters.columns) == EXPECTED_CLUSTER_COLUMNS, (
        f"calo_clusters schema drifted: "
        f"unexpected={sorted(set(clusters.columns) - EXPECTED_CLUSTER_COLUMNS)}, "
        f"missing={sorted(EXPECTED_CLUSTER_COLUMNS - set(clusters.columns))}"
    )
    assert "cluster_time" not in clusters.columns


@pytest.mark.network
@pytest.mark.gpu
def test_event_counts_are_consistent_across_tables(tmp_path):
    """Every table must cover the same events, so downstream joins line up."""
    out_dir = tmp_path / "out"
    _cli(
        "preprocess", "--config", str(SMOKE_CONFIG),
        "--set", "mode=all_vertices",
        "--set", f"runtime.output_dir={out_dir}",
        "--set", f"runtime.tmp_dir={tmp_path / 'tmp'}",
        cwd=tmp_path,
    )
    event_sets = {
        key: set(pl.read_parquet(out_dir / f"{key}-00000.parquet")["event_id"].to_list())
        for key in OUTPUT_KEYS
    }
    reference = event_sets["target_particles"]
    for key, ids in event_sets.items():
        assert ids == reference, (
            f"{key} covers different events than target_particles: "
            f"only in {key}: {sorted(ids - reference)}, "
            f"missing from {key}: {sorted(reference - ids)}"
        )
    assert len(reference) == 4, f"expected the 4 configured events, got {len(reference)}"


@pytest.mark.network
@pytest.mark.gpu
def test_norm_stats_and_split(tmp_path):
    """The two post-processing subcommands must run on a written dataset."""
    out_dir = tmp_path / "out"
    common = [
        "--config", str(SMOKE_CONFIG),
        "--set", "mode=all_vertices",
        "--set", f"runtime.output_dir={out_dir}",
        "--set", f"runtime.tmp_dir={tmp_path / 'tmp'}",
    ]
    _cli("preprocess", *common, cwd=tmp_path)

    _cli("norm-stats", *common, "--max-files", "0", cwd=tmp_path)
    stats_path = out_dir / "normalization_stats.yaml"
    assert stats_path.exists()
    stats = yaml.safe_load(stats_path.read_text())
    assert "cluster_time" not in stats, "normalization spec still references cluster_time"
    for feature in ("eta", "rho", "e", "pt", "number_of_hits"):
        assert feature in stats, f"normalization stats missing {feature}"
        assert {"type", "mean", "std", "min", "max"} <= set(stats[feature])

    _cli("split", *common, cwd=tmp_path)
    total = 0
    for split in ("train", "val", "test"):
        path = out_dir / split / "target_particles-00000.parquet"
        assert path.exists(), f"{split} split not written"
        total += pl.read_parquet(path).height
    assert total == 4, f"splits cover {total} events, expected the original 4"


def test_submit_dry_run_splits_the_shard_range(tmp_path):
    """The submitter must emit one job per group, each pinned to its own shards."""
    result = _cli(
        "submit", "--config", str(REPO_ROOT / "configs" / "ttbar_pu200_all_vertices.yaml"),
        "--dry-run", "--group-size", "2",
        "--set", "dataset.file_indices=[0,1,2,3,4]",
        cwd=tmp_path,
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("qsub")]
    assert len(lines) == 3, f"expected 3 groups of 2 from 5 shards, got {len(lines)}"
    # The group's own shard list must be the last override, so it wins.
    assert lines[0].rstrip().endswith("'dataset.file_indices=[0, 1]'"), lines[0]
    assert lines[1].rstrip().endswith("'dataset.file_indices=[2, 3]'"), lines[1]
    assert lines[2].rstrip().endswith("'dataset.file_indices=[4]'"), lines[2]
