"""Configuration loading, validation and the mode/vertex-policy contract.

The vertex policy is derived from ``mode`` rather than configured, so that a run
cannot silently drift from the published datasets. These tests pin that mapping,
along with the shipped presets and the ``--set`` override mechanism.

Fast: no network, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from colliderml_pflow.config import (
    CPU_BACKENDS, GPU_BACKEND, ClusteringConfig, Config, DatasetConfig, load_config,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
PRESETS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("mode,expected", [
    ("hard_scatter", False),
    ("overlay", False),
    ("all_vertices", True),
])
def test_vertex_policy_follows_mode(mode, expected):
    """Only all_vertices keeps every vertex.

    overlay is hard_scatter plus an overlay stage, so its hard-scatter side
    applies the same ``vertex_primary == 1`` filter -- not a third policy.
    """
    assert Config(mode=mode).keep_all_vertices is expected


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        Config(mode="all_verticies")


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_cpu_backends_warn_loudly(backend):
    """Selecting a CPU backend must warn: those backends produce invalid data.

    CLUEstering's CPU implementations emit infinite coordinate and energy values
    and collapse most hits into one cluster, so a dataset built with them is
    unusable. They stay selectable for debugging, but never silently.
    """
    with pytest.warns(RuntimeWarning, match="UNUSABLE"):
        ClusteringConfig(backend=backend)


def test_gpu_backend_does_not_warn(recwarn):
    ClusteringConfig(backend=GPU_BACKEND)
    assert not [w for w in recwarn if w.category is RuntimeWarning]


def test_default_backend_is_the_gpu_one():
    assert ClusteringConfig().backend == GPU_BACKEND
    assert Config().clustering.backend == GPU_BACKEND


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.stem)
def test_presets_use_the_gpu_backend(preset):
    """No shipped config may select a backend that yields invalid clusters."""
    cfg = load_config(preset)
    assert cfg.clustering.backend == GPU_BACKEND, (
        f"{preset.name} selects {cfg.clustering.backend!r}, which produces unusable clusters"
    )


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="clustering.backend must be one of"):
        load_config(None, ["clustering.backend=gpu rocm"])


def test_unknown_config_key_is_rejected():
    """A typo must fail loudly rather than silently running with defaults."""
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(None, ["cuts.truth_ptt=5"])


def test_overrides_are_parsed_as_yaml():
    cfg = load_config(None, [
        "runtime.chunk_size=7",
        "dataset.max_events_per_file=null",
        "dataset.file_indices=[3,4,5]",
        "clustering.deterministic=false",
    ])
    assert cfg.runtime.chunk_size == 7
    assert cfg.dataset.max_events_per_file is None
    assert cfg.dataset.resolved_file_indices() == [3, 4, 5]
    assert cfg.clustering.deterministic is False


def test_later_overrides_win():
    cfg = load_config(None, ["runtime.chunk_size=7", "runtime.chunk_size=9"])
    assert cfg.runtime.chunk_size == 9


@pytest.mark.parametrize("spec,expected", [
    ({"range": [0, 4]}, [0, 1, 2, 3]),
    ([5, 2, 9], [5, 2, 9]),
    ({"map": {1: [10, 11], 0: [20]}}, [0, 1]),
])
def test_file_index_specs(spec, expected):
    assert DatasetConfig(file_indices=spec).resolved_file_indices() == expected


def test_pinned_event_ids_are_returned_per_shard():
    cfg = DatasetConfig(file_indices={"map": {0: [10, 11], 3: [20]}})
    assert cfg.explicit_event_ids() == {0: [10, 11], 3: [20]}


def test_plain_file_index_list_pins_no_events():
    assert DatasetConfig(file_indices=[0, 1]).explicit_event_ids() is None


def test_output_dir_interpolation():
    cfg = load_config(None, [
        "mode=all_vertices",
        "dataset.event_name=ttbar_pu200",
        "runtime.output_dir=data/${dataset.event_name}_${mode}",
    ])
    assert cfg.resolved_output_dir() == Path("data/ttbar_pu200_all_vertices")


def test_overlay_output_dir_interpolates_pileup_level():
    cfg = load_config(CONFIG_DIR / "ttbar_pu0_overlay_pu200.yaml")
    assert cfg.resolved_output_dir() == Path("data/ttbar_pu0_overlay_pu200")


def test_tof_window_must_be_ordered():
    with pytest.raises(ValueError, match="window_ns"):
        load_config(None, ["overlay.tof.window_ns=[10.0, -1.0]"])


def test_presets_exist():
    assert PRESETS, f"no preset configs found in {CONFIG_DIR}"


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.stem)
def test_preset_loads_and_describes(preset):
    """Every shipped preset must load, validate, and be printable."""
    cfg = load_config(preset)
    assert cfg.mode in ("hard_scatter", "all_vertices", "overlay")
    assert cfg.dataset.resolved_file_indices()
    text = cfg.describe()
    assert cfg.mode in text
    assert "${" not in str(cfg.resolved_output_dir()), "output_dir left uninterpolated"


def test_overlay_presets_use_a_pu0_hard_scatter_input():
    """Overlay must start from a pileup-free sample, or it would double-count."""
    for preset in PRESETS:
        cfg = load_config(preset)
        if cfg.is_overlay:
            assert "pu0" in cfg.dataset.event_name, (
                f"{preset.name}: overlay input {cfg.dataset.event_name!r} is not a PU0 sample"
            )
