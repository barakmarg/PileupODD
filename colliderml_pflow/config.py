"""Typed configuration: YAML in, validated dataclasses out.

A run is fully described by one YAML file plus optional ``--set key=value``
overrides. Nothing that affects the output is hardcoded in the pipeline
modules; see ``configs/`` for the presets that reproduce each paper dataset.

The one thing that is deliberately *not* a knob is the vertex policy. It is
implied by ``mode``, because that is exactly how the three original scripts
behaved and making it independently settable would let a run silently drift
from the published datasets:

=================  =====================  ===================================
mode               primary source         pileup source
=================  =====================  ===================================
``hard_scatter``   ``vertex_primary==1``  --
``all_vertices``   all vertices kept      --
``overlay``        ``vertex_primary==1``  overlaid, no vertex filter
=================  =====================  ===================================

Note that ``overlay`` is ``hard_scatter`` plus an overlay stage, not a third
vertex policy: its hard-scatter side applies the same filter. On ``ttbar_pu0``
input that filter is effectively a no-op (every particle already sits on the
one primary vertex), but it is applied, and the snapshot it feeds is what
labels overlaid clusters by originating vertex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

MODES = ("hard_scatter", "all_vertices", "overlay")

CLUE_BACKENDS = ("gpu cuda", "cpu serial", "cpu tbb", "cpu omp")

#: Output tables written per shard, in a stable order.
OUTPUT_KEYS = ("target_particles", "calo_clusters", "tracks", "target_particles_deps")


@dataclass
class Cuts:
    """Selection thresholds applied during preprocessing.

    Args:
        truth_pt: minimum track pT in GeV, and the truth pT threshold used by
            the target-particle definition.
        truth_eta: maximum \\|eta\\| for tracks and truth particles.
        target_pt: minimum pT in GeV for a particle to become a target.
        cluster_energy: minimum calibrated cluster energy in GeV; clusters
            below this are dropped along with their hits.
    """

    truth_pt: float = 1.0
    truth_eta: float = 3.0
    target_pt: float = 0.3
    cluster_energy: float = 0.15


@dataclass
class ClusteringConfig:
    """CLUE parameters and execution backend.

    Args:
        backend: one of ``gpu cuda``, ``cpu serial``, ``cpu tbb``, ``cpu omp``.
            Results are backend-dependent, so comparisons between runs must
            hold this fixed.
        dc: critical distance defining the local-density neighbourhood.
        rhoc: minimum local density required to seed a cluster.
        dm: maximum distance over which a point may attach to a seed.
        ppbin: target points per bin in CLUE's internal spatial tiling.
        deterministic: pin the order in which points reach CLUE so repeated
            runs give identical clusters. ``master`` left this order unpinned,
            so its cluster labels varied run to run; see
            :func:`colliderml_pflow.clustering.clue_clustering`.
    """

    backend: str = "gpu cuda"
    dc: float = 75.88106168184893
    rhoc: float = 104.34315216716726
    dm: float = 87.0967630118376
    ppbin: int = 16
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.backend not in CLUE_BACKENDS:
            raise ValueError(
                f"clustering.backend must be one of {CLUE_BACKENDS}, got {self.backend!r}"
            )


@dataclass
class ToFConfig:
    """Time-of-flight window applied to pileup hits (overlay mode only).

    In a real PU200 bunch crossing, pileup interactions are spread in time, so
    hits arriving outside the read-out window are never recorded. Overlaying
    simulated PU0 events would otherwise pile up an unphysical energy excess,
    because every pileup vertex sits at t=0. Each sampled pileup vertex is
    given a Gaussian time offset and its hits are dropped if the resulting
    corrected hit time falls outside ``window_ns``.

    Applied only to the pileup side: hard-scatter hits are at t=0 in the
    simulation and were already windowed there.

    Args:
        enabled: whether to apply the cut at all.
        sigma_ns: standard deviation of the per-vertex Gaussian time shift.
        window_ns: ``[t_min, t_max]`` acceptance window for the corrected time.
    """

    enabled: bool = True
    sigma_ns: float = 0.185
    window_ns: List[float] = field(default_factory=lambda: [-1.0, 10.0])

    def __post_init__(self) -> None:
        self.window_ns = [float(x) for x in self.window_ns]
        if len(self.window_ns) != 2 or self.window_ns[0] >= self.window_ns[1]:
            raise ValueError(f"tof.window_ns must be [t_min, t_max] with t_min < t_max, got {self.window_ns}")


@dataclass
class OverlayConfig:
    """Synthetic-pileup settings. Read only when ``mode == 'overlay'``.

    Args:
        pu_event_name: HF dataset prefix supplying the pileup events.
        pu_file_indices: shards forming the shared sampling pool. All
            hard-scatter shards in a run draw from this same pool, which gives
            better combinatorics than pairing shard-to-shard.
        pu_max_events: optionally cap the pool size (smoke runs).
        pileup_level: mean of the Poisson distribution for pileup events
            overlaid per hard-scatter event.
        seed: base RNG seed. The seed actually used for shard ``i`` is
            ``seed + i``, so each shard samples differently but reproducibly.
        invisible_pu_prob: probability that a given pileup draw contributes
            nothing, modelling diffractive events that miss the detector.
            Drawn via a Binomial thinning rather than by sampling and
            discarding.
        tof: time-of-flight window settings.
    """

    pu_event_name: str = "pileup_only_pu0"
    pu_file_indices: List[int] = field(default_factory=lambda: [0, 1, 2])
    pu_max_events: Optional[int] = None
    pileup_level: int = 200
    seed: int = 42
    invisible_pu_prob: float = 0.0
    tof: ToFConfig = field(default_factory=ToFConfig)

    def __post_init__(self) -> None:
        if not 0.0 <= self.invisible_pu_prob < 1.0:
            raise ValueError(
                f"overlay.invisible_pu_prob must be in [0, 1), got {self.invisible_pu_prob}"
            )
        if self.pileup_level < 0:
            raise ValueError(f"overlay.pileup_level must be >= 0, got {self.pileup_level}")
        self.pu_file_indices = [int(i) for i in self.pu_file_indices]


@dataclass
class DatasetConfig:
    """Which HuggingFace shards and events to read.

    Args:
        repo: HF dataset repo id.
        shards_total: total shard count, part of the filename
            (``train-00007-of-01000.parquet``).
        event_name: dataset prefix, e.g. ``ttbar_pu200`` or ``ttbar_pu0``.
        file_indices: shard selection. Accepts a plain list, ``{range: [a, b]}``
            (``b`` exclusive), or ``{map: {shard: [event_id, ...]}}`` to pin
            exact events per shard.
        max_events_per_file: process only the first N event ids of each shard.
            Combined with predicate pushdown this makes a smoke run cost
            seconds rather than a multi-GB download.
    """

    repo: str = "CERN/ColliderML-Release-1"
    shards_total: int = 1000
    event_name: str = "ttbar_pu200"
    file_indices: Any = field(default_factory=lambda: {"range": [0, 1]})
    max_events_per_file: Optional[int] = None

    def resolved_file_indices(self) -> List[int]:
        """Shard indices to process, in order."""
        spec = self.file_indices
        if isinstance(spec, dict):
            if "range" in spec:
                lo, hi = spec["range"]
                return list(range(int(lo), int(hi)))
            if "map" in spec:
                return sorted(int(k) for k in spec["map"])
            raise ValueError(f"dataset.file_indices dict must have 'range' or 'map', got {list(spec)}")
        return [int(i) for i in spec]

    def explicit_event_ids(self) -> Optional[Dict[int, List[int]]]:
        """Per-shard event ids when pinned via ``{map: ...}``, else ``None``."""
        if isinstance(self.file_indices, dict) and "map" in self.file_indices:
            return {int(k): [int(e) for e in v] for k, v in self.file_indices["map"].items()}
        return None


@dataclass
class RuntimeConfig:
    """Execution and output settings.

    Args:
        chunk_size: events per spawned worker process. Bounds peak RAM: the
            OS reclaims polars' allocator arenas when each child exits, so
            memory does not accumulate across chunks or shards. ``<= 0``
            processes a whole shard in one child.
        output_dir: where per-shard parquets are written.
        tmp_dir: parent for the temporary per-shard directory holding chunk
            outputs before they are concatenated.
    """

    chunk_size: int = 100
    output_dir: str = "data/${dataset.event_name}_${mode}"
    tmp_dir: str = "data/tmp"


@dataclass
class Config:
    """A complete run specification."""

    mode: str = "all_vertices"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    cuts: Cuts = field(default_factory=Cuts)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def keep_all_vertices(self) -> bool:
        """Whether the primary source keeps every vertex, not just ``vertex_primary == 1``.

        True for ``all_vertices`` only. See the module docstring for why this
        is derived from ``mode`` rather than configured directly.
        """
        return self.mode == "all_vertices"

    @property
    def is_overlay(self) -> bool:
        return self.mode == "overlay"

    def resolved_output_dir(self) -> Path:
        return Path(_interpolate(self.runtime.output_dir, self))

    def resolved_tmp_dir(self) -> Path:
        return Path(_interpolate(self.runtime.tmp_dir, self))

    def describe(self) -> str:
        """Human-readable one-block summary, printed at the start of a run."""
        lines = [
            f"mode              : {self.mode}",
            f"  vertex policy   : {'all vertices' if self.keep_all_vertices else 'vertex_primary == 1'}",
            f"dataset           : {self.dataset.event_name}  ({self.dataset.repo})",
            f"  shards          : {_summarise_indices(self.dataset.resolved_file_indices())}",
            f"  max events/file : {self.dataset.max_events_per_file or 'all'}",
        ]
        if self.is_overlay:
            tof = self.overlay.tof
            tof_desc = (
                f"sigma={tof.sigma_ns} ns window={tof.window_ns} ns"
                if tof.enabled else "off"
            )
            lines += [
                f"overlay           : mu={self.overlay.pileup_level}  seed={self.overlay.seed}",
                f"  pu dataset      : {self.overlay.pu_event_name} shards {self.overlay.pu_file_indices}",
                f"  invisible prob  : {self.overlay.invisible_pu_prob}",
                f"  tof             : {tof_desc}",
            ]
        lines += [
            f"cuts              : truth_pt={self.cuts.truth_pt} truth_eta={self.cuts.truth_eta} "
            f"target_pt={self.cuts.target_pt} cluster_energy={self.cuts.cluster_energy}",
            f"clustering        : {self.clustering.backend}  dc={self.clustering.dc:.4f} "
            f"rhoc={self.clustering.rhoc:.4f} dm={self.clustering.dm:.4f} ppbin={self.clustering.ppbin} "
            f"deterministic={self.clustering.deterministic}",
            f"runtime           : chunk_size={self.runtime.chunk_size}",
            f"  output_dir      : {self.resolved_output_dir()}",
            f"  tmp_dir         : {self.resolved_tmp_dir()}",
        ]
        return "\n".join(lines)


def _summarise_indices(indices: List[int]) -> str:
    if not indices:
        return "(none)"
    if len(indices) <= 6:
        return str(indices)
    return f"[{indices[0]}..{indices[-1]}] ({len(indices)} shards)"


_INTERP = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


def _interpolate(text: str, cfg: "Config") -> str:
    """Expand ``${a.b}`` references against the config object."""
    def repl(m: re.Match) -> str:
        node: Any = cfg
        for part in m.group(1).split("."):
            node = getattr(node, part)
        return str(node)

    return _INTERP.sub(repl, text)


# Which fields hold nested dataclasses. `from __future__ import annotations`
# turns the annotations into strings, so this explicit map is clearer (and more
# robust) than resolving them at runtime.
_NESTED = {
    ("Config", "dataset"): DatasetConfig,
    ("Config", "overlay"): OverlayConfig,
    ("Config", "cuts"): Cuts,
    ("Config", "clustering"): ClusteringConfig,
    ("Config", "runtime"): RuntimeConfig,
    ("OverlayConfig", "tof"): ToFConfig,
}


def _build(cls, data: Any):
    """Recursively instantiate a dataclass tree from plain dicts.

    Unknown keys raise rather than being ignored -- a typo in a config file
    should fail loudly, not silently produce a run with default settings.
    """
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown config key(s) {sorted(unknown)} for {cls.__name__}; "
            f"expected one of {sorted(known)}"
        )
    kwargs = {}
    for key, value in data.items():
        nested = _NESTED.get((cls.__name__, key))
        kwargs[key] = _build(nested, value) if nested is not None else value
    return cls(**kwargs)


def _coerce(value: str) -> Any:
    """Parse a ``--set`` value as YAML so ints/floats/bools/lists work."""
    return yaml.safe_load(value)


def apply_override(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``data['a']['b'] = value`` for ``dotted_key='a.b'``, creating dicts."""
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override {dotted_key!r}: {part!r} is not a mapping")
    node[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: Optional[List[str]] = None) -> Config:
    """Load a YAML config and apply ``key=value`` overrides.

    Args:
        path: YAML file. If ``None``, start from the dataclass defaults.
        overrides: ``["clustering.backend=cpu serial", "runtime.chunk_size=2"]``.
            Values are parsed as YAML, so ``null``, ``true``, ``[0,1]`` all work.

    Returns:
        A validated :class:`Config`.
    """
    data: Dict[str, Any] = {}
    if path is not None:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        apply_override(data, key.strip(), _coerce(raw.strip()))
    return _build(Config, data)
