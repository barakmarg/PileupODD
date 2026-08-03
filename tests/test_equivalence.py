"""Verify each mode reproduces the ``master`` scripts' results.

For every mode, the original implementation and this branch's run in separate
subprocesses over identical cached input, each writing real parquet files, and
the results are compared.

The comparison is split according to what the pipeline can actually reproduce:

**Exactly comparable.** ``target_particles`` and ``tracks`` are deterministic --
they come out of Stage A, before clustering. These are required to match
``master`` byte for byte, every column, including within-event list order.

**Not exactly comparable.** ``calo_clusters`` and ``target_particles_deps``
depend on the clustering, which is stochastic: CLUE assigns cluster ids in
discovery order and its CUDA backend reduces nondeterministically, so two runs
of *master itself* differ in every cluster-dependent column. These are compared
through label-invariant physics quantities instead -- see
:mod:`tests.invariants`.

The one intended difference is ``cluster_time``: this branch drops it from
``calo_clusters`` in all three modes, so it is excluded where ``master`` still
produced it. ``master``'s overlay script already omitted it.

Requires the ``master`` checkout at ``/storage/agrp/barakma/PileupODD``, a CUDA
device, and network access on the first run (to build the fixtures).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests._equiv_worker import MASTER_ROOT, OUTPUT_KEYS
from tests.invariants import relative_deviation, summarise, tolerance_for

WORKER = Path(__file__).parent / "_equiv_worker.py"

#: Deterministic in every mode: produced before clustering.
EXACT_TABLES = ("target_particles", "tracks")

#: In overlay mode the track list is hard-scatter tracks followed by a block of
#: sampled pileup tracks, and master builds that block straight out of a join
#: whose row order is unspecified -- so its ordering varies run to run. This
#: branch sorts the block (see :func:`colliderml_pflow.overlay.overlay_tracks`),
#: but it cannot match master's arbitrary order, so the comparison there is on
#: contents rather than sequence.
ORDER_INSENSITIVE = {"overlay": ("tracks",)}

#: Cluster-dependent: compared through invariants, not row by row.
STOCHASTIC_TABLES = ("calo_clusters", "target_particles_deps")

#: Dropped by this branch in every mode.
DROPPED_COLUMNS = {"cluster_time"}

MODES = ["hard_scatter", "all_vertices", "overlay"]

#: How much worse than master's own run-to-run spread a branch-vs-master
#: deviation may be, in the self-calibrating test.
BASELINE_MULTIPLE = 3.0
#: Floor for that test: master's spread is sometimes exactly zero on a small
#: fixture, and a pure multiple of zero would be unsatisfiable.
BASELINE_FLOOR = 2e-3


def _run_side(side: str, mode: str, fixture_dir: Path, out_dir: Path) -> Path:
    """Run one implementation in a subprocess; fail loudly if it errors."""
    result = subprocess.run(
        [sys.executable, "-u", str(WORKER), side, mode, str(fixture_dir), str(out_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{side}/{mode} worker exited {result.returncode}\n"
            f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
        )
    return out_dir


@pytest.fixture(scope="session")
def master_available() -> None:
    if not Path(MASTER_ROOT).is_dir():
        pytest.skip(f"master checkout not found at {MASTER_ROOT}")


@pytest.fixture(scope="session")
def outputs(request, fixture_dir, tmp_path_factory, master_available):
    """Per-mode ``{'old': dir, 'new': dir}``, computed once and reused."""
    mode = request.param
    return {
        side: _run_side(side, mode, fixture_dir, tmp_path_factory.mktemp(f"{side}_{mode}"))
        for side in ("old", "new")
    }


def _rows_per_event(frame: pl.DataFrame, event_row: int) -> list:
    """Explode one event's list columns into a sorted list of per-track tuples.

    Used where contents must match but sequence need not. ``hit_ids`` is nested
    (a list per track) so it is converted to a tuple to stay hashable/sortable.
    """
    cols = [c for c in frame.columns if c != "event_id"]
    values = []
    for col in cols:
        cell = frame[col][event_row]
        values.append([tuple(v.to_list()) if hasattr(v, "to_list") else v
                       for v in cell])
    return sorted(zip(*values), key=repr)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("outputs", MODES, indirect=True)
def test_deterministic_tables_match_master_exactly(outputs, request):
    """``target_particles`` and ``tracks`` must reproduce master's.

    These come out of Stage A, before clustering, so they are the tables the
    pipeline can reproduce exactly -- full equality is required, including
    column order, dtypes, and the order of values inside each per-event list.

    The one exception is the overlay track list, whose pileup block master
    leaves in an unspecified order; there, contents are compared instead. See
    :data:`ORDER_INSENSITIVE`.
    """
    mode = request.node.callspec.params["outputs"]
    lenient = ORDER_INSENSITIVE.get(mode, ())

    for key in EXACT_TABLES:
        old = pl.read_parquet(outputs["old"] / f"{key}.parquet").sort("event_id")
        new = pl.read_parquet(outputs["new"] / f"{key}.parquet").sort("event_id")
        assert old.height == new.height, (
            f"{key}: {old.height} events (master) vs {new.height} (branch)"
        )
        assert old.columns == new.columns, (
            f"{key}: columns differ\n  master: {old.columns}\n  branch: {new.columns}"
        )

        if key in lenient:
            for row in range(old.height):
                old_rows = _rows_per_event(old, row)
                new_rows = _rows_per_event(new, row)
                assert len(old_rows) == len(new_rows), (
                    f"{key} event row {row}: {len(old_rows)} tracks (master) vs "
                    f"{len(new_rows)} (branch)"
                )
                assert old_rows == new_rows, (
                    f"{key} event row {row}: track contents differ from master"
                )
        else:
            assert_frame_equal(old, new, check_column_order=True, check_dtypes=True)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("outputs", MODES, indirect=True)
def test_cluster_invariants_match_master(outputs):
    """Cluster-dependent tables must agree on label-invariant physics.

    Cluster labels are not reproducible, but the number of clusters, the total
    energy, the energy per physical vertex and the energy per target particle
    all are -- to within the pipeline's intrinsic stochastic spread.
    """
    old = summarise(outputs["old"])
    new = summarise(outputs["new"])
    deviations = relative_deviation(old, new)

    failures = [
        f"  {key}: master={old[key]:.6g} branch={new[key]:.6g} "
        f"deviation={dev:.3%} > tolerance={tolerance_for(key):.3%}"
        for key, dev in sorted(deviations.items())
        if dev > tolerance_for(key)
    ]
    assert not failures, "invariants outside tolerance:\n" + "\n".join(failures)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("outputs", ["all_vertices"], indirect=True)
def test_branch_is_within_masters_own_spread(outputs, fixture_dir, tmp_path_factory):
    """Self-calibrating check: run master twice, and require the branch to be
    no further from master than master is from itself.

    This is the strongest statement available given stochastic clustering. It
    needs no hand-picked tolerance -- the acceptable spread is measured from
    ``master`` on the same input, in the same session.
    """
    second = _run_side("old", "all_vertices", fixture_dir,
                       tmp_path_factory.mktemp("old_all_vertices_2"))

    master_1 = summarise(outputs["old"])
    master_2 = summarise(second)
    branch = summarise(outputs["new"])

    baseline = relative_deviation(master_1, master_2)
    measured = relative_deviation(master_1, branch)

    failures = []
    for key, dev in sorted(measured.items()):
        budget = max(BASELINE_MULTIPLE * baseline[key], BASELINE_FLOOR)
        if dev > budget:
            failures.append(
                f"  {key}: branch deviates {dev:.3%} but master's own spread is "
                f"{baseline[key]:.3%} (budget {budget:.3%})"
            )
    assert not failures, (
        "branch deviates from master by more than master's own run-to-run "
        "spread:\n" + "\n".join(failures)
    )


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("outputs", MODES, indirect=True)
def test_cluster_time_is_the_only_schema_change(outputs, request):
    """Pin the intended schema change so an accidental one cannot slip through."""
    mode = request.node.callspec.params["outputs"]
    removed, added = {}, {}
    for key in OUTPUT_KEYS:
        old_cols = pl.scan_parquet(outputs["old"] / f"{key}.parquet").collect_schema().names()
        new_cols = pl.scan_parquet(outputs["new"] / f"{key}.parquet").collect_schema().names()
        removed[key] = [c for c in old_cols if c not in new_cols]
        added[key] = [c for c in new_cols if c not in old_cols]

    expected_removed = {key: [] for key in OUTPUT_KEYS}
    if mode != "overlay":
        # master's overlay script already had no cluster_time.
        expected_removed["calo_clusters"] = ["cluster_time"]

    assert removed == expected_removed, f"unexpected column removals: {removed}"
    assert all(not v for v in added.values()), f"unexpected column additions: {added}"


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("outputs", MODES, indirect=True)
def test_dropped_columns_are_absent_from_branch_output(outputs):
    """No table on this branch carries a column the branch is meant to drop."""
    for key in OUTPUT_KEYS:
        cols = set(pl.scan_parquet(outputs["new"] / f"{key}.parquet").collect_schema().names())
        assert not (cols & DROPPED_COLUMNS), (
            f"{key} unexpectedly contains {sorted(cols & DROPPED_COLUMNS)}"
        )
