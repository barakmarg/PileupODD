"""Shared test fixtures: a small cached slice of the real ColliderML sample.

The equivalence tests compare this branch against the ``master`` scripts on
identical input, so the input has to be pinned. A few events are fetched once
by predicate pushdown and cached as local parquet; every later run reads the
cache and needs no network.

The cache is derived data and is gitignored -- delete it to refetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colliderml_pflow import hf_io  # noqa: E402

REPO = "CERN/ColliderML-Release-1"
SHARDS_TOTAL = 1000

#: PU200 events are heavy -- a single one carries ~200 interactions' worth of
#: hits -- so two is enough to exercise every code path.
N_PU200_EVENTS = 2
#: PU0 hard-scatter events are light.
N_PU0_EVENTS = 2
#: Pileup pool for the overlay test. Larger than pileup_level so the sampler
#: is not silently capped by the pool size.
N_PU_POOL_EVENTS = 60

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fetch(event_name: str, n_events: int, prefix: str, with_times: bool) -> None:
    """Fetch and cache one triplet, unless it is already on disk."""
    names = [f"{prefix}particles", f"{prefix}tracks", f"{prefix}calo_hits"]
    if all((FIXTURE_DIR / f"{n}.parquet").exists() for n in names):
        return

    print(f"\n[fixtures] fetching {n_events} events of {event_name} ...", flush=True)
    event_ids = hf_io.list_event_ids(REPO, event_name, 0, SHARDS_TOTAL, limit=n_events)
    particles, tracks, calo_hits = hf_io.load_triplet(
        REPO, event_name, 0, SHARDS_TOTAL, event_ids, with_contrib_times=with_times)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(names, (particles, tracks, calo_hits)):
        frame.write_parquet(FIXTURE_DIR / f"{name}.parquet")
        print(f"[fixtures]   {name}: {frame.height} rows", flush=True)


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Directory holding the cached input frames, fetching them if needed.

    Provides three sets:

    - ``particles`` / ``tracks`` / ``calo_hits`` -- ``ttbar_pu200``, used by the
      ``hard_scatter`` and ``all_vertices`` modes.
    - ``hs_*`` -- ``ttbar_pu0``, the overlay hard-scatter side.
    - ``pu_*`` -- ``pileup_only_pu0`` with ``contrib_times``, the overlay
      pileup pool. Only that side needs the times, for the ToF cut.
    """
    # ttbar_pu200 is cached *with* contrib_times because master's hard_scatter
    # and all_vertices pipelines need it to build cluster_time. This branch
    # drops that column and therefore never fetches the times, so the worker
    # removes it again before handing the frame to the new implementation --
    # see `_equiv_worker.run_new`.
    _fetch("ttbar_pu200", N_PU200_EVENTS, "", with_times=True)
    # The overlay hard-scatter side never needed the times on master either.
    _fetch("ttbar_pu0", N_PU0_EVENTS, "hs_", with_times=False)
    _fetch("pileup_only_pu0", N_PU_POOL_EVENTS, "pu_", with_times=True)
    return FIXTURE_DIR
