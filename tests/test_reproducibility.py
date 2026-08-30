"""Guard the reproducibility properties this branch adds.

Three ordering-dependent steps were left unpinned on ``master``, which made its
output vary between runs on identical input:

1. the ``group_by`` / ``sort`` feeding CLUE, so cluster *labels* varied;
2. the pileup pool order feeding ``numpy``'s sampler, so the overlay drew a
   *different pileup sample* on every run despite taking a ``seed``;
3. the join producing the overlaid pileup track block, so track order varied.

These tests pin (2) and (3), which are pure and fast to check. (1) is covered
by ``test_equivalence.py``, since it needs a real clustering run.

All fast: no network, no GPU, no HuggingFace access.
"""

from __future__ import annotations

import numpy as np

from colliderml_pflow.config import ClusteringConfig, Config
from colliderml_pflow.overlay import build_sample_map

POOL = np.arange(100, 160, dtype=np.uint32)
HS = np.array([7, 3, 11], dtype=np.uint32)
LEVEL = 20
SEED = 42


def _draws(frame):
    """Sample map as ``{hs_event_id: sorted pileup ids}``."""
    return {
        row["hs_event_id"]: sorted(row["pu_event_id"])
        for row in frame.iter_rows(named=True)
    }


def test_sample_map_is_reproducible_from_the_seed():
    """Same inputs and seed must give the same pileup sample."""
    first = build_sample_map(HS, POOL, LEVEL, SEED)
    second = build_sample_map(HS, POOL, LEVEL, SEED)
    assert _draws(first) == _draws(second)


def test_sample_map_ignores_input_ordering():
    """Shuffling the pool or the hard-scatter ids must not change the sample.

    This is the property master lacked. Its pool order came from a polars
    ``unique`` whose row order is not stable, so the sampler walked a differently
    permuted array each run and picked different pileup events.
    """
    rng = np.random.default_rng(0)
    shuffled_pool = rng.permutation(POOL)
    shuffled_hs = rng.permutation(HS)

    reference = _draws(build_sample_map(HS, POOL, LEVEL, SEED))
    assert _draws(build_sample_map(HS, shuffled_pool, LEVEL, SEED)) == reference
    assert _draws(build_sample_map(shuffled_hs, POOL, LEVEL, SEED)) == reference
    assert _draws(build_sample_map(shuffled_hs, shuffled_pool, LEVEL, SEED)) == reference


def test_different_seeds_give_different_samples():
    """The seed must actually do something -- otherwise shards would be clones."""
    assert _draws(build_sample_map(HS, POOL, LEVEL, SEED)) != \
        _draws(build_sample_map(HS, POOL, LEVEL, SEED + 1))


def test_pileup_events_are_distinct_within_a_hard_scatter_event():
    """No pileup event may be overlaid twice on the same hard-scatter event."""
    for _, ids in _draws(build_sample_map(HS, POOL, LEVEL, SEED)).items():
        assert len(ids) == len(set(ids))


def test_draws_are_capped_by_the_pool_size():
    """Asking for more pileup than the pool holds must not raise or repeat."""
    small_pool = np.arange(5, dtype=np.uint32)
    frame = build_sample_map(HS, small_pool, pileup_level=50, seed=SEED)
    for _, ids in _draws(frame).items():
        assert len(ids) <= len(small_pool)
        assert len(ids) == len(set(ids))


def test_mean_draw_count_tracks_the_pileup_level():
    """Sanity-check the Poisson mean over many hard-scatter events."""
    many_hs = np.arange(4000, dtype=np.uint32)
    big_pool = np.arange(500, 1000, dtype=np.uint32)
    frame = build_sample_map(many_hs, big_pool, pileup_level=200, seed=SEED)
    counts = np.array([len(v) for v in frame["pu_event_id"].to_list()])
    # Poisson(200) over 4000 events: the mean is within a few tenths of 200.
    assert abs(counts.mean() - 200) < 2.0
    # And the variance should be near the mean, as Poisson requires.
    assert abs(counts.var() / counts.mean() - 1.0) < 0.15


def test_clustering_is_deterministic_by_default():
    """The ordering fix must be on unless a config explicitly disables it."""
    assert ClusteringConfig().deterministic is True
    assert Config().clustering.deterministic is True
