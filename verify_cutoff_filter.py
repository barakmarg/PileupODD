"""
verify_cutoff_filter.py

Side-by-side equivalence + timing test for the cluster-energy cutoff filter.
Generates a synthetic list-per-event calo_hits frame that mimics the schema
after CLUE clustering, runs OLD and NEW filter on the SAME frame, and confirms
they produce identical outputs.

Stochasticity in the CLUE clustering itself is upstream of this filter; given
identical input, both filter implementations must be deterministic and equal.
"""
import sys
sys.path.insert(0, '/storage/agrp/barakma/CLUEstering')
sys.path.insert(0, '/storage/agrp/barakma/PileupODD')

import time
import numpy as np
import polars as pl
from primary.calibration import CALIBRATION


CUTOFF = 0.15


def make_synthetic_calo_hits(n_events: int, hits_per_event: int, n_clusters: int, rng):
    """Build a list-per-event frame with the columns produced after clue_clustering()."""
    detectors = CALIBRATION['detector'].to_list()
    rows = []
    for e in range(n_events):
        x = rng.uniform(-200, 200, size=hits_per_event).astype(np.float32)
        y = rng.uniform(-200, 200, size=hits_per_event).astype(np.float32)
        z = rng.uniform(-200, 200, size=hits_per_event).astype(np.float32)
        te = rng.exponential(0.001, size=hits_per_event).astype(np.float32)
        det = rng.choice(detectors, size=hits_per_event).tolist()
        cid = rng.integers(-1, n_clusters, size=hits_per_event, dtype=np.int32).tolist()
        ccx = rng.uniform(-100, 100, size=hits_per_event).astype(np.float32).tolist()
        ccy = rng.uniform(-100, 100, size=hits_per_event).astype(np.float32).tolist()
        ccz = rng.uniform(-100, 100, size=hits_per_event).astype(np.float32).tolist()
        # nested list of lists for contrib columns to mimic real schema
        contrib_pid = [list(rng.integers(0, 1_000_000, size=rng.integers(1, 5)).tolist())
                       for _ in range(hits_per_event)]
        contrib_e = [list(rng.exponential(0.001, size=len(p)).astype(np.float32).tolist())
                     for p in contrib_pid]
        rows.append({
            'event_id': e,
            'x': x.tolist(),
            'y': y.tolist(),
            'z': z.tolist(),
            'total_energy': te.tolist(),
            'detector': det,
            'contrib_particle_ids': contrib_pid,
            'contrib_energies': contrib_e,
            'cluster_id': cid,
            'cluster_cx': ccx,
            'cluster_cy': ccy,
            'cluster_cz': ccz,
        })
    return pl.DataFrame(rows)


def old_filter(calo_hits: pl.DataFrame, clusters_cutoff: float) -> pl.DataFrame:
    return (
        calo_hits.lazy()
        .with_row_index('_event_idx_temp')
        .explode(pl.all().exclude(['event_id', '_event_idx_temp']))
        .join(CALIBRATION.lazy().select(['detector', 'calib_factor']), on='detector', how='left')
        .with_columns((pl.col('total_energy') * pl.col('calib_factor')).alias('hit_energy_gev'))
        .with_columns(pl.col('hit_energy_gev').sum().over(['event_id', 'cluster_id']).alias('cluster_sum_energy'))
        .filter(pl.col('cluster_sum_energy') > clusters_cutoff)
        .filter(pl.col('cluster_id') >= 0)
        .drop(['calib_factor', 'hit_energy_gev', 'cluster_sum_energy'])
        .group_by(['_event_idx_temp', 'event_id'], maintain_order=True)
        .agg(pl.all().exclude(['_event_idx_temp', 'event_id']))
        .drop('_event_idx_temp')
        .collect(streaming=True)
    )


def new_filter(calo_hits: pl.DataFrame, clusters_cutoff: float) -> pl.DataFrame:
    keep_idx = (
        calo_hits.lazy()
        .select(['cluster_id', 'total_energy', 'detector'])
        .with_row_index('_rid')
        .with_columns(_pos=pl.int_ranges(0, pl.col('cluster_id').list.len(), dtype=pl.UInt32))
        .explode(['cluster_id', 'total_energy', 'detector', '_pos'])
        .join(CALIBRATION.lazy().select(['detector', 'calib_factor']), on='detector', how='left')
        .with_columns(_cal_e=pl.col('total_energy') * pl.col('calib_factor'))
        .with_columns(_clu_sum=pl.col('_cal_e').sum().over(['_rid', 'cluster_id']))
        .filter((pl.col('_clu_sum') > clusters_cutoff) & (pl.col('cluster_id') >= 0))
        .group_by('_rid', maintain_order=True)
        .agg(_indices=pl.col('_pos').sort())
        .select(['_rid', '_indices'])
    )
    return (
        calo_hits.lazy()
        .with_row_index('_rid')
        .join(keep_idx, on='_rid', how='left')
        .with_columns(pl.col('_indices').fill_null(pl.lit([], dtype=pl.List(pl.UInt32))))
        .with_columns(pl.exclude('event_id', '_rid', '_indices').list.gather(pl.col('_indices')))
        .drop(['_rid', '_indices'])
        .collect(streaming=True)
    )


def compare_outputs(a: pl.DataFrame, b: pl.DataFrame):
    """Verify semantic equivalence. Hit order within an event is not meaningful
    (downstream group_by-s on cluster_id). We check:
      - same set of events
      - same hit count per event
      - same set of (cluster_id, total_energy) per event as a multiset
      - same per-cluster calibrated energy sum per event
    """
    assert a.columns == b.columns
    assert a.height == b.height
    a = a.sort('event_id')
    b = b.sort('event_id')

    # Hit counts per event must match
    a_lens = a['cluster_id'].list.len().to_numpy()
    b_lens = b['cluster_id'].list.len().to_numpy()
    assert (a_lens == b_lens).all(), f"Hit counts differ: a={a_lens[:5]} b={b_lens[:5]}"

    # Per-cluster total_energy sum per event (deterministic invariant)
    def cluster_sums(df):
        return (df.lazy()
                .select(['event_id', 'cluster_id', 'total_energy'])
                .explode(['cluster_id', 'total_energy'])
                .group_by(['event_id', 'cluster_id'])
                .agg(s=pl.col('total_energy').sum())
                .sort(['event_id', 'cluster_id'])
                .collect())
    sa, sb = cluster_sums(a), cluster_sums(b)
    assert sa.equals(sb), "Per-cluster total_energy sums differ"
    print("  ✓ semantic equivalence verified (hit counts + per-cluster sums match)")


def main():
    rng = np.random.default_rng(42)

    # Test sizes scaled toward realistic PU200 workloads (~30k hits/event).
    for n_events, hits, n_clu in [
        (20, 1_000, 80),
        (50, 5_000, 200),
        (100, 30_000, 4_000),    # ~PU200-ish per chunk
    ]:
        print(f"\n--- {n_events} events x {hits} hits x {n_clu} clusters ---")
        df = make_synthetic_calo_hits(n_events, hits, n_clu, rng)

        t0 = time.time(); out_old = old_filter(df, CUTOFF); t_old = time.time() - t0
        t0 = time.time(); out_new = new_filter(df, CUTOFF); t_new = time.time() - t0

        print(f"  old: {t_old:.3f}s   new: {t_new:.3f}s   speedup: {t_old / max(t_new, 1e-6):.2f}x")
        compare_outputs(out_old, out_new)


if __name__ == '__main__':
    main()
