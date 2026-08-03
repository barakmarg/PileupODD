"""Compare this branch's output against a dataset produced by the ``master`` branch.

Regenerates a chosen set of events with this branch and compares the result,
column by column, against the same events in an existing dataset directory.

**Why a subset of events is enough.** Every stage of the pipeline is per-event:
clustering runs per event, the masks and target selection are per-event
expressions, and ``particle_idx`` / ``cluster_idx`` are dense indices *within*
an event. So regenerating 5 events of a shard gives exactly what a full-shard
run gives for those 5 events. That makes a genuine value-level comparison cheap.

**How columns are compared.** Never by row position -- always joined on stable
physical identifiers, because the row order and the per-event index columns are
not stable across runs:

============================  =========================================================
table                          comparison
============================  =========================================================
``target_particles``           joined on ``(event_id, particle_id)``; every physics
                               column compared exactly
``tracks``                     joined on ``(event_id, track_id)``; every column
                               compared exactly
``calo_clusters``              label-invariant aggregates only -- CLUE cluster ids are
                               per-run labels (see below)
``target_particles_deps``      ``particle_idx`` resolved back to ``particle_id`` via
                               ``target_particles``, energy summed per particle, then
                               joined on ``(event_id, particle_id)``. The particle set
                               must match exactly; a small fraction of energies may
                               differ (see ``--max-differing-frac``)
============================  =========================================================

``cluster_id`` and ``cluster_idx`` are labels handed out in CLUE's discovery
order, so they carry no meaning across runs and nothing can be joined on them.
That is why the two cluster-dependent tables are compared through quantities
that do not depend on the labelling.

**Calibrating "match".** Clustering is stochastic, so a stored dataset is one
draw rather than the answer. Measured on 3 events of
``dihiggs_pu200_all_vertices_paper`` (8452 target particles, 8413 with deposits):
re-running ``master``'s own code differs from its own stored output on 9
particles' summed deposit energy, while this branch differs on 7. The
deterministic tables -- ``target_particles`` and ``tracks`` -- match the stored
output exactly, for both.

**Overlay mode** additionally restricts the comparison to entries that are
independent of the pileup draw, and of the ``filter_orphans_and_reindex`` fix
that postdates ``data/ttbar_pu0_overlay_pu200`` -- see
:data:`OVERLAY_HS_ONLY_NOTE` and the README.

Usage::

    python tools/compare_to_reference.py \\
        --reference /storage/agrp/barakma/PileupODD/data/dihiggs_pu200_all_vertices_paper \\
        --mode all_vertices --event-name dihiggs_pu200 --shard 0 --n-events 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Physics columns of target_particles. Excludes particle_idx, which is a
#: per-event positional index rather than a value.
TARGET_VALUE_COLS = [
    "pdg_id", "energy", "eta", "phi", "px", "py", "pz", "pt",
    "has_track", "vertex_primary", "vx", "vy", "vz",
]

#: Track columns excluded from the value comparison, and why.
TRACK_SKIP = {
    "track_id",              # the join key
    "particle_idx",          # per-event positional index
    "source_pileup_event_id",  # overlay only, and a per-run sample label
}

#: In overlay mode the stored dataset's pileup rows are not comparable, for two
#: independent reasons, so the comparison keeps only hard-scatter tracks:
#:
#: 1. They come from a different pileup draw. Master's sampler walked an
#:    order-unstable pool, so its pileup content is not reproducible by any code.
#: 2. `data/ttbar_pu0_overlay_pu200` predates the `filter_orphans_and_reindex`
#:    fix (commit c3b2171), which is the *only* thing that fix changed: pileup
#:    tracks whose event-local `majority_particle_id` collided with a
#:    hard-scatter target's were wired to that target instead of being marked
#:    `-1`. Measured on the stored dataset: 6.0% of pileup tracks carry a
#:    spurious `particle_idx >= 0`.
#:
#: Hard-scatter rows are untouched on both counts -- the fix's `otherwise`
#: branch is the original expression -- so they compare exactly. And because
#: `valid_ids` is built from `majority_particle_id`, which the fix did not
#: touch, `target_particles` and `target_particles_deps` are unaffected by the
#: bug; they are still pileup-dependent through the clustering, though.
OVERLAY_HS_ONLY_NOTE = "hard-scatter tracks only"


def hard_scatter_only(tracks: pl.DataFrame) -> pl.DataFrame:
    """Keep only hard-scatter track rows (null ``source_pileup_event_id``)."""
    if "source_pileup_event_id" not in tracks.columns:
        return tracks
    return tracks.filter(pl.col("source_pileup_event_id").is_null())


def hs_vertex_energy(clusters: pl.DataFrame, event_ids: List[int]) -> pl.DataFrame:
    """Total energy attributed to the hard-scatter vertex, per event.

    In overlay mode the total cluster energy depends on which pileup events were
    drawn, so it is not comparable. The energy attributed to ``vertex_primary ==
    1`` is the hard-scatter contribution, which is conserved regardless of how
    the hits happen to be partitioned into clusters or which pileup landed on
    top -- making it the one cluster-level quantity worth comparing here.
    """
    clusters = clusters.filter(pl.col("event_id").is_in(event_ids)).sort("event_id")
    rows = []
    for row in range(clusters.height):
        total = 0.0
        for idxs, ens in zip(clusters["vertex_primary_indices"][row],
                             clusters["vertex_primary_energies"][row]):
            for v, e in zip(idxs.to_list(), ens.to_list()):
                if v == 1:
                    total += e
        rows.append({"event_id": clusters["event_id"][row], "hs_energy": total})
    return pl.DataFrame(rows)


def explode_table(frame: pl.DataFrame, event_ids: List[int]) -> pl.DataFrame:
    """Flatten a one-row-per-event table into one row per item."""
    frame = frame.filter(pl.col("event_id").is_in(event_ids))
    list_cols = [c for c in frame.columns
                 if c != "event_id" and isinstance(frame.schema[c], pl.List)]
    return frame.select(["event_id", *list_cols]).explode(list_cols)


def resolve_deps_to_particle_ids(deps: pl.DataFrame, targets: pl.DataFrame,
                                 event_ids: List[int]) -> pl.DataFrame:
    """Sum deposited energy per (event_id, particle_id).

    ``target_particles_deps`` refers to particles by ``particle_idx``, a dense
    per-event index. Resolving it back through ``target_particles`` gives a key
    that means the same thing in both datasets, which ``particle_idx`` alone
    does not once the surviving particle set differs at all.
    """
    idx_to_id = (
        targets.filter(pl.col("event_id").is_in(event_ids))
        .select(["event_id", "particle_id", "particle_idx"])
        .explode(["particle_id", "particle_idx"])
        .with_columns(pl.col("particle_idx").cast(pl.Int64))
    )
    return (
        explode_table(deps, event_ids)
        .select(["event_id", "particle_idx", "total_energy_deps_in_cluster"])
        .with_columns(pl.col("particle_idx").cast(pl.Int64))
        .join(idx_to_id, on=["event_id", "particle_idx"], how="left")
        .group_by(["event_id", "particle_id"])
        .agg(pl.col("total_energy_deps_in_cluster").sum().alias("energy_deps_total"))
        .sort(["event_id", "particle_id"])
    )


def cluster_invariants(clusters: pl.DataFrame, event_ids: List[int]) -> Dict[str, float]:
    """Per-event quantities that do not depend on the cluster labelling."""
    clusters = clusters.filter(pl.col("event_id").is_in(event_ids)).sort("event_id")
    out: Dict[str, float] = {}
    for row in range(clusters.height):
        eid = clusters["event_id"][row]
        out[f"e{eid}.n_clusters"] = float(len(clusters["cluster_id"][row]))
        out[f"e{eid}.E_total"] = float(sum(clusters["total_cluster_energy"][row].to_list()))
        out[f"e{eid}.E_hcal"] = float(sum(clusters["hcal_energy"][row].to_list()))
        out[f"e{eid}.n_hits"] = float(sum(clusters["number_of_hits"][row].to_list()))
        per_vertex: Dict[int, float] = {}
        for idxs, ens in zip(clusters["vertex_primary_indices"][row],
                             clusters["vertex_primary_energies"][row]):
            for v, e in zip(idxs.to_list(), ens.to_list()):
                per_vertex[v] = per_vertex.get(v, 0.0) + e
        out[f"e{eid}.n_vtx"] = float(len(per_vertex))
        out[f"e{eid}.E_vtx_total"] = float(sum(per_vertex.values()))
    return out


def _report_join(name: str, ref: pl.DataFrame, new: pl.DataFrame,
                 keys: List[str], value_cols: List[str], rtol: float) -> bool:
    """Join on `keys` and compare `value_cols`. Returns True if it all matched."""
    ref_keys = ref.select(keys).unique()
    new_keys = new.select(keys).unique()
    only_ref = ref_keys.join(new_keys, on=keys, how="anti").height
    only_new = new_keys.join(ref_keys, on=keys, how="anti").height

    print(f"\n--- {name}")
    print(f"    rows: reference {ref.height}, branch {new.height}")
    print(f"    keys only in reference: {only_ref}    only in branch: {only_new}")

    merged = ref.join(new, on=keys, how="inner", suffix="__new")
    print(f"    matched on {keys}: {merged.height}")
    if merged.height == 0:
        print("    !! nothing matched -- cannot compare values")
        return False

    ok = only_ref == 0 and only_new == 0
    for col in value_cols:
        left, right = pl.col(col), pl.col(f"{col}__new")
        # Nulls are meaningful here -- a track whose particle was not matched has
        # a null production vertex. Two nulls agree; a null against a value does
        # not. `eq_missing` gives exactly that, and `ne_missing` its negation.
        both_null = left.is_null() & right.is_null()

        if merged.schema[col].is_float():
            rel = ((left - right).abs()
                   / pl.max_horizontal(left.abs(), pl.lit(1e-30)))
            mismatch = pl.when(both_null).then(False) \
                         .when(left.is_null() | right.is_null()).then(True) \
                         .otherwise(rel > rtol)
            stats = merged.select(
                bad=mismatch.sum(),
                worst=pl.when(both_null | left.is_null() | right.is_null())
                        .then(None).otherwise(rel).max(),
                nulls=both_null.sum(),
            ).row(0)
            bad, worst, nulls = int(stats[0]), stats[1] or 0.0, int(stats[2])
            note = f"  ({nulls} null on both sides)" if nulls else ""
            status = "OK  " if bad == 0 else "FAIL"
            print(f"    {status} {col:34s} mismatches={bad:6d}  worst rel={worst:.3e}{note}")
        else:
            bad = int(merged.select(left.ne_missing(right).sum()).item())
            status = "OK  " if bad == 0 else "FAIL"
            print(f"    {status} {col:34s} mismatches={bad:6d}")
        ok &= bad == 0
    return ok


def _report_deposits(ref: pl.DataFrame, new: pl.DataFrame,
                     rtol: float, max_frac: float) -> bool:
    """Compare summed deposit energy per particle, allowing a small differing fraction.

    Deposits inherit the clustering's stochasticity. A cluster whose energy sits
    near the ``cluster_energy`` cutoff can be kept in one run and dropped in the
    next, which moves that entire cluster's energy off the particles that fed it.
    So a handful of particles out of thousands legitimately differ, and requiring
    zero would be a test that ``master`` fails against its own stored output.

    The particle *set* is still required to match exactly.
    """
    keys = ["event_id", "particle_id"]
    only_ref = ref.select(keys).join(new.select(keys), on=keys, how="anti").height
    only_new = new.select(keys).join(ref.select(keys), on=keys, how="anti").height

    merged = ref.join(new, on=keys, how="inner", suffix="__new").with_columns(
        ((pl.col("energy_deps_total__new") - pl.col("energy_deps_total")).abs()
         / pl.col("energy_deps_total").abs().clip(lower_bound=1e-30)).alias("_rel")
    )
    n_diff = int((merged["_rel"] > rtol).sum())
    frac = n_diff / merged.height if merged.height else 0.0
    worst = float(merged["_rel"].max() or 0.0)

    print("\n--- target_particles_deps (energy summed per particle, keyed on particle_id)")
    print(f"    particles: reference {ref.height}, branch {new.height}")
    print(f"    keys only in reference: {only_ref}    only in branch: {only_new}")
    ok = only_ref == 0 and only_new == 0
    if not ok:
        print("    FAIL the particle set differs")
    verdict = "OK  " if frac <= max_frac else "FAIL"
    print(f"    {verdict} {n_diff}/{merged.height} particles differ >{rtol:.1%} "
          f"({frac:.3%} of them; allowed {max_frac:.2%})   worst rel={worst:.3e}")
    return ok and frac <= max_frac


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True,
                    help="Dataset directory produced by the master branch.")
    ap.add_argument("--mode", required=True,
                    choices=["hard_scatter", "all_vertices", "overlay"])
    ap.add_argument("--event-name", required=True, help="HF dataset prefix.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-events", type=int, default=5,
                    help="Events from that shard to regenerate and compare.")
    ap.add_argument("--chunk-size", type=int, default=2)
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="Relative tolerance for float columns (default: 1e-6).")
    ap.add_argument("--cluster-rtol", type=float, default=5e-3,
                    help="Relative tolerance for cluster invariants (default: 0.5%%).")
    ap.add_argument("--max-differing-frac", type=float, default=5e-3,
                    help="Fraction of particles whose summed deposit energy may differ "
                         "beyond --cluster-rtol (default: 0.5%%). Deposits inherit the "
                         "clustering's stochasticity: a cluster sitting near the energy "
                         "cutoff can be kept in one run and dropped in the next, moving "
                         "that whole cluster's energy. Master differs from its own stored "
                         "output by a comparable fraction.")
    ap.add_argument("--keep", type=Path, default=None,
                    help="Keep regenerated output here instead of a temp dir.")
    args = ap.parse_args()

    ref_dir = args.reference
    ref_targets = pl.read_parquet(ref_dir / f"target_particles-{args.shard:05d}.parquet")
    event_ids = sorted(ref_targets["event_id"].to_list())[: args.n_events]
    is_overlay = args.mode == "overlay"
    print(f"Reference : {ref_dir}")
    print(f"Mode      : {args.mode}   dataset: {args.event_name}   shard: {args.shard:05d}")
    print(f"Events    : {event_ids}")
    if is_overlay:
        print("\nOverlay mode: comparing only the pileup-independent entries.")
        print("  - tracks       : hard-scatter rows only")
        print("  - calo_clusters: hard-scatter vertex energy only")
        print("  - deposits     : reported, not gated (depend on the pileup draw)")

    out_root = args.keep
    tmp_ctx = tempfile.TemporaryDirectory(prefix="cmp_") if out_root is None else None
    if out_root is None:
        out_root = Path(tmp_ctx.name)
    out_dir = out_root / "regenerated"

    cmd = [
        sys.executable, "-u", "-m", "colliderml_pflow", "preprocess",
        "--set", f"mode={args.mode}",
        "--set", f"dataset.event_name={args.event_name}",
        "--set", f"dataset.file_indices={{map: {{{args.shard}: {event_ids}}}}}",
        "--set", f"runtime.chunk_size={args.chunk_size}",
        "--set", f"runtime.output_dir={out_dir}",
        "--set", f"runtime.tmp_dir={out_root / 'tmp'}",
    ]
    print(f"\nRegenerating with:\n  {' '.join(cmd[4:])}\n")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        return 1

    all_ok = True

    # --- target_particles: joined on the stable (event_id, particle_id) -------
    ref_tp = explode_table(ref_targets, event_ids)
    new_targets = pl.read_parquet(out_dir / f"target_particles-{args.shard:05d}.parquet")
    new_tp = explode_table(new_targets, event_ids)
    all_ok &= _report_join("target_particles", ref_tp, new_tp,
                           ["event_id", "particle_id"], TARGET_VALUE_COLS, args.rtol)

    # --- tracks: joined on the stable (event_id, track_id) --------------------
    ref_tr = explode_table(
        pl.read_parquet(ref_dir / f"tracks-{args.shard:05d}.parquet"), event_ids)
    new_tr = explode_table(
        pl.read_parquet(out_dir / f"tracks-{args.shard:05d}.parquet"), event_ids)
    track_label = "tracks"
    skip = set(TRACK_SKIP)
    if is_overlay:
        ref_tr, new_tr = hard_scatter_only(ref_tr), hard_scatter_only(new_tr)
        track_label = f"tracks ({OVERLAY_HS_ONLY_NOTE})"
        # particle_idx IS comparable on hard-scatter rows: the fix only rewrote
        # the pileup branch, leaving the hard-scatter expression unchanged.
        skip.discard("particle_idx")
    track_cols = [c for c in ref_tr.columns
                  if c in new_tr.columns and c not in skip and c != "event_id"
                  and not isinstance(ref_tr.schema[c], pl.List)]
    all_ok &= _report_join(track_label, ref_tr, new_tr,
                           ["event_id", "track_id"], track_cols, args.rtol)

    # --- target_particles_deps: energy per particle, keyed on particle_id -----
    ref_deps = resolve_deps_to_particle_ids(
        pl.read_parquet(ref_dir / f"target_particles_deps-{args.shard:05d}.parquet"),
        ref_targets, event_ids)
    new_deps = resolve_deps_to_particle_ids(
        pl.read_parquet(out_dir / f"target_particles_deps-{args.shard:05d}.parquet"),
        new_targets, event_ids)
    if is_overlay:
        # Deposits are attributed through clusters of the *overlaid* hits, so a
        # different pileup draw changes them. Report, do not gate.
        _report_deposits(ref_deps, new_deps, args.cluster_rtol, 1.0)
        print("         (informational: overlay deposits depend on the pileup draw)")
    else:
        all_ok &= _report_deposits(ref_deps, new_deps, args.cluster_rtol,
                                   args.max_differing_frac)

    # --- calo_clusters: label-invariant aggregates only ----------------------
    ref_clusters = pl.read_parquet(ref_dir / f"calo_clusters-{args.shard:05d}.parquet")
    new_clusters = pl.read_parquet(out_dir / f"calo_clusters-{args.shard:05d}.parquet")

    if is_overlay:
        # Only the hard-scatter contribution is pileup-independent.
        ref_hs = hs_vertex_energy(ref_clusters, event_ids)
        new_hs = hs_vertex_energy(new_clusters, event_ids)
        all_ok &= _report_join("calo_clusters (hard-scatter vertex energy only)",
                               ref_hs, new_hs, ["event_id"], ["hs_energy"],
                               args.cluster_rtol)
        print("\n" + "=" * 70)
        print("RESULT: MATCH" if all_ok else "RESULT: MISMATCH")
        print("=" * 70)
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return 0 if all_ok else 1

    ref_inv = cluster_invariants(ref_clusters, event_ids)
    new_inv = cluster_invariants(new_clusters, event_ids)
    print("\n--- calo_clusters (label-invariant aggregates; cluster ids are per-run labels)")
    worst = 0.0
    for key in sorted(ref_inv):
        a, b = ref_inv[key], new_inv.get(key)
        if b is None:
            print(f"    FAIL {key:28s} missing from branch output")
            all_ok = False
            continue
        dev = abs(b - a) / abs(a) if a else abs(b)
        worst = max(worst, dev)
        if dev > args.cluster_rtol:
            print(f"    FAIL {key:28s} reference={a:.6g} branch={b:.6g} dev={dev:.3%}")
            all_ok = False
    print(f"    worst deviation across {len(ref_inv)} aggregates: {worst:.4%} "
          f"(tolerance {args.cluster_rtol:.2%})")

    print("\n" + "=" * 70)
    print("RESULT: MATCH" if all_ok else "RESULT: MISMATCH")
    print("=" * 70)
    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
