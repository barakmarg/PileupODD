"""
Diffs the OLD baseline pipeline outputs against the NEW pipeline outputs.

Required:
  - target_particles, tracks, target_particles_deps:    bit-identical
  - calo_clusters legacy columns:                        bit-identical
  - calo_clusters new columns (vertex_primary_*):        consistent with totals
"""
import sys
import pathlib
import polars as pl
import numpy as np


OLD = pathlib.Path("/storage/agrp/barakma/PileupODD/data/_baseline_old")
NEW = pathlib.Path("/storage/agrp/barakma/PileupODD/data/_baseline_new")


def load(d: pathlib.Path, name: str) -> pl.DataFrame:
    return pl.read_parquet(d / f"{name}.parquet")


def diff_lists_allclose(a: pl.Series, b: pl.Series, rtol: float, atol: float) -> bool:
    """Compare two list-of-numeric Series with tolerance (handles Float32 noise)."""
    if len(a) != len(b):
        return False
    for av, bv in zip(a.to_list(), b.to_list()):
        if av is None and bv is None:
            continue
        if av is None or bv is None:
            return False
        ax = np.asarray(av)
        bx = np.asarray(bv)
        if ax.shape != bx.shape:
            return False
        if not np.allclose(ax, bx, rtol=rtol, atol=atol, equal_nan=True):
            return False
    return True


def diff_frames(a: pl.DataFrame, b: pl.DataFrame, name: str, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    """Per-column diff. exact equality first, fall back to allclose for numeric lists."""
    ok = True
    if a.columns != b.columns:
        only_a = set(a.columns) - set(b.columns)
        only_b = set(b.columns) - set(a.columns)
        print(f"  ! schema diverged: only_old={sorted(only_a)} only_new={sorted(only_b)}")
        ok = False
    common = [c for c in a.columns if c in b.columns]
    for c in common:
        sa, sb = a[c], b[c]
        if sa.equals(sb):
            continue
        if isinstance(sa.dtype, pl.List) and sa.dtype.inner in (
            pl.Float32, pl.Float64
        ) and diff_lists_allclose(sa, sb, rtol=rtol, atol=atol):
            print(f"  ~ {name}.{c}: equal under allclose(rtol={rtol})")
            continue
        print(f"  X {name}.{c}: DIFFER")
        ok = False
    return ok


def main() -> int:
    failed = []

    # 1. Bit-identical (or numerically equivalent) DataFrames
    for key in ("target_particles", "tracks", "target_particles_deps"):
        a = load(OLD, key).sort("event_id")
        b = load(NEW, key).sort("event_id")
        print(f"=== {key} === old_rows={len(a)} new_rows={len(b)}")
        if diff_frames(a, b, key):
            print(f"  OK identical")
        else:
            failed.append(key)

    # 2. calo_clusters: legacy columns must match; new columns checked separately
    new_cols = {"vertex_primary_indices", "vertex_primary_energies"}
    a = load(OLD, "calo_clusters").sort("event_id")
    b_full = load(NEW, "calo_clusters").sort("event_id")
    b = b_full.drop(*new_cols)
    print(f"=== calo_clusters (legacy cols) === old_rows={len(a)} new_rows={len(b)}")
    if diff_frames(a, b, "calo_clusters"):
        print("  OK legacy columns identical")
    else:
        failed.append("calo_clusters legacy")

    # 3. New vertex deps consistency:
    #    sum(vertex_primary_energies[i]) ~= total_cluster_energy[i]
    #    len(vertex_primary_*) per row matches len(cluster_id) per row
    print("=== calo_clusters new vertex_primary columns ===")
    n_clusters_total = 0
    n_mismatch = 0
    max_rel = 0.0
    null_vp_seen = False
    for row in b_full.iter_rows(named=True):
        tce = row["total_cluster_energy"]
        vpe = row["vertex_primary_energies"]
        vpi = row["vertex_primary_indices"]

        if not (len(tce) == len(vpe) == len(vpi)):
            print(f"  X event {row['event_id']}: list lengths mismatch "
                  f"clusters={len(tce)} vp_idx={len(vpi)} vp_e={len(vpe)}")
            failed.append("calo_clusters length mismatch")
            break
        for i, (e_total, vp, ve) in enumerate(zip(tce, vpi, vpe)):
            n_clusters_total += 1
            if any(v is None for v in vp):
                null_vp_seen = True
            if list(vp) != sorted(vp):
                print(f"  X event {row['event_id']} cluster {i}: vp not sorted: {vp}")
                failed.append("vp sort")
                break
            ssum = float(sum(ve))
            denom = max(abs(e_total), 1e-9)
            rel = abs(ssum - e_total) / denom
            if rel > max_rel:
                max_rel = rel
            if not np.isclose(ssum, e_total, rtol=1e-4, atol=1e-3):
                n_mismatch += 1
    print(f"  clusters checked: {n_clusters_total}")
    print(f"  null vertex_primary seen: {null_vp_seen}")
    print(f"  max rel error sum(vp_energies) vs total_cluster_energy: {max_rel:.3e}")
    print(f"  mismatched clusters (rtol=1e-4): {n_mismatch}")
    if null_vp_seen or n_mismatch:
        failed.append("vertex_primary energy sum")

    # 4. tracks already had vertex_primary; report distribution
    t = load(NEW, "tracks")
    if "vertex_primary" not in t.columns:
        print("X tracks missing vertex_primary")
        failed.append("tracks.vertex_primary")
    else:
        flat = t.select(pl.col("vertex_primary").explode()).get_column("vertex_primary")
        unique_vps = sorted(set(flat.unique().to_list()))
        n_pileup = (flat > 1).sum()
        print(f"=== tracks.vertex_primary === unique={unique_vps[:10]}{'...' if len(unique_vps)>10 else ''}  n_pileup_tracks={n_pileup} / {len(flat)}")

    print()
    if failed:
        print(f"FAIL: {failed}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
