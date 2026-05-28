"""
Identify the exact Higgs decay chain in each event by combining N orphan
groups (N=1..4). The first N that yields a particle set whose invariant
mass is "exactly" 125 GeV (within HIGGS_TOL) wins.

An "orphan group" = vp=1 particles whose parent_id does NOT appear in this
event's particle_id list (i.e. their parent is unstored — typically the
Higgs itself or an unstored intermediate W/Z/τ), bucketed by that missing
parent_id.

  Strategy n=1 — orphan siblings: one orphan group of ≥2 daughters whose
    parent is unstored (the Higgs). Catches H→γγ, bb̄, ττ, cc̄, gg, μμ
    when the simulator gives the Higgs's immediate daughters as orphans.

  Strategy n=2 — two orphan-groups: combine two distinct orphan groups
    (each with its own missing parent) and see if the union sums to MH.
    Catches H→W⁺W⁻ → 4 fermions, H→ZZ → 4 fermions when the W/Z
    intermediates are unstored.

  Strategy n=3 / n=4 — three / four orphan-groups: same idea but with
    longer cascades (H → 2 intermediates → 2 further unstored
    intermediates → leaves, with 3 or 4 distinct surviving leaf groups).

For each event we print the strategy that matched and the exact set of
particles whose 4-momenta sum to 125 GeV.
"""

import argparse
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/storage/agrp/barakma/PileupODD")

from load_higgs_diphoton_events import DEFAULT_COLUMNS, load_events  # noqa: E402
from classify_hf_decay_channels import B_HADRON_PDGS, DEFAULT_LABEL_DIR  # noqa: E402
from inspect_hbb_top_bottoms import load_bb_event_ids  # noqa: E402

HIGGS_MASS = 125.0
HIGGS_TOL  = 0.01   # GeV — "exactly 125" up to floating-point + simulator quantization

PDG_NAME = {
    1: "d", -1: "anti-d", 2: "u", -2: "anti-u", 3: "s", -3: "anti-s",
    4: "c", -4: "anti-c", 5: "b", -5: "anti-b", 6: "t", -6: "anti-t",
    11: "e-", -11: "e+", 12: "νe", -12: "anti-νe",
    13: "μ-", -13: "μ+", 14: "νμ", -14: "anti-νμ",
    15: "τ-", -15: "τ+", 16: "ντ", -16: "anti-ντ",
    21: "g", 22: "γ", 23: "Z", 24: "W+", -24: "W-", 25: "H",
    111: "π0", 211: "π+", -211: "π-",
    130: "K0L", 310: "K0S", 311: "K0", -311: "anti-K0", 321: "K+", -321: "K-",
    411: "D+", -411: "D-", 421: "D0", -421: "anti-D0",
    511: "B0", -511: "anti-B0", 513: "B*0", -513: "anti-B*0",
    521: "B+", -521: "B-", 523: "B*+", -523: "B*-",
    531: "Bs0", -531: "anti-Bs0", 5122: "Λb", -5122: "anti-Λb",
    2212: "p", -2212: "anti-p", 2112: "n", -2112: "anti-n",
}


def pdg_name(p: int) -> str:
    if p in PDG_NAME:
        return PDG_NAME[p]
    a = abs(p)
    if 500 <= a < 600 or 5000 <= a < 6000:
        return f"b-had({p:+d})"
    if 400 <= a < 500 or 4000 <= a < 5000:
        return f"c-had({p:+d})"
    return f"pdg={p:+d}"


def _invariant_mass(px, py, pz, E, idxs) -> float:
    pxs = float(px[idxs].sum()); pys = float(py[idxs].sum())
    pzs = float(pz[idxs].sum()); Es  = float(E[idxs].sum())
    m2 = Es * Es - (pxs * pxs + pys * pys + pzs * pzs)
    return math.sqrt(max(m2, 0.0))


def _build_orphan_groups(vp, particle_id, parent_id) -> dict[int, list[int]]:
    """Particles at vp=1 whose parent_id is NOT in this event's particle_id
    set, grouped by their (missing) parent_id."""
    pid_set = set(int(p) for p in particle_id)
    orphan_mask = (vp == 1) & ~np.isin(parent_id, np.array(list(pid_set), dtype=parent_id.dtype))
    orphan_idx = np.where(orphan_mask)[0]
    groups: dict[int, list[int]] = defaultdict(list)
    for i in orphan_idx:
        groups[int(parent_id[i])].append(int(i))
    return dict(groups)


def strategy_n_orphan_groups(n: int,
                             pdg, vp, particle_id, parent_id,
                             px, py, pz, E) -> dict | None:
    """Try every combination of n distinct orphan groups; return the first
    union whose invariant mass is exactly MH."""
    groups = _build_orphan_groups(vp, particle_id, parent_id)
    keys = [k for k, v in groups.items() if len(v) >= 1]
    if len(keys) < n:
        return None
    for combo in combinations(keys, n):
        merged: list[int] = []
        for k in combo:
            merged.extend(groups[k])
        if len(merged) < 2:
            continue
        m = _invariant_mass(px, py, pz, E, np.array(merged))
        if abs(m - HIGGS_MASS) < HIGGS_TOL:
            return {"strategy": n, "mass": m,
                    "parent_ids": list(combo),
                    "particle_idxs": merged,
                    "groups": [(k, groups[k]) for k in combo]}
    return None


def _collect_descendants(particle_id, parent_id,
                         root_idxs: list[int]) -> set[int]:
    """Forward BFS through parent_id graph: collect every particle whose
    ancestry includes any of root_idxs. Includes the roots themselves.
    Since each parent's 4-momentum is conserved as the sum of its children,
    summing the LEAVES of this descendant set gives back the roots' total
    4-momentum exactly (no double counting, no contribution from ISR/UE)."""
    n = len(particle_id)
    children_of: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        children_of[int(parent_id[i])].append(i)
    descendants: set[int] = set(int(r) for r in root_idxs)
    queue: list[int] = list(int(r) for r in root_idxs)
    while queue:
        i = queue.pop()
        for ch in children_of.get(int(particle_id[i]), []):
            if ch not in descendants:
                descendants.add(ch)
                queue.append(ch)
    return descendants


def _cone_invariant_masses(pdg, px, py, pz, E,
                           family_particle_idxs: list[int],
                           dr_cuts: list[float],
                           eta_cut: float | None = None,
                           ) -> dict:
    """Take the matched family (the particles that summed to MH=125),
    pick the two most-energetic B-hadrons inside it, and for each ΔR cut
    return the invariant mass of the family particles within ΔR of EITHER
    of those two B-hadrons. If eta_cut is set, only family particles with
    |η| < eta_cut (including the B-hadrons used as cone centres) are
    considered — that subset's max possible mass is no longer guaranteed
    to be MH because forward family fragments are excluded."""
    fam = np.asarray(family_particle_idxs, dtype=int)
    if fam.size == 0:
        return {}

    # Local kinematics for the family particles
    pt  = np.hypot(px[fam], py[fam])
    eta = np.arcsinh(pz[fam] / np.maximum(pt, 1e-30))
    phi = np.arctan2(py[fam], px[fam])

    # Apply η cut to family particles (and therefore also to the top-B
    # candidates since they're picked from this subset).
    if eta_cut is not None:
        eta_mask = np.abs(eta) < eta_cut
        if not eta_mask.any():
            return {"_no_B": True}
        fam = fam[eta_mask]
        pt = pt[eta_mask]
        eta = eta[eta_mask]
        phi = phi[eta_mask]

    # Two highest-E B-hadrons among the (possibly η-filtered) family
    is_B = np.isin(np.abs(pdg[fam]), list(B_HADRON_PDGS))
    if not is_B.any():
        return {"_no_B": True}
    b_local = np.where(is_B)[0]
    b_local = b_local[np.argsort(E[fam][b_local])[::-1]][:2]
    b_etas = eta[b_local]
    b_phis = phi[b_local]

    # Sanity: full family sum (should equal 125 exactly)
    m_total = _invariant_mass(px, py, pz, E, fam)
    out: dict = {"_meta": {
        "n_family": int(fam.size),
        "M_total":  m_total,
        "top_B":    [int(fam[i]) for i in b_local],
    }}

    for cut in dr_cuts:
        cut2 = cut * cut
        in_cone = np.zeros(fam.size, dtype=bool)
        for be, bp in zip(b_etas, b_phis):
            dphi = phi - bp
            dphi = np.where(dphi >  math.pi, dphi - 2 * math.pi, dphi)
            dphi = np.where(dphi < -math.pi, dphi + 2 * math.pi, dphi)
            in_cone |= ((eta - be) ** 2 + dphi * dphi < cut2)
        sel = fam[in_cone]
        if sel.size == 0:
            out[cut] = {"n": 0, "M": 0.0, "E": 0.0, "pT": 0.0}
            continue
        m = _invariant_mass(px, py, pz, E, sel)
        pxs = float(px[sel].sum()); pys = float(py[sel].sum())
        out[cut] = {
            "n":  int(sel.size),
            "M":  m,
            "E":  float(E[sel].sum()),
            "pT": math.hypot(pxs, pys),
        }
    return out


def _print_particle(i: int, role: str, pdg, px, py, pz, E, parent_id) -> None:
    pt = math.hypot(float(px[i]), float(py[i]))
    p_mag = math.sqrt(float(px[i])**2 + float(py[i])**2 + float(pz[i])**2)
    eta = math.asinh(float(pz[i]) / max(pt, 1e-30))
    phi = math.atan2(float(py[i]), float(px[i]))
    print(f"    {role:<10}  idx={i:>4}  pdg={int(pdg[i]):+5d} "
          f"({pdg_name(int(pdg[i])):<10})  "
          f"pT={pt:7.2f}  η={eta:+5.2f}  φ={phi:+5.2f}  E={float(E[i]):7.2f}  "
          f"parent_id={int(parent_id[i])}")


def inspect_event(eid: int, fi: int, raw: dict) -> dict:
    """Returns {strategy: int|None, mass: float|None, abs_diff: float|None}
    where strategy is 1/2/3 (or None if no match)."""
    pdg         = np.asarray(raw["pdg_id"],         dtype=np.int64)
    vp          = np.asarray(raw["vertex_primary"], dtype=np.int64)
    particle_id = np.asarray(raw["particle_id"],    dtype=np.int64)
    parent_id   = np.asarray(raw["parent_id"],      dtype=np.int64)
    px = np.asarray(raw["px"], dtype=np.float64)
    py = np.asarray(raw["py"], dtype=np.float64)
    pz = np.asarray(raw["pz"], dtype=np.float64)
    E  = np.asarray(raw["energy"], dtype=np.float64)

    print(f"\n=== event_id={eid}  (file_idx={fi})  N_particles={pdg.size} ===")

    for n in (1, 2, 3, 4):
        result = strategy_n_orphan_groups(
            n, pdg, vp, particle_id, parent_id, px, py, pz, E,
        )
        if result is None:
            print(f"  strategy n={n} orphan group(s): no match")
            continue

        s = result["strategy"]
        m = result["mass"]
        print(f"\n  ✓ HIGGS IDENTIFIED via strategy n={s} orphan group(s)  →  "
              f"M = {m:.4f} GeV (tol ±{HIGGS_TOL})")
        for par_id, idxs in result["groups"]:
            # Look up the parent particle (if it IS stored — strategy 1 case)
            par_pos = np.where(particle_id == par_id)[0]
            if par_pos.size == 1:
                pi = int(par_pos[0])
                par_str = (f"parent_id={par_id} = stored particle pdg="
                           f"{int(pdg[pi]):+d} ({pdg_name(int(pdg[pi]))})")
            else:
                par_str = f"parent_id={par_id}  (NOT in event — unstored ancestor)"
            print(f"\n  group from {par_str}  ({len(idxs)} sibling{'s' if len(idxs)!=1 else ''}):")
            # Sort daughters by pT desc for readability
            pt_idx = np.argsort(np.hypot(px[idxs], py[idxs]))[::-1]
            for i in np.array(idxs)[pt_idx]:
                _print_particle(int(i), "Higgs-d", pdg, px, py, pz, E, parent_id)

        # Sanity: also print the total invariant mass of all chosen particles
        idxs_all = np.array(result["particle_idxs"])
        m_all = _invariant_mass(px, py, pz, E, idxs_all)
        E_tot = float(E[idxs_all].sum())
        pt_tot = math.hypot(float(px[idxs_all].sum()), float(py[idxs_all].sum()))
        print(f"\n  combined ({len(idxs_all)} particles): "
              f"E={E_tot:.2f}  pT={pt_tot:.2f}  M={m_all:.4f} GeV")

        # Among the matched family, find the two highest-E B-hadrons, then
        # for each ΔR cut sum the family particles within ΔR of either of
        # those two B-hadrons. At ΔR → ∞ this hits 125 by construction.
        family_idxs = list(result["particle_idxs"])
        cones = _cone_invariant_masses(
            pdg, px, py, pz, E,
            family_particle_idxs=family_idxs,
            dr_cuts=[0.4, 0.5, 0.6, 0.7],
        )
        # Parallel computation with central-only family (|η|<3 on every
        # family particle including the B-hadron cone centres).
        cones_eta3 = _cone_invariant_masses(
            pdg, px, py, pz, E,
            family_particle_idxs=family_idxs,
            dr_cuts=[0.4, 0.5, 0.6, 0.7],
            eta_cut=3.0,
        )
        cone_masses_by_dr: dict[float, float] = {}
        if cones.get("_no_B"):
            print(f"\n  (no B-hadron in matched family → ΔR sweep skipped)")
        elif cones:
            meta = cones.pop("_meta")
            print(f"\n  Top-2 B-hadrons inside the matched family:")
            for b_i in meta["top_B"]:
                _print_particle(int(b_i), "topB", pdg, px, py, pz, E, parent_id)
            print(f"\n  Family sum sanity: N={meta['n_family']}  "
                  f"M={meta['M_total']:.4f}  (expected MH = {HIGGS_MASS})")
            print(f"\n  Invariant mass of family particles within ΔR of either top-B:")
            print(f"    {'ΔR':>5}  {'N':>5}  {'E':>10}  {'pT':>8}  "
                  f"{'M':>10}  {'Δ from MH':>10}  {'M/MH':>7}")
            for cut, info in cones.items():
                d_mh = info["M"] - HIGGS_MASS
                frac = info["M"] / HIGGS_MASS
                print(f"    {cut:>5.1f}  {info['n']:>5d}  {info['E']:>10.2f}  "
                      f"{info['pT']:>8.2f}  {info['M']:>10.4f}  "
                      f"{d_mh:>+10.4f}  {frac:>7.1%}")
                cone_masses_by_dr[float(cut)] = float(info["M"])
        # Extract per-ΔR masses for the |η|<3 variant (no print spam — the
        # interesting comparison is in the histogram plot)
        cone_masses_eta3: dict[float, float] = {}
        if not cones_eta3.get("_no_B") and cones_eta3:
            cones_eta3.pop("_meta", None)
            for cut, info in cones_eta3.items():
                cone_masses_eta3[float(cut)] = float(info["M"])

        return {"strategy": s, "mass": m, "abs_diff": abs(m - HIGGS_MASS),
                "cone_masses": cone_masses_by_dr,
                "cone_masses_eta3": cone_masses_eta3}

    print(f"  ✗ Higgs NOT identified by any strategy")
    return {"strategy": None, "mass": None, "abs_diff": None,
            "cone_masses": {}, "cone_masses_eta3": {}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR),
                        help=f"Where the bb̄ labels live (default: {DEFAULT_LABEL_DIR})")
    parser.add_argument("--n-events", type=int, default=2000,
                        help="Number of bb̄ events to inspect (default: 100)")
    parser.add_argument("--plot-out", type=str,
                        default="/storage/agrp/barakma/PileupODD/analysis/plots/"
                                "cone_invariant_mass.png",
                        help="Output PNG path for the per-ΔR M histogram")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    file_to_events = load_bb_event_ids(label_dir, args.n_events)
    flat = sorted((fi, eid) for fi, eids in file_to_events.items() for eid in eids)
    print(f"selected {len(flat)} events from {label_dir}")

    by_file: dict[int, list[int]] = {}
    for fi, eid in flat:
        by_file.setdefault(fi, []).append(eid)

    all_results: list[dict] = []
    for fi, eids in sorted(by_file.items()):
        particles = load_events({fi: eids}, kind="particles",
                                columns=DEFAULT_COLUMNS["particles"])
        by_eid = {int(r["event_id"]): r
                  for r in particles.iter_rows(named=True)}
        for eid in eids:
            row = by_eid.get(eid)
            if row is None:
                print(f"(event_id={eid} not present after load — skipping)")
                continue
            res = inspect_event(eid, fi, row)
            res["event_id"] = eid
            res["file_idx"] = fi
            all_results.append(res)

    _print_summary(all_results)
    plot_out = Path(args.plot_out)
    _plot_cone_mass_histograms(
        all_results, plot_out,
        masses_key="cone_masses",
        title_extra="(no η cut)",
    )
    eta3_path = plot_out.with_name(plot_out.stem + "_eta3" + plot_out.suffix)
    _plot_cone_mass_histograms(
        all_results, eta3_path,
        masses_key="cone_masses_eta3",
        title_extra="(family particles restricted to |η|<3)",
    )


def _plot_cone_mass_histograms(results: list[dict], out_path: Path,
                               masses_key: str = "cone_masses",
                               title_extra: str = "") -> None:
    """Per-ΔR histograms of the cone-restricted family invariant mass."""
    import matplotlib
    matplotlib.use("Agg")  # no display
    import matplotlib.pyplot as plt

    per_dr: dict[float, list[float]] = defaultdict(list)
    for r in results:
        for dr, m in (r.get(masses_key) or {}).items():
            per_dr[float(dr)].append(float(m))

    if not per_dr:
        print("(no cone-mass data to plot)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0.0, 140.0, 71)  # 2 GeV bins, 0..140 GeV
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    drs_sorted = sorted(per_dr.keys())
    for dr, color in zip(drs_sorted, colors):
        masses = np.array(per_dr[dr])
        ax.hist(
            masses, bins=bins, histtype="step", linewidth=2.0,
            color=color,
            label=f"ΔR < {dr}   (N={len(masses)}, mean={masses.mean():.2f}, "
                  f"median={np.median(masses):.2f})",
        )

    ax.axvline(HIGGS_MASS, color="k", linestyle="--", linewidth=1.5,
               label=f"M_H = {HIGGS_MASS:.1f} GeV")
    ax.set_xlabel("Invariant mass of family particles within ΔR of either top-B [GeV]")
    ax.set_ylabel("Events")
    n_per_dr = sum(len(v) for v in per_dr.values()) // max(len(per_dr), 1)
    title = f"H→family cone-restricted invariant mass ({n_per_dr} events per ΔR)"
    if title_extra:
        title += f"  {title_extra}"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nsaved cone-mass histogram → {out_path}")


def _print_summary(results: list[dict]) -> None:
    """Aggregate: how many events did each strategy nail, and how close to MH."""
    n_total = len(results)
    by_strategy: dict[int | None, list[dict]] = {1: [], 2: [], 3: [], 4: [], None: []}
    for r in results:
        by_strategy.setdefault(r["strategy"], []).append(r)

    print("\n" + "=" * 70)
    print(f"=== summary across {n_total} events ===")
    print("=" * 70)
    print(f"{'strategy':<32}  {'count':>6}  {'frac':>6}  "
          f"{'mean |M-125|':>13}  {'max |M-125|':>13}  {'worst event':>22}")
    print("-" * 102)

    for s, label in [(1, "n=1 orphan siblings"),
                     (2, "n=2 orphan groups"),
                     (3, "n=3 orphan groups"),
                     (4, "n=4 orphan groups"),
                     (None, "NOT IDENTIFIED")]:
        sub = by_strategy.get(s, [])
        c = len(sub)
        frac = c / n_total if n_total else 0.0
        if sub and s is not None:
            diffs = np.array([r["abs_diff"] for r in sub])
            worst = sub[int(np.argmax(diffs))]
            worst_str = (f"file={worst['file_idx']},eid={worst['event_id']}"
                         f" Δ={worst['abs_diff']:.4f}")
            print(f"  {label:<30}  {c:>6d}  {frac:>6.1%}  "
                  f"{diffs.mean():>13.4f}  {diffs.max():>13.4f}  "
                  f"{worst_str:>22}")
        else:
            print(f"  {label:<30}  {c:>6d}  {frac:>6.1%}  "
                  f"{'—':>13}  {'—':>13}  {'—':>22}")

    # Per-event one-liner so user can see WHICH events failed
    failed = [r for r in results if r["strategy"] is None]
    if failed:
        print(f"\n  events where NO strategy matched ({len(failed)}):")
        for r in failed:
            print(f"    file_idx={r['file_idx']}  event_id={r['event_id']}")

    # Sum of |Δ| over all matched events combined
    matched = [r for r in results if r["strategy"] is not None]
    if matched:
        total_abs_diff = float(np.sum([r["abs_diff"] for r in matched]))
        print(f"\n  total |M-125| over {len(matched)} matched events: "
              f"{total_abs_diff:.4f} GeV  "
              f"(mean {total_abs_diff/len(matched):.4f}, tol={HIGGS_TOL})")


if __name__ == "__main__":
    main()
