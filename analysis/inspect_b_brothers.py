"""
Diagnostic: print the 'brother' structure of each leaf B-hadron (particles
sharing the same parent_id) for the first N H->bb~ events.

Mirrors the leaf-B selection used by inspect_hbb_top_bottoms.py
(b_plus_brothers_in_cone), but does NOT require a jet match — every leaf B
in the event is inspected.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/storage/agrp/barakma/PileupODD")

from load_higgs_diphoton_events import DEFAULT_COLUMNS, load_events  # noqa: E402
from classify_hf_decay_channels import B_HADRON_PDGS, DEFAULT_LABEL_DIR  # noqa: E402
from inspect_hbb_top_bottoms import load_bb_event_ids  # noqa: E402


# Minimal PDG-id -> name table. Anything not listed prints as pdg=<n>.
PDG_NAME = {
    11: "e-", -11: "e+",
    12: "nu_e", -12: "anti-nu_e",
    13: "mu-", -13: "mu+",
    14: "nu_mu", -14: "anti-nu_mu",
    15: "tau-", -15: "tau+",
    16: "nu_tau", -16: "anti-nu_tau",
    21: "g", 22: "gamma",
    111: "pi0", 211: "pi+", -211: "pi-",
    113: "rho0", 213: "rho+", -213: "rho-",
    221: "eta", 223: "omega", 331: "eta'", 333: "phi",
    130: "K0L", 310: "K0S",
    311: "K0", -311: "anti-K0",
    321: "K+", -321: "K-",
    313: "K*0", -313: "anti-K*0", 323: "K*+", -323: "K*-",
    411: "D+", -411: "D-",
    421: "D0", -421: "anti-D0",
    413: "D*+", -413: "D*-",
    423: "D*0", -423: "anti-D*0",
    431: "Ds+", -431: "Ds-",
    433: "Ds*+", -433: "Ds*-",
    511: "B0", -511: "anti-B0",
    513: "B*0", -513: "anti-B*0",
    521: "B+", -521: "B-",
    523: "B*+", -523: "B*-",
    531: "Bs0", -531: "anti-Bs0",
    533: "Bs*0", -533: "anti-Bs*0",
    541: "Bc+", -541: "Bc-",
    443: "J/psi", 553: "Upsilon",
    2212: "p", -2212: "anti-p",
    2112: "n", -2112: "anti-n",
    3122: "Lambda", -3122: "anti-Lambda",
    3222: "Sigma+", 3212: "Sigma0", 3112: "Sigma-",
    3322: "Xi0", 3312: "Xi-",
    3334: "Omega-",
    4122: "Lambda_c+", -4122: "anti-Lambda_c-",
    5122: "Lambda_b0", -5122: "anti-Lambda_b0",
}


def pdg_name(p: int) -> str:
    n = PDG_NAME.get(p)
    if n is not None:
        return n
    # b-mesons live in 5xx (and excited 5xxx); b-baryons live in 5xxx.
    a = abs(p)
    if a // 100 in {5}:
        return f"b-meson({p:+d})"
    if 5000 <= a < 6000:
        return f"b-baryon({p:+d})"
    return f"pdg={p:+d}"


def inspect_event(eid: int, fi: int, raw: dict) -> None:
    pdg         = np.asarray(raw["pdg_id"],         dtype=np.int64)
    vp          = np.asarray(raw["vertex_primary"], dtype=np.int64)
    particle_id = np.asarray(raw["particle_id"],    dtype=np.int64)
    parent_id   = np.asarray(raw["parent_id"],      dtype=np.int64)
    px = np.asarray(raw["px"], dtype=np.float64)
    py = np.asarray(raw["py"], dtype=np.float64)
    pz = np.asarray(raw["pz"], dtype=np.float64)
    E  = np.asarray(raw["energy"], dtype=np.float64)
    vx = np.asarray(raw["vx"], dtype=np.float64)
    vy = np.asarray(raw["vy"], dtype=np.float64)
    vz = np.asarray(raw["vz"], dtype=np.float64)
    pt  = np.hypot(px, py)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-30))
    phi = np.arctan2(py, px)

    # Same vp=1 B-hadron selection used by sum_leaf_b_hadrons_in_cone.
    b_mask = (vp == 1) & np.isin(np.abs(pdg), list(B_HADRON_PDGS))
    sel_idx = np.where(b_mask)[0]
    if sel_idx.size == 0:
        print(f"  (no vp=1 B-hadrons in event {eid})")
        return

    # Leaf cut: drop a B whose particle_id is the parent_id of another
    # selected B (its decay product is in the set, so summing both would
    # double-count the 4-momentum cascade).
    parents_of_other = {int(parent_id[j]) for j in sel_idx}
    leaf_idx = [int(i) for i in sel_idx
                if int(particle_id[i]) not in parents_of_other]
    if not leaf_idx:
        leaf_idx = [int(sel_idx[int(np.argmax(pt[sel_idx]))])]

    print(f"\n=== event_id={eid}  (file_idx={fi})  "
          f"N_leaf_B={len(leaf_idx)}  N_vp1_B={sel_idx.size}  "
          f"N_particles={pdg.size} ===")

    # Group leaf B's by parent_id; each unique parent contributes one
    # 'B + brothers' family.
    parent_to_leaves: dict[int, list[int]] = {}
    for i in leaf_idx:
        parent_to_leaves.setdefault(int(parent_id[i]), []).append(int(i))

    for k, (pid_par, leaves) in enumerate(parent_to_leaves.items(), 1):
        # Locate the parent particle to print its info (if it's in the record).
        parent_pos = np.where(particle_id == pid_par)[0]
        if parent_pos.size == 1:
            pi = int(parent_pos[0])
            par_str = (f"parent_id={pid_par}  pdg={int(pdg[pi]):+d} "
                       f"({pdg_name(int(pdg[pi]))})  vp={int(vp[pi])}  "
                       f"pT={pt[pi]:7.2f}  eta={eta[pi]:+6.2f}  "
                       f"phi={phi[pi]:+6.2f}  E={E[pi]:7.2f}")
        else:
            par_str = (f"parent_id={pid_par}  (not present in event record — "
                       f"likely outside the saved particle table)")

        print(f"\n  --- Family #{k}: {par_str}")

        # All daughters of this parent = leaf B + its brothers.
        sib_mask = parent_id == pid_par
        sib_idx_arr = np.where(sib_mask)[0]
        # Sort daughters by pT desc for readability.
        sib_idx_arr = sib_idx_arr[np.argsort(pt[sib_idx_arr])[::-1]]
        leaf_set = set(leaves)

        print(f"  Daughters of parent_id={pid_par}  (N={sib_idx_arr.size}):")
        print(f"    {'role':<6}  {'idx':>5}  {'pdg':>7}  {'name':<14}  "
              f"{'pT':>7}  {'eta':>6}  {'phi':>6}  {'E':>7}  "
              f"{'vx':>8}  {'vy':>8}  {'vz':>8}")
        for i in sib_idx_arr:
            i = int(i)
            role = "leaf-B" if i in leaf_set else "bro"
            print(f"    {role:<6}  {i:>5}  {int(pdg[i]):>+7d}  "
                  f"{pdg_name(int(pdg[i])):<14}  "
                  f"{pt[i]:>7.2f}  {eta[i]:>+6.2f}  {phi[i]:>+6.2f}  "
                  f"{E[i]:>7.2f}  "
                  f"{vx[i]:>8.2f}  {vy[i]:>8.2f}  {vz[i]:>8.2f}")

        # Family-sum invariant mass (the per-jet contribution before
        # cross-jet deduplication in the main analysis).
        pxs = float(px[sib_idx_arr].sum())
        pys = float(py[sib_idx_arr].sum())
        pzs = float(pz[sib_idx_arr].sum())
        Es  = float(E[sib_idx_arr].sum())
        m2 = Es * Es - (pxs * pxs + pys * pys + pzs * pzs)
        m_family = math.sqrt(max(m2, 0.0))
        print(f"  Family sum: pT={math.hypot(pxs, pys):7.2f}  "
              f"E={Es:7.2f}  invariant mass = {m_family:7.2f} GeV")

    # Event-level deduplicated total (the actual M(SigmaB+bro, ...) input).
    all_parents = list(parent_to_leaves.keys())
    union_mask = np.isin(parent_id, np.array(all_parents, dtype=np.int64))
    union_idx = np.where(union_mask)[0]
    pxs = float(px[union_idx].sum())
    pys = float(py[union_idx].sum())
    pzs = float(pz[union_idx].sum())
    Es  = float(E[union_idx].sum())
    m2 = Es * Es - (pxs * pxs + pys * pys + pzs * pzs)
    m_all = math.sqrt(max(m2, 0.0))
    print(f"\n  ALL B+brothers (deduplicated union over {len(all_parents)} parent_id(s), "
          f"{union_idx.size} particles): "
          f"E={Es:7.2f}  pT={math.hypot(pxs, pys):7.2f}  M={m_all:7.2f} GeV")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR),
                        help=f"Where the bb~ labels live (default: {DEFAULT_LABEL_DIR})")
    parser.add_argument("--n-events", type=int, default=20,
                        help="Number of H->bb~ events to inspect (default: 20)")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    file_to_events = load_bb_event_ids(label_dir, args.n_events)
    flat = sorted((fi, eid) for fi, eids in file_to_events.items() for eid in eids)
    print(f"selected {len(flat)} H->bb~ events from {label_dir}:")
    for fi, eid in flat:
        print(f"  file_idx={fi}  event_id={eid}")

    # Group events by file so we make exactly one load_events call per file.
    by_file: dict[int, list[int]] = {}
    for fi, eid in flat:
        by_file.setdefault(fi, []).append(eid)

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
            inspect_event(eid, fi, row)


if __name__ == "__main__":
    main()
