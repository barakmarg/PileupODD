"""
Reconstruct M_γγ for the saved H -> γγ events.

Reads the per-batch particle parquets at
/storage/agrp/barakma/PileupODD/data/ggf_pu200_higgs_yy/particles/*.parquet,
for each event picks the two highest-pT primary photons (pdg_id == 22,
vertex_primary == 1), and computes the invariant mass

    M_γγ = sqrt( (E1+E2)^2 - |p1+p2|^2 )

For a real H -> γγ event the two Higgs daughters carry ~60 GeV pT each
and dominate the photon pT spectrum, so top-2-by-pT should give the
right pair. The peak of M_γγ across events should land at ~125 GeV.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_DIR = "/storage/agrp/barakma/PileupODD/data/ggf_pu200_higgs_yy/particles"
HIGGS_MASS = 125.10  # GeV (PDG)

# EM candidates: photons + e± (catches Higgs photons that converted to e+e- in
# the tracker — without this we miss events where the Higgs daughter γ is no
# longer the leading EM object at vp=1).
EM_PDGS = (22, 11, -11)

# Standard diphoton-style selection cuts
ETA_CUT = 2.5      # |η| < ETA_CUT for both candidates
DR_CUT = 0.4       # ΔR(c1, c2) > DR_CUT — kills collimated π0/conversion pairs
MIN_PT = 20.0      # GeV — Higgs daughters carry tens of GeV

# EM supercluster ΔR: merge nearby γ/e fragments (conversions split one γ
# into e+e- collinear, brem leaves photons very close to electrons). 0.1
# is small enough to stay local but big enough to absorb conversion debris.
CLUSTER_DR = 0.1


def _pair_invariant_mass(E, px, py, pz, i, j):
    Es = E[i] + E[j]
    pxs = px[i] + px[j]
    pys = py[i] + py[j]
    pzs = pz[i] + pz[j]
    m2 = Es * Es - (pxs * pxs + pys * pys + pzs * pzs)
    return float(np.sqrt(max(m2, 0.0)))


def _build_em_superclusters(pt, eta, phi, energy, px, py, pz):
    """Greedy ΔR < CLUSTER_DR clustering of EM candidates, leading-pT first.

    Returns arrays of cluster-level (pT, eta, phi, E, px, py, pz, n_seeds).
    A "cluster" sums 4-momenta of all members. Conversion debris (e+e-
    from a single γ) and brem clouds get absorbed into one cluster.
    """
    order = np.argsort(pt)[::-1]
    used = np.zeros(len(pt), dtype=bool)
    cl_E, cl_px, cl_py, cl_pz, cl_n = [], [], [], [], []
    for k in order:
        if used[k]:
            continue
        members = [k]
        used[k] = True
        # absorb later (lower-pT) candidates within ΔR of this seed
        for m in order:
            if used[m]:
                continue
            dphi = phi[m] - phi[k]
            if dphi >  np.pi: dphi -= 2 * np.pi
            if dphi < -np.pi: dphi += 2 * np.pi
            if (eta[m] - eta[k]) ** 2 + dphi ** 2 < CLUSTER_DR ** 2:
                members.append(m)
                used[m] = True
        cl_E.append(energy[members].sum())
        cl_px.append(px[members].sum())
        cl_py.append(py[members].sum())
        cl_pz.append(pz[members].sum())
        cl_n.append(len(members))
    E = np.array(cl_E)
    PX = np.array(cl_px)
    PY = np.array(cl_py)
    PZ = np.array(cl_pz)
    PT = np.hypot(PX, PY)
    ETA = np.arcsinh(PZ / np.maximum(PT, 1e-9))
    PHI = np.arctan2(PY, PX)
    return PT, ETA, PHI, E, PX, PY, PZ, np.array(cl_n)


def reconstruct_one(row: dict) -> dict:
    """Pick the highest-pT EM-supercluster pair satisfying central+ΔR+pT.

    Always returns a dict; "skipped" events have skip_reason set.
    """
    base = {"event_id": int(row["event_id"]), "skip_reason": None, "M_yy": None}

    pdg = np.asarray(row["pdg_id"])
    vp = np.asarray(row["vertex_primary"])
    em_mask = np.isin(pdg, EM_PDGS) & (vp == 1)
    n_em = int(em_mask.sum())
    base["n_em_primary"] = n_em
    if n_em < 2:
        base["skip_reason"] = "n_em<2"
        return base

    energy = np.asarray(row["energy"])[em_mask]
    px = np.asarray(row["px"])[em_mask]
    py = np.asarray(row["py"])[em_mask]
    pz = np.asarray(row["pz"])[em_mask]
    pt0 = np.hypot(px, py)
    eta0 = np.arcsinh(pz / np.maximum(pt0, 1e-9))
    phi0 = np.arctan2(py, px)

    PT, ETA, PHI, E, PX, PY, PZ, N_SEEDS = _build_em_superclusters(
        pt0, eta0, phi0, energy, px, py, pz,
    )
    base["n_clusters"] = int(len(PT))

    good = (np.abs(ETA) < ETA_CUT) & (PT > MIN_PT)
    base["n_after_cuts"] = int(good.sum())
    base["leading_pt_in_cuts"] = float(PT[good].max()) if good.any() else 0.0
    if good.sum() < 2:
        base["skip_reason"] = "n_clusters_after_pt_eta<2"
        return base

    g_idx = np.where(good)[0]
    g_idx = g_idx[np.argsort(PT[g_idx])[::-1]]
    i = g_idx[0]
    j = None
    for k in g_idx[1:]:
        deta = ETA[k] - ETA[i]
        dphi = PHI[k] - PHI[i]
        if dphi >  np.pi: dphi -= 2 * np.pi
        if dphi < -np.pi: dphi += 2 * np.pi
        if np.hypot(deta, dphi) > DR_CUT:
            j = int(k)
            break
    if j is None:
        base["skip_reason"] = "no_dR_separated_pair"
        return base

    mass = _pair_invariant_mass(E, PX, PY, PZ, i, j)
    deta = float(ETA[j] - ETA[i])
    dphi = float(PHI[j] - PHI[i])
    if dphi >  np.pi: dphi -= 2 * np.pi
    if dphi < -np.pi: dphi += 2 * np.pi

    base.update({
        "M_yy": mass,
        "pt1": float(PT[i]), "pt2": float(PT[j]),
        "E1":  float(E[i]),  "E2":  float(E[j]),
        "eta1": float(ETA[i]), "eta2": float(ETA[j]),
        "dphi": dphi, "dR": float(np.hypot(deta, dphi)),
        "n_seeds1": int(N_SEEDS[i]), "n_seeds2": int(N_SEEDS[j]),
    })
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help=f"Particles parquet dir (default: {DEFAULT_DIR})")
    parser.add_argument("--limit", type=int, default=200,
                        help="Process only the first N events (for testing)")
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob("batch_*.parquet"))
    if not files:
        raise SystemExit(f"no batch parquets found in {args.dir}")
    print(f"reading {len(files)} batch files from {args.dir}")

    df = pl.concat([pl.read_parquet(f) for f in files])
    print(f"total events: {df.height}")
    if args.limit is not None:
        df = df.head(args.limit)
        print(f"limited to first {df.height}")

    results = []
    skips: dict[str, int] = {}
    for row in df.iter_rows(named=True):
        rec = reconstruct_one(row)
        if rec["skip_reason"] is not None:
            skips[rec["skip_reason"]] = skips.get(rec["skip_reason"], 0) + 1
            print(
                f"event {rec['event_id']:>6}: SKIP ({rec['skip_reason']})   "
                f"n_EM={rec['n_em_primary']}   "
                f"n_clusters={rec.get('n_clusters', 0)}   "
                f"n_after_cuts={rec.get('n_after_cuts', 0)}   "
                f"leading_pT_in_cuts={rec.get('leading_pt_in_cuts', 0):.1f}"
            )
            continue
        results.append(rec)
        print(
            f"event {rec['event_id']:>6}: M = {rec['M_yy']:7.2f} GeV   "
            f"pT=({rec['pt1']:6.1f}, {rec['pt2']:6.1f})   "
            f"η=({rec['eta1']:+.2f}, {rec['eta2']:+.2f})   "
            f"Δφ={rec['dphi']:+.2f}   ΔR={rec['dR']:.2f}   "
            f"seeds=({rec['n_seeds1']}+{rec['n_seeds2']})   "
            f"n_EM={rec['n_em_primary']}"
        )

    print("\n=== skip reasons ===")
    for reason, n in sorted(skips.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {n}")

    if not results:
        return
    masses = np.array([r["M_yy"] for r in results])
    print("\n=== M_γγ summary ===")
    print(f"  events reconstructed: {len(masses)}")
    print(f"  mean   : {masses.mean():7.2f} GeV   (target {HIGGS_MASS:.2f})")
    print(f"  median : {np.median(masses):7.2f} GeV")
    print(f"  std    : {masses.std():7.2f} GeV")
    print(f"  min/max: {masses.min():.2f} / {masses.max():.2f} GeV")
    within_5 = (np.abs(masses - HIGGS_MASS) < 5).sum()
    within_10 = (np.abs(masses - HIGGS_MASS) < 10).sum()
    print(f"  within ±5  GeV of 125: {within_5}/{len(masses)} "
          f"({within_5/len(masses):.0%})")
    print(f"  within ±10 GeV of 125: {within_10}/{len(masses)} "
          f"({within_10/len(masses):.0%})")


if __name__ == "__main__":
    main()
