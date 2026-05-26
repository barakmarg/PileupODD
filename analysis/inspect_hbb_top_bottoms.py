"""
Pipeline (truth-level, no detector, no external PU contamination):

1. Read the per-file labels parquets written by classify_hf_decay_channels.py
   (/storage/agrp/barakma/PileupODD/data/hf_decay_labels/labels_file_*.parquet)
   and pick the first N events tagged H→bb̄. Each label row is
   (file_idx, event_id, channel).
2. Download ONLY those (file_idx, event_id) events from HF (particles,
   tracks, calo_hits) using load_events with predicate pushdown.
3. Run preprocess_for_model from create_trainning_dataset_pileup.py to
   build target_particles.
4. Cluster target_particles into anti-kt R=0.4 jets per event (FastJet).
5. For each event, find all B-hadrons at vp=1 from the RAW HF record. 
   Pair the two whose invariant mass is closest to 125 GeV (the Higgs mass).
   These are the truth directions of the two signal b quarks.
6. Match each chosen B-hadron to its closest jet within ΔR<0.4 → b-jets.
7. Compute the dijet invariant mass.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/storage/agrp/barakma/PileupODD")

from load_higgs_diphoton_events import (  # noqa: E402
    DEFAULT_COLUMNS,
    load_events,
)
from classify_hf_decay_channels import (  # noqa: E402
    B_HADRON_PDGS, DEFAULT_LABEL_DIR, NU_PDGS,
)
from primary.create_trainning_dataset_pileup import preprocess_for_model  # noqa: E402

JET_ALGO = "antikt"
JET_R = 0.4
MIN_CONSTITUENTS = 2
MIN_JET_PT = 10.0
JET_ETA_CUT = 4.0  # Kept exactly as original
DR_MATCH = 0.4
B_HAD_MIN_PT = 5.0  


def cluster_jets_event(pt, eta, phi,
                       algo=JET_ALGO, jet_R=JET_R,
                       min_const=MIN_CONSTITUENTS, min_pt=MIN_JET_PT,
                       eta_cut=JET_ETA_CUT) -> list[dict]:
    """FastJet jet clustering on one event's particles."""
    import fastjet as fj
    if len(pt) < min_const:
        return []
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    E = np.sqrt(px * px + py * py + pz * pz)
    pj = [fj.PseudoJet(float(px[k]), float(py[k]), float(pz[k]), float(E[k]))
          for k in range(len(px))]
    algo_id = fj.antikt_algorithm if algo == "antikt" else fj.kt_algorithm
    cs = fj.ClusterSequence(pj, fj.JetDefinition(algo_id, jet_R))
    records = []
    for j in fj.sorted_by_pt(cs.inclusive_jets()):
        consts = j.constituents()
        if len(consts) < min_const or j.pt() <= min_pt or abs(j.eta()) >= eta_cut:
            continue
        records.append({
            "pt": float(j.pt()),  "eta": float(j.eta()), "phi": float(j.phi()),
            "m":  float(j.m()),   "E":   float(j.E()),
            "px": float(j.px()),  "py":  float(j.py()),  "pz": float(j.pz()),
            "nconst": int(len(consts)),
        })
    return records


def _wrap_phi(p):
    while p >  math.pi: p -= 2 * math.pi
    while p < -math.pi: p += 2 * math.pi
    return float(p)


def _delta_r(eta1, phi1, eta2, phi2):
    dphi = _wrap_phi(phi1 - phi2)
    return float(math.hypot(eta1 - eta2, dphi))


def _mass_two_dict(p1: dict, p2: dict) -> float:
    """Helper to calculate invariant mass of two particles stored as dicts."""
    px1 = p1["pt"] * math.cos(p1["phi"])
    py1 = p1["pt"] * math.sin(p1["phi"])
    pz1 = p1["pt"] * math.sinh(p1["eta"])
    
    px2 = p2["pt"] * math.cos(p2["phi"])
    py2 = p2["pt"] * math.sin(p2["phi"])
    pz2 = p2["pt"] * math.sinh(p2["eta"])
    
    e_tot = p1["E"] + p2["E"]
    px_tot = px1 + px2
    py_tot = py1 + py2
    pz_tot = pz1 + pz2
    
    m2 = e_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
    return float(math.sqrt(max(m2, 0.0)))


def load_bb_event_ids(label_dir: Path, n_keep: int) -> dict[int, list[int]]:
    """Read all labels_file_*.parquet, pick the first n_keep H→bb̄ events."""
    paths = sorted(label_dir.glob("labels_file_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no label parquets in {label_dir}. Run "
            f"classify_hf_decay_channels.py first."
        )
    df = (
        pl.concat([pl.read_parquet(p) for p in paths])
        .filter(pl.col("channel") == "bb̄")
        .sort("file_idx", "event_id")
        .head(n_keep)
    )
    out: dict[int, list[int]] = defaultdict(list)
    for fi, eid, _ in df.iter_rows():
        out[int(fi)].append(int(eid))
    return dict(out)


def get_all_b_hadrons(raw_row: dict, cluster_dR: float = 0.4) -> list[dict]:
    """From a raw HF particle row, return ALL distinct B-hadrons at vp=1."""
    pdg = np.asarray(raw_row["pdg_id"])
    vp  = np.asarray(raw_row["vertex_primary"])
    px  = np.asarray(raw_row["px"]); py = np.asarray(raw_row["py"]); pz = np.asarray(raw_row["pz"])
    E   = np.asarray(raw_row["energy"])
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-30))
    phi = np.arctan2(py, px)

    b_mask = np.isin(np.abs(pdg), list(B_HADRON_PDGS)) & (vp == 1) & (pt > B_HAD_MIN_PT)
    idx = np.where(b_mask)[0]
    if len(idx) == 0:
        return []

    # Walk B-hadrons in descending pT; keep each only if it's ΔR > cluster_dR
    # away from every already-chosen cluster lead.
    order = idx[np.argsort(pt[idx])[::-1]]
    leads: list[int] = []
    for k in order:
        is_new_cluster = True
        for j in leads:
            if _delta_r(eta[k], phi[k], eta[j], phi[j]) < cluster_dR:
                is_new_cluster = False
                break
        if is_new_cluster:
            leads.append(int(k))

    return [
        {"pdg": int(pdg[k]), "pt": float(pt[k]),
         "eta": float(eta[k]), "phi": float(phi[k]),
         "E": float(E[k])}
        for k in leads
    ]


def get_neutrinos_vp1(raw_row: dict) -> dict:
    """Pull all vp==1 neutrinos as a vectorized record (px,py,pz,E,eta,phi,pt)
    so we can sum their 4-momentum within any cone direction."""
    pdg = np.asarray(raw_row["pdg_id"])
    vp  = np.asarray(raw_row["vertex_primary"])
    px  = np.asarray(raw_row["px"])
    py  = np.asarray(raw_row["py"])
    pz  = np.asarray(raw_row["pz"])
    E   = np.asarray(raw_row["energy"])
    mask = (vp == 1) & np.isin(np.abs(pdg), list(NU_PDGS))
    if not mask.any():
        return {"n": 0, "px": np.array([]), "py": np.array([]), "pz": np.array([]),
                "E": np.array([]), "eta": np.array([]), "phi": np.array([])}
    px = px[mask]; py = py[mask]; pz = pz[mask]; E = E[mask]
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-30))
    phi = np.arctan2(py, px)
    return {"n": int(mask.sum()), "px": px, "py": py, "pz": pz, "E": E,
            "eta": eta, "phi": phi, "pt": pt}


def sum_neutrinos_in_cone(nus: dict, eta0: float, phi0: float,
                          dR: float = JET_R) -> dict:
    """Sum the 4-momentum of vp=1 neutrinos within ΔR<dR of (eta0, phi0)."""
    if nus["n"] == 0:
        return {"n": 0, "px": 0.0, "py": 0.0, "pz": 0.0, "E": 0.0, "pt": 0.0}
    dphi = nus["phi"] - phi0
    dphi = np.where(dphi >  math.pi, dphi - 2 * math.pi, dphi)
    dphi = np.where(dphi < -math.pi, dphi + 2 * math.pi, dphi)
    in_cone = (nus["eta"] - eta0) ** 2 + dphi ** 2 < dR ** 2
    if not in_cone.any():
        return {"n": 0, "px": 0.0, "py": 0.0, "pz": 0.0, "E": 0.0, "pt": 0.0}
    px_s = float(nus["px"][in_cone].sum())
    py_s = float(nus["py"][in_cone].sum())
    pz_s = float(nus["pz"][in_cone].sum())
    E_s  = float(nus["E"][in_cone].sum())
    return {"n": int(in_cone.sum()), "px": px_s, "py": py_s, "pz": pz_s,
            "E": E_s, "pt": math.hypot(px_s, py_s)}


def match_b_to_jets(bhads: list[dict], jets: list[dict],
                    dr_cut: float = DR_MATCH) -> list[int | None]:
    """Greedy ΔR-min: each B-hadron grabs the closest unused jet within dr_cut."""
    pairs = []
    for bi, b in enumerate(bhads):
        for ji, j in enumerate(jets):
            dr = _delta_r(b["eta"], b["phi"], j["eta"], j["phi"])
            if dr <= dr_cut:
                pairs.append((dr, bi, ji))
    pairs.sort()
    matched: list[int | None] = [None] * len(bhads)
    used: set[int] = set()
    for dr, bi, ji in pairs:
        if matched[bi] is None and ji not in used:
            matched[bi] = ji
            used.add(ji)
    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR),
                        help=f"Where the bb̄ labels live (default: {DEFAULT_LABEL_DIR})")
    parser.add_argument("--n-bb-events", type=int, default=50,
                        help="How many bb̄ events to analyse (default: 5)")
    
    # CUTS KEPT EXACTLY AS ORIGINAL
    parser.add_argument("--truth-pt-cut", type=float, default=1.0)
    parser.add_argument("--truth-eta-cut", type=float, default=3.0)
    parser.add_argument("--target-pt-cut", type=float, default=0.3)
    parser.add_argument("--clusters-cutoff", type=float, default=0.15)
    args = parser.parse_args()

    file_to_events = load_bb_event_ids(Path(args.label_dir), args.n_bb_events)
    flat = sorted((fi, eid) for fi, eids in file_to_events.items() for eid in eids)
    print(f"selected {len(flat)} H→bb̄ events from {args.label_dir}:")

    print(f"\ndownloading particles + tracks + calo_hits for "
          f"{len(flat)} events across {len(file_to_events)} HF files …")
    particles_bb = load_events(file_to_events, kind="particles",
                               columns=DEFAULT_COLUMNS["particles"])
    tracks_bb    = load_events(file_to_events, kind="tracks",
                               columns=DEFAULT_COLUMNS["tracks"])
    calo_hits_bb = load_events(file_to_events, kind="calo_hits",
                               columns=DEFAULT_COLUMNS["calo_hits"])

    print("\nrunning preprocess_for_model …")
    out = preprocess_for_model(
        particles=particles_bb, tracks=tracks_bb, calo_hits=calo_hits_bb,
        num_of_events=-1,
        truth_pt_cut=args.truth_pt_cut, truth_eta_cut=args.truth_eta_cut,
        target_pt_cut=args.target_pt_cut, clusters_cutoff=args.clusters_cutoff,
    )
    target_particles = out["target_particles"]
    raw_by_eid = {int(r["event_id"]): r
                  for r in particles_bb.iter_rows(named=True)}

    masses_vis: list[float] = []
    masses_corr: list[float] = []
    for tp in target_particles.iter_rows(named=True):
        eid = int(tp["event_id"])
        pt  = np.asarray(tp["pt"],  dtype=np.float64)
        eta = np.asarray(tp["eta"], dtype=np.float64)
        phi = np.asarray(tp["phi"], dtype=np.float64)

        # 1. Get all distinct B-Hadron directions
        bhads_all = get_all_b_hadrons(raw_by_eid[eid])
        
        # 2. Select the pair of B-hadrons that most likely came from the Higgs 
        # (Mass closest to 125 GeV).
        bhads = []
        best_mass = 0.0
        if len(bhads_all) >= 2:
            best_diff = float('inf')
            for i in range(len(bhads_all)):
                for j in range(i + 1, len(bhads_all)):
                    m = _mass_two_dict(bhads_all[i], bhads_all[j])
                    if abs(m - 125.0) < best_diff:
                        best_diff = abs(m - 125.0)
                        bhads = [bhads_all[i], bhads_all[j]]
                        best_mass = m
        else:
            bhads = bhads_all

        print(f"\n=== event {eid} ===")
        print(f"  {len(pt)} target particles  (target pT range {pt.min():.2f}-{pt.max():.2f})")
        print(f"  found {len(bhads_all)} B-hadron candidates.")
        if len(bhads) == 2:
            print(f"  Selected pair closest to Higgs mass (M = {best_mass:.2f} GeV):")
        
        for k, b in enumerate(bhads, 1):
            print(f"    bH#{k}  pdg={b['pdg']:+5d}  pT={b['pt']:7.2f}  "
                  f"η={b['eta']:+.2f}  φ={b['phi']:+.2f}  E={b['E']:7.2f}")

        jets = cluster_jets_event(pt, eta, phi)
        for j in jets:
            j["phi"] = _wrap_phi(j["phi"])
            
        print(f"  {len(jets)} jets ({JET_ALGO}, R={JET_R}, min_pt={MIN_JET_PT}, |η|<{JET_ETA_CUT})")
        for k, j in enumerate(jets[:4], 1):
            print(f"    j#{k}  pT={j['pt']:7.2f}  η={j['eta']:+.2f}  "
                  f"φ={j['phi']:+.2f}  m={j['m']:6.2f}  nconst={j['nconst']}")

        nus = get_neutrinos_vp1(raw_by_eid[eid])
        matches = match_b_to_jets(bhads, jets, dr_cut=DR_MATCH)
        print(f"  ghost-association (ΔR<{DR_MATCH}), with in-cone ν sums:")

        nu_sums: list[dict] = []   # in-cone ν sum per matched jet (or zero)
        for k, (b, ji) in enumerate(zip(bhads, matches), 1):
            if ji is None:
                print(f"    bH#{k} → NO MATCHED JET within ΔR<{DR_MATCH}")
                nu_sums.append({"n": 0, "px": 0.0, "py": 0.0, "pz": 0.0,
                                "E": 0.0, "pt": 0.0})
                continue
            j = jets[ji]
            dr = _delta_r(b["eta"], b["phi"], j["eta"], j["phi"])
            d_pt = j["pt"] - b["pt"]
            rel = d_pt / b["pt"] if b["pt"] > 0 else 0.0
            nu = sum_neutrinos_in_cone(nus, j["eta"], j["phi"], dR=JET_R)
            nu_sums.append(nu)
            nu_frac = nu["pt"] / j["pt"] if j["pt"] > 0 else 0.0
            print(f"    bH#{k} → j#{ji+1}  pT_jet={j['pt']:7.2f}  "
                  f"ΔR={dr:.3f}  ΔpT={d_pt:+6.2f} ({rel:+.0%})  "
                  f"in-cone ν: n={nu['n']}  pT_ν={nu['pt']:6.2f}  "
                  f"E_ν={nu['E']:6.2f}  (ν/jet pT = {nu_frac:+.0%})")

        if (len(matches) == 2 and all(m is not None for m in matches)
                and matches[0] != matches[1]):
            j1, j2 = jets[matches[0]], jets[matches[1]]
            # Visible jet-only dijet mass
            m2_vis = ((j1["E"] + j2["E"]) ** 2
                      - (j1["px"] + j2["px"]) ** 2
                      - (j1["py"] + j2["py"]) ** 2
                      - (j1["pz"] + j2["pz"]) ** 2)
            mjj_vis = float(math.sqrt(max(m2_vis, 0.0)))
            masses_vis.append(mjj_vis)
            # ν-corrected: add the in-cone ν 4-momentum of each matched jet
            nu1, nu2 = nu_sums[0], nu_sums[1]
            E_c = j1["E"] + j2["E"] + nu1["E"] + nu2["E"]
            px_c = j1["px"] + j2["px"] + nu1["px"] + nu2["px"]
            py_c = j1["py"] + j2["py"] + nu1["py"] + nu2["py"]
            pz_c = j1["pz"] + j2["pz"] + nu1["pz"] + nu2["pz"]
            mjj_corr = float(math.sqrt(max(E_c * E_c - (px_c * px_c + py_c * py_c + pz_c * pz_c), 0.0)))
            masses_corr.append(mjj_corr)
            print(f"  → visible dijet mass        M(bj,b̄j)     = {mjj_vis:7.2f} GeV")
            print(f"  → ν-corrected dijet mass    M(bj+ν,b̄j+ν) = {mjj_corr:7.2f} GeV  "
                  f"(in-cone ν pT: {nu1['pt']:.2f} + {nu2['pt']:.2f})")
        else:
            print("  → could not form b-tagged dijet (one or both b-jets missing)")

    if masses_vis:
        vis = np.array(masses_vis)
        cor = np.array(masses_corr)
        print(f"\n=== summary across {len(vis)} reconstructed events ===")
        print(f"{'':<20}  {'visible':>10}  {'ν-corrected':>12}")
        print("-" * 48)
        print(f"  {'mean M(bb̄)':<20}  {vis.mean():>10.2f}  {cor.mean():>12.2f}")
        print(f"  {'median':<20}  {float(np.median(vis)):>10.2f}  "
              f"{float(np.median(cor)):>12.2f}")
        print(f"  {'std':<20}  {vis.std():>10.2f}  {cor.std():>12.2f}")
        print(f"  {'min':<20}  {vis.min():>10.2f}  {cor.min():>12.2f}")
        print(f"  {'max':<20}  {vis.max():>10.2f}  {cor.max():>12.2f}")
        within_5  = (np.abs(cor - 125) < 5).sum()
        within_10 = (np.abs(cor - 125) < 10).sum()
        within_20 = (np.abs(cor - 125) < 20).sum()
        print(f"\n  ν-corrected within ±5  GeV of 125: "
              f"{within_5}/{len(cor)} ({within_5/len(cor):.0%})")
        print(f"  ν-corrected within ±10 GeV of 125: "
              f"{within_10}/{len(cor)} ({within_10/len(cor):.0%})")
        print(f"  ν-corrected within ±20 GeV of 125: "
              f"{within_20}/{len(cor)} ({within_20/len(cor):.0%})")

if __name__ == "__main__":
    main()