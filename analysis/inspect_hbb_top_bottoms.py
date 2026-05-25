"""
Pre-select 5 H -> bb̄ events from higgs_decays_enriched.parquet (which
adds truth-level out_pt/out_eta/out_phi for each Higgs daughter), fetch
their (particles, tracks, calo_hits) via load_higgs_diphoton_events.py,
run preprocess_for_model to get target_particles, cluster them into
anti-kt R=0.4 jets, and ΔR-match the truth b directions to the jets to
check pT / direction agreement.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

# Make sibling scripts and the primary preprocessing package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/storage/agrp/barakma/PileupODD")

from load_higgs_diphoton_events import (  # noqa: E402
    DECAYS_EVENTS_PER_RUN,
    HF_EVENTS_PER_FILE,
    NUM_HF_REPO_FILES,
    DEFAULT_COLUMNS,
    load_events,
)
from primary.create_trainning_dataset_pileup import preprocess_for_model  # noqa: E402

ENRICHED_PATH = "/storage/agrp/barakma/PileupODD/data/higgs_decays_enriched.parquet"
DR_MATCH = 0.4  # ΔR threshold for truth b → reconstructed jet match

# Jet clustering for b-jets: anti-kt R=0.4 is the LHC b-tagging standard.
# A wide-R kt jet absorbs ISR / UE / pileup remnants — the leading jet ends
# up at hundreds of GeV and the dijet mass overshoots 125. With anti-kt R=0.4
# each b is reconstructed as a narrow, well-defined jet.
JET_ALGO = "antikt"   # vs. "kt" for the reco_analysis.py default
JET_R = 0.4
MIN_CONSTITUENTS = 2
MIN_JET_PT = 25.0     # ATLAS/CMS b-jet baseline cut
JET_ETA_CUT = 2.5     # tracker acceptance for b-tagging


def select_hbb_truth(n: int) -> pl.DataFrame:
    """Return n H -> bb̄ rows from the enriched table with truth kinematics."""
    return (
        pl.read_parquet(ENRICHED_PATH)
        .filter(
            (pl.col("out_pids").list.len() == 2)
            & pl.col("out_pids").list.eval(pl.element().abs() == 5).list.all()
        )
        .with_columns(
            (pl.col("run") * DECAYS_EVENTS_PER_RUN + pl.col("event_id"))
            .alias("global_eid")
        )
        .filter(pl.col("global_eid") < NUM_HF_REPO_FILES * HF_EVENTS_PER_FILE)
        .sort("global_eid")
        .head(n)
    )


def _delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = eta1 - eta2
    dphi = phi1 - phi2
    while dphi > np.pi:  dphi -= 2 * np.pi
    while dphi < -np.pi: dphi += 2 * np.pi
    return float(np.hypot(deta, dphi))


def match_truth_to_jets(
    truth_eta: list[float], truth_phi: list[float],
    jets: list[dict],
    dr_cut: float = DR_MATCH,
) -> list[int | None]:
    """Greedy ΔR-min matching: each truth gets the closest unused jet
    within `dr_cut`. Returns list of jet indices (None if no match)."""
    n_t = len(truth_eta)
    pairs = []
    for ti in range(n_t):
        for ji, j in enumerate(jets):
            dr = _delta_r(truth_eta[ti], truth_phi[ti], j["eta"], j["phi"])
            if dr <= dr_cut:
                pairs.append((dr, ti, ji))
    pairs.sort()
    matched: list[int | None] = [None] * n_t
    used_jets: set[int] = set()
    for dr, ti, ji in pairs:
        if matched[ti] is None and ji not in used_jets:
            matched[ti] = ji
            used_jets.add(ji)
    return matched


def cluster_jets_event(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray,
                       algo: str = JET_ALGO,
                       jet_R: float = JET_R,
                       min_const: int = MIN_CONSTITUENTS,
                       min_pt: float = MIN_JET_PT,
                       eta_cut: float = JET_ETA_CUT) -> list[dict]:
    """FastJet jet clustering on one event's particles.

    For b-jet reconstruction we use anti-kt R=0.4 with |η| < 2.5 and a
    25-GeV cut, matching the LHC b-jet baseline. Jet info is copied into
    plain dicts while the ClusterSequence is alive so callers can read it
    after the CS goes out of scope.
    """
    import fastjet as fj

    if len(pt) < min_const:
        return []
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px * px + py * py + pz * pz)
    pj = [
        fj.PseudoJet(float(px[k]), float(py[k]), float(pz[k]), float(E[k]))
        for k in range(len(px))
    ]
    algo_id = fj.antikt_algorithm if algo == "antikt" else fj.kt_algorithm
    cs = fj.ClusterSequence(pj, fj.JetDefinition(algo_id, jet_R))

    records: list[dict] = []
    for j in fj.sorted_by_pt(cs.inclusive_jets()):
        consts = j.constituents()
        if (len(consts) < min_const
                or j.pt() <= min_pt
                or abs(j.eta()) >= eta_cut):
            continue
        records.append({
            "pt":  float(j.pt()),
            "eta": float(j.eta()),
            "phi": float(j.phi()),
            "m":   float(j.m()),
            "E":   float(j.E()),
            "px":  float(j.px()),
            "py":  float(j.py()),
            "pz":  float(j.pz()),
            "nconst": int(len(consts)),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=5,
                        help="How many H->bb events to inspect (default: 5)")
    parser.add_argument("--top-jets", type=int, default=2,
                        help="How many jets to print per event (default: 2)")
    parser.add_argument("--truth-pt-cut", type=float, default=1.0)
    parser.add_argument("--truth-eta-cut", type=float, default=3.0)
    parser.add_argument("--target-pt-cut", type=float, default=0.3)
    parser.add_argument("--clusters-cutoff", type=float, default=0.15)
    args = parser.parse_args()

    truth_df = select_hbb_truth(args.n_events)
    if truth_df.is_empty():
        print(f"no in-range H->bb̄ events in {ENRICHED_PATH}")
        return

    file_to_events: dict[int, list[int]] = defaultdict(list)
    truth_by_eid: dict[int, dict] = {}
    for row in truth_df.iter_rows(named=True):
        eid = int(row["global_eid"])
        file_to_events[eid // HF_EVENTS_PER_FILE].append(eid)
        truth_by_eid[eid] = row

    flat = sorted((fi, eid) for fi, eids in file_to_events.items() for eid in eids)
    print(f"selected {len(flat)} H -> bb̄ events from {ENRICHED_PATH}:")
    for fi, eid in flat:
        t = truth_by_eid[eid]
        print(f"  file {fi:>4}  event {eid:>5}   "
              f"out_pids={t['out_pids']}   "
              f"pT={[f'{p:.1f}' for p in t['out_pt']]}")
    print()

    # Same loader the diphoton script uses (per-file pl.scan_parquet + is_in)
    particles = load_events(file_to_events, kind="particles",
                            columns=DEFAULT_COLUMNS["particles"])
    tracks = load_events(file_to_events, kind="tracks",
                         columns=DEFAULT_COLUMNS["tracks"])
    calo_hits = load_events(file_to_events, kind="calo_hits",
                            columns=DEFAULT_COLUMNS["calo_hits"])
    print(f"\nloaded particles={particles.shape}, "
          f"tracks={tracks.shape}, calo_hits={calo_hits.shape}\n")

    print("running preprocess_for_model from create_trainning_dataset_pileup.py …")
    out = preprocess_for_model(
        particles=particles, tracks=tracks, calo_hits=calo_hits,
        num_of_events=-1,
        truth_pt_cut=args.truth_pt_cut,
        truth_eta_cut=args.truth_eta_cut,
        target_pt_cut=args.target_pt_cut,
        clusters_cutoff=args.clusters_cutoff,
    )
    target_particles = out["target_particles"]
    print(f"\ntarget_particles: {target_particles.shape}")
    print(f"columns: {target_particles.columns}\n")

    def _wrap_phi(p: float) -> float:
        while p >  np.pi: p -= 2 * np.pi
        while p < -np.pi: p += 2 * np.pi
        return float(p)

    for row in target_particles.iter_rows(named=True):
        eid = int(row["event_id"])
        if eid not in truth_by_eid:
            continue
        truth = truth_by_eid[eid]
        truth_pids = truth["out_pids"]
        truth_pt = list(truth["out_pt"])
        truth_eta = list(truth["out_eta"])
        truth_phi = list(truth["out_phi"])

        pt = np.asarray(row["pt"], dtype=np.float64)
        eta = np.asarray(row["eta"], dtype=np.float64)
        phi = np.asarray(row["phi"], dtype=np.float64)
        n_const = len(pt)

        print(f"=== event {eid} ===")
        print(f"  {n_const} target particles  (target pT range "
              f"{pt.min():.2f}-{pt.max():.2f})")
        print("  truth Higgs daughters:")
        for k, (pid, p_t, p_eta, p_phi) in enumerate(
                zip(truth_pids, truth_pt, truth_eta, truth_phi), 1):
            print(f"    #{k}  pdg={pid:+d}  pT={p_t:7.2f}  "
                  f"η={p_eta:+.2f}  φ={p_phi:+.2f}")

        jets = cluster_jets_event(pt, eta, phi)
        # normalize jet φ to [-π, π] for comparison with truth
        for j in jets:
            j["phi"] = _wrap_phi(j["phi"])
        print(f"  found {len(jets)} jets ({JET_ALGO}, R={JET_R}, "
              f"min_const={MIN_CONSTITUENTS}, min_pt={MIN_JET_PT}, "
              f"|η|<{JET_ETA_CUT})")
        for k, j in enumerate(jets[:args.top_jets], 1):
            print(f"    j#{k}  pT={j['pt']:7.2f}  η={j['eta']:+.2f}  "
                  f"φ={j['phi']:+.2f}  m={j['m']:6.2f}  nconst={j['nconst']}")

        matches = match_truth_to_jets(truth_eta, truth_phi, jets, DR_MATCH)
        print(f"  truth → jet matching (ΔR < {DR_MATCH}):")
        for k, (pid, p_t, p_eta, p_phi, ji) in enumerate(
                zip(truth_pids, truth_pt, truth_eta, truth_phi, matches), 1):
            label = f"truth#{k} pdg={pid:+d}  pT_truth={p_t:6.2f}"
            if ji is None:
                print(f"    {label}   →  NO MATCH within ΔR<{DR_MATCH}")
                continue
            j = jets[ji]
            dr = _delta_r(p_eta, p_phi, j["eta"], j["phi"])
            d_pt = j["pt"] - p_t
            rel = d_pt / p_t if p_t > 0 else 0.0
            print(f"    {label}   →  j#{ji+1}  pT_jet={j['pt']:6.2f}  "
                  f"ΔR={dr:.3f}  ΔpT={d_pt:+6.2f} ({rel:+.1%})  "
                  f"nconst={j['nconst']}")

        # Truth-matched dijet mass (only if both b's matched)
        if all(m is not None for m in matches) and len(matches) == 2:
            j1, j2 = jets[matches[0]], jets[matches[1]]
            m2 = ((j1["E"] + j2["E"]) ** 2
                  - (j1["px"] + j2["px"]) ** 2
                  - (j1["py"] + j2["py"]) ** 2
                  - (j1["pz"] + j2["pz"]) ** 2)
            mjj = float(np.sqrt(max(m2, 0.0)))
            print(f"  truth-matched dijet mass: M(jb, jb̄) = {mjj:.2f} GeV "
                  f"(target ~125 GeV)")
        print()


if __name__ == "__main__":
    main()
