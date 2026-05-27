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
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

# Sentinel the worker prints on its last line so the manager can pick up
# the per-chunk mass arrays from the subprocess stdout.
RESULT_SENTINEL = "RESULT::"

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
B_HAD_ETA_CUT = 3.5            # drop forward B-hadrons (beam remnants / ISR fakes)
B_HAD_PAIR_DR_MIN = 0.4        # require the chosen pair be ΔR-separated
HIGGS_MASS = 125.0             # target for the closest-mass pair selection


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

    # vp=1, pT>min, AND central (|η|<B_HAD_ETA_CUT) — forward B-hadrons at
    # |η|>3.5 are essentially never Higgs daughters; they are beam-remnant
    # / ISR fakes that carry the entire ~TeV proton momentum forward.
    b_mask = (
        np.isin(np.abs(pdg), list(B_HADRON_PDGS))
        & (vp == 1)
        & (pt > B_HAD_MIN_PT)
        & (np.abs(eta) < B_HAD_ETA_CUT)
    )
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


def sum_leaf_b_hadrons_in_cone(raw_row: dict, eta0: float, phi0: float,
                               dR: float = JET_R) -> dict:
    """Pick all vp=1 B-hadrons within ΔR<dR of (eta0, phi0), then drop any
    that are the parent (via parent_id) of another in-cone B-hadron. The
    survivors are the LEAF B-hadrons of the b-quark fragmentation cascade
    inside this jet — summing them never double-counts (a B* whose decay
    product is the B in the same set has its 4-momentum already redirected
    to the daughter B by 4-momentum conservation)."""
    pdg = np.asarray(raw_row["pdg_id"], dtype=np.int64)
    vp  = np.asarray(raw_row["vertex_primary"], dtype=np.int64)
    particle_id = np.asarray(raw_row["particle_id"], dtype=np.int64)
    parent_id   = np.asarray(raw_row["parent_id"],   dtype=np.int64)
    px  = np.asarray(raw_row["px"], dtype=np.float64)
    py  = np.asarray(raw_row["py"], dtype=np.float64)
    pz  = np.asarray(raw_row["pz"], dtype=np.float64)
    E   = np.asarray(raw_row["energy"], dtype=np.float64)
    pt = np.hypot(px, py)
    eta_all = np.arcsinh(pz / np.maximum(pt, 1e-30))
    phi_all = np.arctan2(py, px)

    b_mask = (vp == 1) & np.isin(np.abs(pdg), list(B_HADRON_PDGS))
    if not b_mask.any():
        return {"n_in_cone": 0, "n_leaf": 0, "px": 0.0, "py": 0.0, "pz": 0.0,
                "E": 0.0, "pt": 0.0, "pdgs": []}

    dphi = phi_all - phi0
    dphi = np.where(dphi >  math.pi, dphi - 2 * math.pi, dphi)
    dphi = np.where(dphi < -math.pi, dphi + 2 * math.pi, dphi)
    in_cone = (eta_all - eta0) ** 2 + dphi ** 2 < dR ** 2
    sel = b_mask & in_cone
    n_in_cone = int(sel.sum())
    if n_in_cone == 0:
        return {"n_in_cone": 0, "n_leaf": 0, "px": 0.0, "py": 0.0, "pz": 0.0,
                "E": 0.0, "pt": 0.0, "pdgs": []}

    # Drop any selected B-hadron whose particle_id appears as the parent_id
    # of ANOTHER selected B-hadron — that's the parent-of-cascade case.
    sel_idx = np.where(sel)[0]
    sel_pids = set(int(particle_id[i]) for i in sel_idx)
    parents_of_other = {
        int(parent_id[j]) for j in sel_idx
    }
    keep_idx = np.array(
        [i for i in sel_idx if int(particle_id[i]) not in parents_of_other],
        dtype=int,
    )
    if len(keep_idx) == 0:
        # Pathological: all are parents of each other; fall back to highest pT
        keep_idx = np.array([int(sel_idx[np.argmax(pt[sel_idx])])])

    pxs = float(px[keep_idx].sum()); pys = float(py[keep_idx].sum())
    return {
        "n_in_cone": n_in_cone,
        "n_leaf":    int(len(keep_idx)),
        "px": pxs, "py": pys,
        "pz": float(pz[keep_idx].sum()),
        "E":  float(E[keep_idx].sum()),
        "pt": float(math.hypot(pxs, pys)),
        "pdgs": [int(pdg[i]) for i in keep_idx],
    }


def b_plus_brothers_in_cone(raw_row: dict, eta0: float, phi0: float,
                            dR: float = JET_R) -> dict:
    """Extend `sum_leaf_b_hadrons_in_cone` with each leaf-B's brothers
    (particles sharing the same `parent_id` as a leaf B in cone). Returns
    the 4-momentum sum AND the *set of particle indices* contributing —
    callers union the indices across the two matched jets so the dijet
    sum never double-counts the same particle (relevant when two leaf B's
    in the same cone share a parent → they're each other's brothers, and
    the full sibling group is collected only once)."""
    pdg = np.asarray(raw_row["pdg_id"], dtype=np.int64)
    vp  = np.asarray(raw_row["vertex_primary"], dtype=np.int64)
    particle_id = np.asarray(raw_row["particle_id"], dtype=np.int64)
    parent_id   = np.asarray(raw_row["parent_id"],   dtype=np.int64)
    px  = np.asarray(raw_row["px"], dtype=np.float64)
    py  = np.asarray(raw_row["py"], dtype=np.float64)
    pz  = np.asarray(raw_row["pz"], dtype=np.float64)
    E   = np.asarray(raw_row["energy"], dtype=np.float64)
    pt = np.hypot(px, py)
    eta_all = np.arcsinh(pz / np.maximum(pt, 1e-30))
    phi_all = np.arctan2(py, px)

    empty = {"n_b": 0, "n_bro": 0, "px": 0.0, "py": 0.0, "pz": 0.0,
             "E": 0.0, "pt": 0.0, "indices": set()}
    b_mask = (vp == 1) & np.isin(np.abs(pdg), list(B_HADRON_PDGS))
    if not b_mask.any():
        return empty

    dphi = phi_all - phi0
    dphi = np.where(dphi >  math.pi, dphi - 2 * math.pi, dphi)
    dphi = np.where(dphi < -math.pi, dphi + 2 * math.pi, dphi)
    in_cone = (eta_all - eta0) ** 2 + dphi ** 2 < dR ** 2
    sel = b_mask & in_cone
    if not sel.any():
        return empty

    # Same leaf cut as sum_leaf_b_hadrons_in_cone: drop a B-hadron whose
    # particle_id is the parent_id of another selected B-hadron.
    sel_idx = np.where(sel)[0]
    parents_of_other = {int(parent_id[j]) for j in sel_idx}
    keep_idx = [i for i in sel_idx
                if int(particle_id[i]) not in parents_of_other]
    if not keep_idx:
        keep_idx = [int(sel_idx[np.argmax(pt[sel_idx])])]

    # Expand: collect every particle whose parent_id matches the parent
    # of a kept leaf B. That's "B + its brothers" for each kept B; using
    # a set over indices makes shared-parent leaf B's contribute the
    # sibling group exactly once.
    unique_parents = np.fromiter(
        {int(parent_id[i]) for i in keep_idx}, dtype=np.int64
    )
    sibling_mask = np.isin(parent_id, unique_parents)
    indices = set(int(i) for i in np.where(sibling_mask)[0])
    if not indices:
        return empty

    idx_arr = np.fromiter(indices, dtype=int)
    pxs = float(px[idx_arr].sum()); pys = float(py[idx_arr].sum())
    return {
        "n_b":   len(keep_idx),
        "n_bro": len(indices) - len(keep_idx),
        "px": pxs, "py": pys,
        "pz": float(pz[idx_arr].sum()),
        "E":  float(E[idx_arr].sum()),
        "pt": float(math.hypot(pxs, pys)),
        "indices": indices,
    }


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


def _diagnose_list_column_lengths(df: pl.DataFrame, label: str) -> None:
    """If any event has list-columns of inconsistent lengths, print the
    offending (event_id, column, length) details. preprocess_for_model
    explodes multiple list columns together and will raise ShapeError
    if any single event has mismatched lengths, so we surface the cause
    explicitly here instead of letting it crash without context."""
    list_cols = [c for c, dtype in df.schema.items()
                 if isinstance(dtype, pl.List)]
    if not list_cols:
        return
    any_bad = False
    for row in df.iter_rows(named=True):
        lens = {c: len(row[c]) for c in list_cols}
        if len(set(lens.values())) > 1:
            if not any_bad:
                print(f"\n!!! list-column length MISMATCH in {label} !!!")
                any_bad = True
            print(f"  event_id={row['event_id']}:")
            for c, n in sorted(lens.items()):
                print(f"    {c:<25}  len={n}")


def process_chunk(file_to_events: dict[int, list[int]],
                  truth_pt_cut: float, truth_eta_cut: float,
                  target_pt_cut: float, clusters_cutoff: float
                  ) -> tuple[list[float], list[float], list[float], list[float],
                             list[float], list[float], list[float], list[float]]:
    """One subprocess's worth of work: process all events in the chunk,
    one HF file at a time. Particle_ids are local to each HF file, so
    concatenating events across files would break the joins inside
    preprocess_for_model — we keep each file's events isolated."""
    masses_vis: list[float] = []
    masses_corr: list[float] = []
    masses_bsum: list[float] = []
    masses_bpbr: list[float] = []
    masses_top2: list[float] = []
    lead_pt: list[float] = []
    sublead_pt: list[float] = []
    delta_r_top2: list[float] = []
    for fi in sorted(file_to_events):
        eids = file_to_events[fi]
        v, c, b, bb, t, lp, slp, dr = _process_single_file(
            fi, eids,
            truth_pt_cut, truth_eta_cut, target_pt_cut, clusters_cutoff,
        )
        masses_vis.extend(v)
        masses_corr.extend(c)
        masses_bsum.extend(b)
        masses_bpbr.extend(bb)
        masses_top2.extend(t)
        lead_pt.extend(lp)
        sublead_pt.extend(slp)
        delta_r_top2.extend(dr)
    return (masses_vis, masses_corr, masses_bsum, masses_bpbr, masses_top2,
            lead_pt, sublead_pt, delta_r_top2)


def _process_single_file(file_idx: int, event_ids: list[int],
                         truth_pt_cut: float, truth_eta_cut: float,
                         target_pt_cut: float, clusters_cutoff: float
                         ) -> tuple[list[float], list[float]]:
    """Download + preprocess + analyse the events of a single HF file. No
    cross-file particle_id collisions can occur here."""
    single = {file_idx: event_ids}
    print(f"\n--- HF file {file_idx}: {len(event_ids)} events ---")
    particles_bb = load_events(single, kind="particles",
                               columns=DEFAULT_COLUMNS["particles"])
    tracks_bb    = load_events(single, kind="tracks",
                               columns=DEFAULT_COLUMNS["tracks"])
    calo_hits_bb = load_events(single, kind="calo_hits",
                               columns=DEFAULT_COLUMNS["calo_hits"])

    # Diagnostic: surface any per-event list-column length mismatch before
    # preprocess_for_model raises its less-informative ShapeError.
    _diagnose_list_column_lengths(particles_bb, f"particles_bb[file={file_idx}]")
    _diagnose_list_column_lengths(tracks_bb,    f"tracks_bb[file={file_idx}]")
    _diagnose_list_column_lengths(calo_hits_bb, f"calo_hits_bb[file={file_idx}]")

    print(f"running preprocess_for_model on file {file_idx} …")
    out = preprocess_for_model(
        particles=particles_bb, tracks=tracks_bb, calo_hits=calo_hits_bb,
        num_of_events=-1,
        truth_pt_cut=truth_pt_cut, truth_eta_cut=truth_eta_cut,
        target_pt_cut=target_pt_cut, clusters_cutoff=clusters_cutoff,
    )
    target_particles = out["target_particles"]
    raw_by_eid = {int(r["event_id"]): r
                  for r in particles_bb.iter_rows(named=True)}

    masses_vis: list[float] = []
    masses_corr: list[float] = []
    masses_bsum: list[float] = []
    masses_bpbr: list[float] = []
    masses_top2: list[float] = []
    lead_pt: list[float] = []
    sublead_pt: list[float] = []
    delta_r_top2: list[float] = []
    for tp in target_particles.iter_rows(named=True):
        eid = int(tp["event_id"])
        pt  = np.asarray(tp["pt"],  dtype=np.float64)
        eta = np.asarray(tp["eta"], dtype=np.float64)
        phi = np.asarray(tp["phi"], dtype=np.float64)

        # 1. Get all distinct, central B-hadron directions (|η|<3.5, ΔR-clustered)
        bhads_all = get_all_b_hadrons(raw_by_eid[eid])

        # 2. Pick the pair that best matches the Higgs, applying ALL conditions:
        #    - both candidates already pass |η|<3.5 (enforced in get_all_b_hadrons)
        #    - both already at distinct cluster centers (ΔR>0.4 via clustering)
        #    - additionally require explicit ΔR>B_HAD_PAIR_DR_MIN between them
        #    - among surviving pairs, choose the one whose M is closest to 125 GeV.
        bhads = []
        best_mass = 0.0
        if len(bhads_all) >= 2:
            best_diff = float('inf')
            for i in range(len(bhads_all)):
                for j in range(i + 1, len(bhads_all)):
                    a, b = bhads_all[i], bhads_all[j]
                    dr_ab = _delta_r(a["eta"], a["phi"], b["eta"], b["phi"])
                    if dr_ab < B_HAD_PAIR_DR_MIN:
                        continue
                    m = _mass_two_dict(a, b)
                    if abs(m - HIGGS_MASS) < best_diff:
                        best_diff = abs(m - HIGGS_MASS)
                        bhads = [a, b]
                        best_mass = m
        # if exactly 1 candidate survived (and we still want to report it)
        if not bhads and bhads_all:
            bhads = bhads_all[:1]

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

        # Top-2 leading-pT dijet mass — b-tag agnostic baseline.
        # jets are pT-sorted (fastjet sorted_by_pt at cluster_jets_event).
        # Always record leading/sub-leading pT and ΔR for distributions;
        # only record the dijet mass when ΔR>0.4.
        if len(jets) >= 2:
            ja, jb = jets[0], jets[1]
            dr_ab = _delta_r(ja["eta"], ja["phi"], jb["eta"], jb["phi"])
            lead_pt.append(float(ja["pt"]))
            sublead_pt.append(float(jb["pt"]))
            delta_r_top2.append(float(dr_ab))
            if dr_ab > 0.4:
                m2_top2 = ((ja["E"] + jb["E"]) ** 2
                           - (ja["px"] + jb["px"]) ** 2
                           - (ja["py"] + jb["py"]) ** 2
                           - (ja["pz"] + jb["pz"]) ** 2)
                m_top2 = float(math.sqrt(max(m2_top2, 0.0)))
                masses_top2.append(m_top2)
                print(f"  → top-2 leading dijet mass  M(j1,j2)      = {m_top2:7.2f} GeV  "
                      f"(ΔR={dr_ab:.3f})")

        nus = get_neutrinos_vp1(raw_by_eid[eid])
        matches = match_b_to_jets(bhads, jets, dr_cut=DR_MATCH)
        print(f"  ghost-association (ΔR<{DR_MATCH}), with in-cone ν sums:")

        nu_sums:   list[dict] = []   # in-cone ν sum per matched jet (or zero)
        b_sums:    list[dict] = []   # in-cone leaf-B-hadron sum per matched jet
        bpbr_sums: list[dict] = []   # leaf-B + brothers (by parent_id) per jet
        for k, (b, ji) in enumerate(zip(bhads, matches), 1):
            zero4 = {"n": 0, "px": 0.0, "py": 0.0, "pz": 0.0, "E": 0.0, "pt": 0.0}
            zero_bpbr = {**zero4, "n_b": 0, "n_bro": 0, "indices": set()}
            if ji is None:
                print(f"    bH#{k} → NO MATCHED JET within ΔR<{DR_MATCH}")
                nu_sums.append(dict(zero4))
                b_sums.append(dict(zero4))
                bpbr_sums.append(dict(zero_bpbr))
                continue
            j = jets[ji]
            dr = _delta_r(b["eta"], b["phi"], j["eta"], j["phi"])
            d_pt = j["pt"] - b["pt"]
            rel = d_pt / b["pt"] if b["pt"] > 0 else 0.0
            nu = sum_neutrinos_in_cone(nus, j["eta"], j["phi"], dR=JET_R)
            nu_sums.append(nu)
            nu_frac = nu["pt"] / j["pt"] if j["pt"] > 0 else 0.0
            bsum = sum_leaf_b_hadrons_in_cone(raw_by_eid[eid],
                                              j["eta"], j["phi"], dR=JET_R)
            b_sums.append(bsum)
            bpbr = b_plus_brothers_in_cone(raw_by_eid[eid],
                                           j["eta"], j["phi"], dR=JET_R)
            bpbr_sums.append(bpbr)
            E_b = bsum["E"]
            dE = j["E"] - E_b
            rel_E = dE / E_b if E_b > 0 else 0.0
            print(f"    bH#{k} → j#{ji+1}  pT_jet={j['pt']:7.2f}  "
                  f"ΔR={dr:.3f}  ΔpT={d_pt:+6.2f} ({rel:+.0%})")
            print(f"        E_jet={j['E']:7.2f}  E_b_leaves={E_b:7.2f}  "
                  f"ΔE={dE:+6.2f} ({rel_E:+.0%})  "
                  f"(n_B_in_cone={bsum['n_in_cone']}, n_leaf={bsum['n_leaf']}, pdgs={bsum['pdgs']})")
            print(f"        B+brothers: n_B={bpbr['n_b']}  n_bro={bpbr['n_bro']}  "
                  f"E={bpbr['E']:7.2f}  pT={bpbr['pt']:7.2f}")
            print(f"        in-cone ν: n={nu['n']}  pT_ν={nu['pt']:6.2f}  "
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
            # Leaf-B-hadron-only dijet mass: invariant mass of the two
            # leaf-B-hadron sums in the matched jet cones (truth-level
            # B-pair, jet-clustering bypassed).
            b1, b2 = b_sums[0], b_sums[1]
            if b1["E"] > 0 and b2["E"] > 0:
                E_b  = b1["E"]  + b2["E"]
                px_b = b1["px"] + b2["px"]
                py_b = b1["py"] + b2["py"]
                pz_b = b1["pz"] + b2["pz"]
                mjj_b = float(math.sqrt(max(
                    E_b * E_b - (px_b * px_b + py_b * py_b + pz_b * pz_b), 0.0
                )))
            else:
                mjj_b = 0.0
            masses_bsum.append(mjj_b)
            # (Leaf-B + brothers) dijet mass: sum 4-momentum over the UNION
            # of the two jets' index sets, so any particle in both cones
            # (or any sibling shared between leaf B's) is counted exactly
            # once. Falls back to 0.0 if no B's were found in either cone.
            bb1, bb2 = bpbr_sums[0], bpbr_sums[1]
            union_idx = bb1["indices"] | bb2["indices"]
            if union_idx:
                raw = raw_by_eid[eid]
                px_r = np.asarray(raw["px"], dtype=np.float64)
                py_r = np.asarray(raw["py"], dtype=np.float64)
                pz_r = np.asarray(raw["pz"], dtype=np.float64)
                E_r  = np.asarray(raw["energy"], dtype=np.float64)
                u = np.fromiter(union_idx, dtype=int)
                E_u  = float(E_r[u].sum())
                px_u = float(px_r[u].sum())
                py_u = float(py_r[u].sum())
                pz_u = float(pz_r[u].sum())
                mjj_bpbr = float(math.sqrt(max(
                    E_u * E_u - (px_u * px_u + py_u * py_u + pz_u * pz_u), 0.0
                )))
            else:
                mjj_bpbr = 0.0
            masses_bpbr.append(mjj_bpbr)
            print(f"  → visible dijet mass        M(bj,b̄j)     = {mjj_vis:7.2f} GeV")
            print(f"  → ν-corrected dijet mass    M(bj+ν,b̄j+ν) = {mjj_corr:7.2f} GeV  "
                  f"(in-cone ν pT: {nu1['pt']:.2f} + {nu2['pt']:.2f})")
            print(f"  → leaf-B-hadron dijet mass  M(ΣB,ΣB̄)     = {mjj_b:7.2f} GeV  "
                  f"(E_b: {b1['E']:.1f} + {b2['E']:.1f})")
            print(f"  → B+brothers dijet mass     M(ΣB+bro,…)  = {mjj_bpbr:7.2f} GeV  "
                  f"(n_particles_union={len(union_idx)})")
        else:
            print("  → could not form b-tagged dijet (one or both b-jets missing)")

    return (masses_vis, masses_corr, masses_bsum, masses_bpbr, masses_top2,
            lead_pt, sublead_pt, delta_r_top2)


def print_running_summary(masses_vis: list[float],
                          masses_corr: list[float],
                          masses_bsum: list[float],
                          masses_bpbr: list[float] | None = None,
                          masses_top2: list[float] | None = None) -> None:
    if not masses_vis and not masses_top2:
        print("(no events reconstructed yet)")
        return
    vis = np.array(masses_vis)
    cor = np.array(masses_corr)
    bsm = np.array(masses_bsum) if masses_bsum else np.array([])
    bpr = np.array(masses_bpbr) if masses_bpbr else np.array([])
    top = np.array(masses_top2) if masses_top2 else np.array([])
    print(f"\n=== running aggregate across {len(vis)} reconstructed events "
          f"(top-2 jets: {len(top)} events) ===")
    print(f"{'':<20}  {'jet visible':>12}  {'jet+ν':>10}  {'leaf-B':>10}  "
          f"{'leaf-B+bro':>11}  {'top-2 jets':>11}")
    print("-" * 86)
    def _row(label, get):
        a = get(vis) if vis.size else float('nan')
        b = get(cor) if cor.size else float('nan')
        c = get(bsm) if bsm.size else float('nan')
        d = get(bpr) if bpr.size else float('nan')
        e = get(top) if top.size else float('nan')
        print(f"  {label:<20}  {a:>12.2f}  {b:>10.2f}  {c:>10.2f}  "
              f"{d:>11.2f}  {e:>11.2f}")
    _row("mean M(bb̄)",  lambda x: float(x.mean()))
    _row("median",      lambda x: float(np.median(x)))
    _row("std",         lambda x: float(x.std()))
    _row("min",         lambda x: float(x.min()))
    _row("max",         lambda x: float(x.max()))

    def _hits(arr, w):
        return f"{int((np.abs(arr - HIGGS_MASS) < w).sum())}/{len(arr)} " \
               f"({(np.abs(arr - HIGGS_MASS) < w).mean():.0%})"
    print(f"  within ±5  of 125    {_hits(vis, 5):>12}  {_hits(cor, 5):>10}  "
          f"{_hits(bsm, 5) if bsm.size else '—':>10}  "
          f"{_hits(bpr, 5) if bpr.size else '—':>11}  "
          f"{_hits(top, 5) if top.size else '—':>11}")
    print(f"  within ±10 of 125    {_hits(vis, 10):>12}  {_hits(cor, 10):>10}  "
          f"{_hits(bsm, 10) if bsm.size else '—':>10}  "
          f"{_hits(bpr, 10) if bpr.size else '—':>11}  "
          f"{_hits(top, 10) if top.size else '—':>11}")
    print(f"  within ±20 of 125    {_hits(vis, 20):>12}  {_hits(cor, 20):>10}  "
          f"{_hits(bsm, 20) if bsm.size else '—':>10}  "
          f"{_hits(bpr, 20) if bpr.size else '—':>11}  "
          f"{_hits(top, 20) if top.size else '—':>11}")


def get_chunk_file_to_events(label_dir: Path, n_bb_events: int,
                             chunk_size: int, chunk_idx: int
                             ) -> dict[int, list[int]]:
    """Deterministically slice the first n_bb_events bb̄ events into chunks
    of chunk_size and return the file_to_events dict for chunk_idx."""
    full = load_bb_event_ids(label_dir, n_bb_events)
    flat = sorted((fi, eid) for fi, eids in full.items() for eid in eids)
    start = chunk_idx * chunk_size
    chunk_pairs = flat[start : start + chunk_size]
    out: dict[int, list[int]] = defaultdict(list)
    for fi, eid in chunk_pairs:
        out[fi].append(eid)
    return dict(out)


def worker_main(label_dir: Path, n_bb_events: int, chunk_size: int,
                chunk_idx: int, truth_pt_cut: float, truth_eta_cut: float,
                target_pt_cut: float, clusters_cutoff: float) -> None:
    """Subprocess entry: process exactly one chunk and print results as a
    sentinel-tagged JSON line on stdout. Polars / FastJet allocations all
    die when this process exits."""
    file_to_events = get_chunk_file_to_events(
        label_dir, n_bb_events, chunk_size, chunk_idx
    )
    flat = sorted((fi, eid) for fi, eids in file_to_events.items() for eid in eids)
    print(f"[chunk {chunk_idx}] {len(flat)} events: {flat}")

    (masses_vis, masses_corr, masses_bsum, masses_bpbr, masses_top2,
     lead_pt, sublead_pt, delta_r_top2) = process_chunk(
        file_to_events,
        truth_pt_cut=truth_pt_cut, truth_eta_cut=truth_eta_cut,
        target_pt_cut=target_pt_cut, clusters_cutoff=clusters_cutoff,
    )
    payload = {"vis": masses_vis, "corr": masses_corr,
               "bsum": masses_bsum, "bpbr": masses_bpbr,
               "top2": masses_top2,
               "lead_pt": lead_pt, "sublead_pt": sublead_pt,
               "dr_top2": delta_r_top2}
    print(f"{RESULT_SENTINEL}{json.dumps(payload)}")


def plot_histograms(masses_vis: list[float],
                    masses_corr: list[float],
                    masses_bsum: list[float],
                    masses_bpbr: list[float],
                    masses_top2: list[float],
                    lead_pt: list[float],
                    sublead_pt: list[float],
                    delta_r_top2: list[float],
                    out_dir: Path) -> None:
    """Write PNGs of the four dijet-mass distributions plus leading/
    sub-leading jet pT and ΔR(j1, j2). Overwrites on each call so the
    plots track the running aggregate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def _mass_hist(values: list[float], title: str, fname: str) -> None:
        if not values:
            return
        arr = np.asarray(values)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(arr, bins=np.linspace(0, 250, 51),
                color="steelblue", edgecolor="black", linewidth=0.4)
        ax.axvline(HIGGS_MASS, color="crimson", linestyle="--",
                   linewidth=1.2, label=f"M_H = {HIGGS_MASS:.1f} GeV")
        ax.set_xlabel("M(j,j) [GeV]")
        ax.set_ylabel("events / 5 GeV")
        ax.set_title(f"{title}  (N = {len(arr)}, "
                     f"mean = {arr.mean():.1f}, median = {np.median(arr):.1f})")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)

    _mass_hist(masses_vis,  "jet visible — M(bj, b̄j)",        "mass_jet_visible.png")
    _mass_hist(masses_corr, "jet + in-cone ν — M(bj+ν, b̄j+ν)", "mass_jet_plus_nu.png")
    _mass_hist(masses_bsum, "leaf-B-hadron sum — M(ΣB, ΣB̄)",   "mass_leaf_b.png")
    _mass_hist(masses_bpbr, "leaf-B + brothers — M(ΣB+bro, ΣB̄+bro)",
               "mass_b_plus_brothers.png")
    _mass_hist(masses_top2, "top-2 leading-pT jets — M(j1, j2)", "mass_top2_jets.png")

    def _pt_hist(values: list[float], title: str, fname: str) -> None:
        if not values:
            return
        arr = np.asarray(values)
        upper = max(300.0, float(np.percentile(arr, 99)) * 1.05)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(arr, bins=np.linspace(0, upper, 41),
                color="darkorange", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("p_T [GeV]")
        ax.set_ylabel("events")
        ax.set_title(f"{title}  (N = {len(arr)}, "
                     f"mean = {arr.mean():.1f}, median = {np.median(arr):.1f})")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)

    _pt_hist(lead_pt,    "leading jet pT",     "pt_leading_jet.png")
    _pt_hist(sublead_pt, "sub-leading jet pT", "pt_subleading_jet.png")

    if delta_r_top2:
        arr = np.asarray(delta_r_top2)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(arr, bins=np.linspace(0, 6.0, 41),
                color="seagreen", edgecolor="black", linewidth=0.4)
        ax.axvline(0.4, color="crimson", linestyle="--", linewidth=1.0,
                   label="ΔR = 0.4 (jet R)")
        ax.set_xlabel("ΔR(j1, j2)")
        ax.set_ylabel("events")
        ax.set_title(f"ΔR between top-2 leading jets  (N = {len(arr)}, "
                     f"mean = {arr.mean():.2f}, median = {np.median(arr):.2f})")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "deltaR_top2_jets.png", dpi=120)
        plt.close(fig)

    print(f"  [plots] wrote PNGs to {out_dir}")


def manager_loop(label_dir: Path, n_bb_events: int, chunk_size: int,
                 truth_pt_cut: float, truth_eta_cut: float,
                 target_pt_cut: float, clusters_cutoff: float,
                 plots_dir: Path) -> None:
    """Spawn one subprocess per chunk, sequentially. After each chunk
    completes, parse its results from stdout and print the running aggregate.
    Memory from each chunk is fully returned to the OS at subprocess exit."""
    full = load_bb_event_ids(label_dir, n_bb_events)
    flat = sorted((fi, eid) for fi, eids in full.items() for eid in eids)
    n_total = len(flat)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    print(f"selected {n_total} H→bb̄ events from {label_dir}")
    print(f"splitting into {n_chunks} chunks of {chunk_size} "
          f"(each chunk runs in its own subprocess)")

    script = str(Path(__file__).resolve())
    cum_vis:  list[float] = []
    cum_corr: list[float] = []
    cum_bsum: list[float] = []
    cum_bpbr: list[float] = []
    cum_top2: list[float] = []
    cum_lead_pt:    list[float] = []
    cum_sublead_pt: list[float] = []
    cum_dr_top2:    list[float] = []
    for ci in range(n_chunks):
        cmd = [
            sys.executable, "-u", script,
            "--worker", "--chunk-idx", str(ci),
            "--chunk-size", str(chunk_size),
            "--n-bb-events", str(n_bb_events),
            "--label-dir", str(label_dir),
            "--truth-pt-cut", str(truth_pt_cut),
            "--truth-eta-cut", str(truth_eta_cut),
            "--target-pt-cut", str(target_pt_cut),
            "--clusters-cutoff", str(clusters_cutoff),
        ]
        print(f"\n[chunk {ci+1}/{n_chunks}] spawning subprocess …")
        t0 = time.perf_counter()
        # Popen + line-by-line streaming so worker stdout appears LIVE in
        # the manager's terminal instead of arriving as a single bulk
        # block after the subprocess exits.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        result_line: str | None = None
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith(RESULT_SENTINEL):
                result_line = line
        rc = proc.wait()
        dt = time.perf_counter() - t0

        if rc != 0:
            print(f"  ! chunk {ci} subprocess exited {rc} ({dt:.1f}s)")
            continue

        chunk_vis:  list[float] | None = None
        chunk_corr: list[float] | None = None
        chunk_bsum: list[float] | None = None
        chunk_bpbr: list[float] | None = None
        chunk_top2: list[float] | None = None
        chunk_lead_pt:    list[float] | None = None
        chunk_sublead_pt: list[float] | None = None
        chunk_dr_top2:    list[float] | None = None
        if result_line is not None:
            try:
                payload = json.loads(result_line[len(RESULT_SENTINEL):])
                chunk_vis  = payload["vis"]
                chunk_corr = payload["corr"]
                chunk_bsum = payload.get("bsum", [])
                chunk_bpbr = payload.get("bpbr", [])
                chunk_top2 = payload.get("top2", [])
                chunk_lead_pt    = payload.get("lead_pt", [])
                chunk_sublead_pt = payload.get("sublead_pt", [])
                chunk_dr_top2    = payload.get("dr_top2", [])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ! could not parse worker result: {e}")
        if chunk_vis is None:
            print(f"  ! chunk {ci} produced no RESULT line — skipping")
            continue

        cum_vis.extend(chunk_vis)
        cum_corr.extend(chunk_corr)
        if chunk_bsum:
            cum_bsum.extend(chunk_bsum)
        if chunk_bpbr:
            cum_bpbr.extend(chunk_bpbr)
        if chunk_top2:
            cum_top2.extend(chunk_top2)
        if chunk_lead_pt:
            cum_lead_pt.extend(chunk_lead_pt)
        if chunk_sublead_pt:
            cum_sublead_pt.extend(chunk_sublead_pt)
        if chunk_dr_top2:
            cum_dr_top2.extend(chunk_dr_top2)
        print(f"  ✓ chunk {ci+1}/{n_chunks} done in {dt:.1f}s "
              f"({len(chunk_vis)} dijet masses recovered)")
        print_running_summary(cum_vis, cum_corr, cum_bsum, cum_bpbr, cum_top2)
        plot_histograms(cum_vis, cum_corr, cum_bsum, cum_bpbr, cum_top2,
                        cum_lead_pt, cum_sublead_pt, cum_dr_top2,
                        out_dir=plots_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR),
                        help=f"Where the bb̄ labels live (default: {DEFAULT_LABEL_DIR})")
    parser.add_argument("--n-bb-events", type=int, default=300,
                        help="Total bb̄ events to analyse (default: 300)")
    parser.add_argument("--chunk-size", type=int, default=75,
                        help="Events per subprocess chunk (default: 75)")
    parser.add_argument("--truth-pt-cut", type=float, default=1.0)
    parser.add_argument("--truth-eta-cut", type=float, default=3.0)
    parser.add_argument("--target-pt-cut", type=float, default=0.3)
    parser.add_argument("--clusters-cutoff", type=float, default=0.15)
    parser.add_argument("--plots-dir", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "plots_inspect_hbb"),
                        help="Where to write histogram PNGs")
    parser.add_argument("--worker", action="store_true",
                        help="(internal) worker mode: process --chunk-idx and exit")
    parser.add_argument("--chunk-idx", type=int, default=None,
                        help="(worker mode) chunk index to process")
    args = parser.parse_args()

    if args.worker:
        assert args.chunk_idx is not None, "--worker requires --chunk-idx"
        worker_main(
            label_dir=Path(args.label_dir),
            n_bb_events=args.n_bb_events,
            chunk_size=args.chunk_size,
            chunk_idx=args.chunk_idx,
            truth_pt_cut=args.truth_pt_cut, truth_eta_cut=args.truth_eta_cut,
            target_pt_cut=args.target_pt_cut, clusters_cutoff=args.clusters_cutoff,
        )
        return

    manager_loop(
        label_dir=Path(args.label_dir),
        n_bb_events=args.n_bb_events,
        chunk_size=args.chunk_size,
        truth_pt_cut=args.truth_pt_cut, truth_eta_cut=args.truth_eta_cut,
        target_pt_cut=args.target_pt_cut, clusters_cutoff=args.clusters_cutoff,
        plots_dir=Path(args.plots_dir),
    )


if __name__ == "__main__":
    main()