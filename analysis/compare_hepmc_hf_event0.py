"""
Side-by-side physics comparison: HepMC run 0 event 0  vs  HF file 0 event 0.

Both sides reduced to the "leaves of the stored decay tree" — particles
whose id appears in NO other particle's parent list, AND |pdg| not in
{12,14,16}. This is the strictest no-double-counting subset and is
defined identically on both sides so the numbers are directly comparable.

Run:
    python /storage/agrp/barakma/PileupODD/analysis/compare_hepmc_hf_event0.py
"""

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pyhepmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_higgs_diphoton_events import (  # noqa: E402
    HF_RESOLVE, HS_EVENT_NAME, NUM_HF_REPO_FILES,
)

HEPMC_FILE = "/storage/agrp/barakma/PileupODD/data/hepmc_dumps/run0.hepmc"
HF_FILE_IDX = 0
HF_EVENT_ID = 0
NU_PDGS = {12, 14, 16}


def _kin(px: float, py: float, pz: float) -> tuple[float, float, float]:
    pt = math.hypot(px, py)
    p_mag = math.hypot(pt, pz)
    if p_mag == abs(pz):
        eta = math.inf if pz > 0 else -math.inf
    else:
        eta = 0.5 * math.log((p_mag + pz) / (p_mag - pz))
    phi = math.atan2(py, px)
    return pt, eta, phi


def _stats_from_arrays(pdg, px, py, pz, energy) -> dict:
    pt = np.hypot(px, py)
    p_mag = np.sqrt(px * px + py * py + pz * pz)
    # safe eta
    eta = np.where(p_mag == np.abs(pz), np.sign(pz) * np.inf,
                   0.5 * np.log((p_mag + pz) / np.maximum(p_mag - pz, 1e-30)))
    phi = np.arctan2(py, px)

    n = len(pt)
    sum_pt = float(pt.sum())
    sum_E = float(energy.sum())
    # MET on this subset = magnitude of the vector pT sum of *these* particles
    # (gives "recoil-against-this-set" not real ν-MET, which we compute
    # separately from neutrinos)
    pxs = float(px.sum()); pys = float(py.sum())
    recoil_pt = math.hypot(pxs, pys)

    order = np.argsort(pt)[::-1]
    lead = order[0] if n >= 1 else None
    sub = order[1] if n >= 2 else None
    def _row(i):
        if i is None: return None
        return {
            "pdg": int(pdg[i]),
            "pt": float(pt[i]),
            "eta": float(eta[i]),
            "phi": float(phi[i]),
        }

    pdg_hist = Counter(int(p) for p in pdg).most_common(5)

    return {
        "n_stable": n,
        "sum_pt": sum_pt,
        "sum_E": sum_E,
        "recoil_pt_of_subset": recoil_pt,
        "lead": _row(lead),
        "sublead": _row(sub),
        "n_pt_gt_5":  int((pt > 5).sum()),
        "n_pt_gt_20": int((pt > 20).sum()),
        "pdg_hist_top5": pdg_hist,
    }


def compute_hepmc_stats(path: str, event_index: int = 0) -> dict:
    with pyhepmc.open(path) as f:
        evt = None
        for i, e in enumerate(f):
            if i == event_index:
                evt = e
                break
        if evt is None:
            raise RuntimeError(f"event_index {event_index} not in {path}")
        parts = evt.particles
        # leaf set: particle ids that appear in NO other particle's
        # production vertex's incoming list (i.e., have no daughters)
        # — equivalently: not in any other particle's parent set.
        # In HepMC pyhepmc API: parents = p.production_vertex.particles_in
        parent_ids: set[int] = set()
        for p in parts:
            pv = p.production_vertex
            if pv is None:
                continue
            for par in pv.particles_in:
                parent_ids.add(par.id)
        # collect leaves status==1 non-ν
        sel = []
        n_status1 = 0
        n_status1_nonleaf = 0
        n_nu = 0
        for p in parts:
            if p.status != 1:
                continue
            n_status1 += 1
            if abs(p.pid) in NU_PDGS:
                n_nu += 1
                continue
            if p.id in parent_ids:
                n_status1_nonleaf += 1
                continue
            sel.append(p)

        pdg = np.array([p.pid for p in sel], dtype=np.int64)
        px = np.array([p.momentum.px for p in sel], dtype=np.float64)
        py = np.array([p.momentum.py for p in sel], dtype=np.float64)
        pz = np.array([p.momentum.pz for p in sel], dtype=np.float64)
        E  = np.array([p.momentum.e  for p in sel], dtype=np.float64)
        stats = _stats_from_arrays(pdg, px, py, pz, E)
        stats["n_total"]              = len(parts)
        stats["n_status1_all"]        = n_status1
        stats["n_status1_neutrinos"]  = n_nu
        stats["n_status1_nonleaf_excluded"] = n_status1_nonleaf

        # MET from neutrinos
        nu_px = np.array([p.momentum.px for p in parts
                          if p.status == 1 and abs(p.pid) in NU_PDGS])
        nu_py = np.array([p.momentum.py for p in parts
                          if p.status == 1 and abs(p.pid) in NU_PDGS])
        stats["MET_from_neutrinos"] = (
            float(math.hypot(nu_px.sum(), nu_py.sum())) if len(nu_px) else 0.0
        )

        # Higgs decay (informational)
        higgs_daughters = []
        for p in parts:
            if p.pid == 25 and p.end_vertex:
                out = [c for c in p.end_vertex.particles_out if c.pid != 25]
                if out:
                    higgs_daughters = out
                    break
        stats["higgs_daughters"] = [
            {"pdg": d.pid, **dict(zip(("pt","eta","phi"),
                                       _kin(d.momentum.px, d.momentum.py, d.momentum.pz)))}
            for d in higgs_daughters
        ]
    return stats


def compute_hf_stats(file_idx: int, event_id: int) -> dict:
    url = (f"{HF_RESOLVE}/data/{HS_EVENT_NAME}_particles/"
           f"train-{file_idx:05d}-of-{NUM_HF_REPO_FILES:05d}.parquet")
    print(f"scanning {url} for event_id={event_id} ...")
    df = (pl.scan_parquet(url)
          .filter(pl.col("event_id") == event_id)
          .select("particle_id", "pdg_id", "px", "py", "pz", "energy",
                  "vertex_primary", "primary", "parent_id")
          .collect())
    if df.is_empty():
        raise RuntimeError(f"HF file {file_idx} has no event_id={event_id}")
    r = df.row(0, named=True)
    particle_id = np.asarray(r["particle_id"], dtype=np.int64)
    pdg        = np.asarray(r["pdg_id"],      dtype=np.int64)
    px         = np.asarray(r["px"],          dtype=np.float64)
    py         = np.asarray(r["py"],          dtype=np.float64)
    pz         = np.asarray(r["pz"],          dtype=np.float64)
    energy     = np.asarray(r["energy"],      dtype=np.float64)
    vp         = np.asarray(r["vertex_primary"], dtype=np.int64)
    primary    = np.asarray(r["primary"],     dtype=bool)
    parent_id  = np.asarray(r["parent_id"],   dtype=np.int64)

    n_total = len(particle_id)
    parent_set = set(int(p) for p in parent_id)
    # primary-true at vp==1, excluding neutrinos
    mask_primary = (vp == 1) & primary
    n_primary = int(mask_primary.sum())
    mask_nonu = mask_primary & ~np.isin(np.abs(pdg), list(NU_PDGS))
    n_nu_primary = int(mask_primary.sum() - mask_nonu.sum())
    # leaf filter
    is_leaf = np.array([int(pid) not in parent_set for pid in particle_id])
    mask_leaf = mask_nonu & is_leaf
    n_nonleaf_excluded = int(mask_nonu.sum() - mask_leaf.sum())

    sel = mask_leaf
    stats = _stats_from_arrays(pdg[sel], px[sel], py[sel], pz[sel], energy[sel])
    stats["n_total"]                   = n_total
    stats["n_vp1_primary"]             = n_primary
    stats["n_vp1_primary_neutrinos"]   = n_nu_primary
    stats["n_vp1_primary_nonleaf_excluded"] = n_nonleaf_excluded

    # MET from neutrinos at vp=1 primary
    mask_nu = mask_primary & np.isin(np.abs(pdg), list(NU_PDGS))
    if mask_nu.any():
        stats["MET_from_neutrinos"] = float(math.hypot(px[mask_nu].sum(), py[mask_nu].sum()))
    else:
        stats["MET_from_neutrinos"] = 0.0

    stats["higgs_daughters"] = None  # HF doesn't store the Higgs directly
    return stats


def _fmt_row(label: str, val) -> str:
    if val is None:
        return f"{label:<32}  {'—':>20}"
    if isinstance(val, float):
        return f"{label:<32}  {val:>20.3f}"
    if isinstance(val, int):
        return f"{label:<32}  {val:>20d}"
    return f"{label:<32}  {str(val):>20}"


def print_side_by_side(h: dict, f: dict) -> None:
    print("\n" + "=" * 78)
    print("HepMC run 0 event 0   vs   HF file 0 event_id 0")
    print("=" * 78)

    # Higgs decay (HepMC only)
    print("\nHiggs decay (HepMC only):")
    if h["higgs_daughters"]:
        for d in h["higgs_daughters"]:
            print(f"  pdg={d['pdg']:+d}  pT={d['pt']:7.2f}  η={d['eta']:+.2f}  φ={d['phi']:+.2f}")
    else:
        print("  not found in HepMC particles list")

    rows = [
        ("n_total (all stored particles)",     h["n_total"],                    f["n_total"]),
        ("HepMC status==1 / HF vp=1&primary",  h["n_status1_all"],              f["n_vp1_primary"]),
        ("  - excluded as neutrinos",          h["n_status1_neutrinos"],        f["n_vp1_primary_neutrinos"]),
        ("  - excluded as non-leaf",           h["n_status1_nonleaf_excluded"], f["n_vp1_primary_nonleaf_excluded"]),
        ("n_stable (comparable subset)",       h["n_stable"],                   f["n_stable"]),
        ("sum_pT  (GeV)",                      h["sum_pt"],                     f["sum_pt"]),
        ("sum_E   (GeV)",                      h["sum_E"],                      f["sum_E"]),
        ("|Σ pT_subset|  (recoil, GeV)",       h["recoil_pt_of_subset"],        f["recoil_pt_of_subset"]),
        ("MET from neutrinos (GeV)",           h["MET_from_neutrinos"],         f["MET_from_neutrinos"]),
        ("n with pT > 5  GeV",                 h["n_pt_gt_5"],                  f["n_pt_gt_5"]),
        ("n with pT > 20 GeV",                 h["n_pt_gt_20"],                 f["n_pt_gt_20"]),
    ]
    print(f"\n{'quantity':<34}  {'HepMC':>16}  {'HF':>16}  {'Δ':>10}")
    print("-" * 84)
    for label, hv, fv in rows:
        if isinstance(hv, (int, float)) and isinstance(fv, (int, float)):
            d = fv - hv
            d_str = f"{d:>+10.2f}" if isinstance(d, float) else f"{d:>+10d}"
            hs = f"{hv:>16.3f}" if isinstance(hv, float) else f"{hv:>16d}"
            fs = f"{fv:>16.3f}" if isinstance(fv, float) else f"{fv:>16d}"
            print(f"{label:<34}  {hs}  {fs}  {d_str}")
        else:
            print(f"{label:<34}  {str(hv):>16}  {str(fv):>16}  {'':>10}")

    def _print_leader(label: str, p: dict | None):
        if p is None:
            return f"  {label:<8}: (none)"
        return f"  {label:<8}: pdg={p['pdg']:+5d}  pT={p['pt']:7.2f}  η={p['eta']:+.2f}  φ={p['phi']:+.2f}"

    print("\nLeading & sub-leading particle in comparable subset:")
    print("  HepMC:")
    print(_print_leader("lead",    h["lead"]))
    print(_print_leader("sublead", h["sublead"]))
    print("  HF:")
    print(_print_leader("lead",    f["lead"]))
    print(_print_leader("sublead", f["sublead"]))

    # ΔR between leadings
    if h["lead"] and f["lead"]:
        deta = h["lead"]["eta"] - f["lead"]["eta"]
        dphi = h["lead"]["phi"] - f["lead"]["phi"]
        while dphi >  math.pi: dphi -= 2 * math.pi
        while dphi < -math.pi: dphi += 2 * math.pi
        dR = math.hypot(deta, dphi)
        same_pdg = h["lead"]["pdg"] == f["lead"]["pdg"]
        print(f"\n  ΔR(lead_HepMC, lead_HF) = {dR:.3f}  "
              f"(pdg match: {same_pdg})")

    print("\nTop-5 pdg in comparable subset:")
    print(f"  HepMC: {h['pdg_hist_top5']}")
    print(f"  HF:    {f['pdg_hist_top5']}")
    print("=" * 78)


def collect_hf_orphans(file_idx: int, event_id: int) -> list[dict]:
    """HF particles at vp==1 whose parent_id is NOT in the stored particle_id
    set. These are the "tops" of the stored chains — closest analog to the
    generator-level particles before Geant4 expands them into showers."""
    url = (f"{HF_RESOLVE}/data/{HS_EVENT_NAME}_particles/"
           f"train-{file_idx:05d}-of-{NUM_HF_REPO_FILES:05d}.parquet")
    df = (pl.scan_parquet(url)
          .filter(pl.col("event_id") == event_id)
          .select("particle_id", "pdg_id", "px", "py", "pz", "energy",
                  "vertex_primary", "parent_id")
          .collect())
    r = df.row(0, named=True)
    particle_id = np.asarray(r["particle_id"], dtype=np.int64)
    pdg = np.asarray(r["pdg_id"], dtype=np.int64)
    px = np.asarray(r["px"], dtype=np.float64)
    py = np.asarray(r["py"], dtype=np.float64)
    pz = np.asarray(r["pz"], dtype=np.float64)
    energy = np.asarray(r["energy"], dtype=np.float64)
    vp = np.asarray(r["vertex_primary"], dtype=np.int64)
    parent_id = np.asarray(r["parent_id"], dtype=np.int64)

    pid_set = set(int(p) for p in particle_id)
    is_orphan = np.array([int(p) not in pid_set for p in parent_id])
    sel = (vp == 1) & is_orphan
    pt = np.hypot(px, py)
    p_mag = np.sqrt(px * px + py * py + pz * pz)
    eta = np.where(p_mag == np.abs(pz), np.sign(pz) * np.inf,
                   0.5 * np.log((p_mag + pz) / np.maximum(p_mag - pz, 1e-30)))
    phi = np.arctan2(py, px)

    out = []
    for k in np.where(sel)[0]:
        out.append({
            "pdg": int(pdg[k]),
            "pt":  float(pt[k]),
            "eta": float(eta[k]),
            "phi": float(phi[k]),
            "E":   float(energy[k]),
            "parent_id_missing": int(parent_id[k]),
        })
    out.sort(key=lambda d: -d["pt"])
    return out


def collect_hepmc_particles(path: str, event_index: int) -> list[dict]:
    """All HepMC particles in the chosen event (any status), with kinematics
    and status so we can do nearest-match against HF orphans."""
    with pyhepmc.open(path) as f:
        evt = None
        for i, e in enumerate(f):
            if i == event_index:
                evt = e
                break
        if evt is None:
            raise RuntimeError(f"event_index {event_index} not in {path}")
        out = []
        for p in evt.particles:
            px, py, pz = p.momentum.px, p.momentum.py, p.momentum.pz
            pt, eta, phi = _kin(px, py, pz)
            out.append({
                "id":  p.id,
                "pdg": p.pid,
                "status": p.status,
                "pt": pt, "eta": eta, "phi": phi,
                "E": p.momentum.e,
            })
    return out


def _dR(e1, p1, e2, p2):
    dphi = p1 - p2
    while dphi >  math.pi: dphi -= 2 * math.pi
    while dphi < -math.pi: dphi += 2 * math.pi
    return math.hypot(e1 - e2, dphi)


def match_hf_orphans_to_hepmc(
    orphans: list[dict], hepmc: list[dict], top_n: int = 20,
) -> None:
    print("\n" + "=" * 92)
    print(f"HF orphans (vp=1, parent_id not in stored set) — top {top_n} by pT")
    print("each matched to its nearest HepMC particle (any status), preferring same-pdg")
    print("=" * 92)
    hep_eta = np.array([p["eta"] for p in hepmc])
    hep_phi = np.array([p["phi"] for p in hepmc])
    hep_pt  = np.array([p["pt"]  for p in hepmc])
    hep_pdg = np.array([p["pdg"] for p in hepmc])
    hep_status = np.array([p["status"] for p in hepmc])
    print(f"\n{'HF orphan':<58}  {'closest HepMC':<60}  {'ΔR':>5}")
    print("-" * 130)
    for o in orphans[:top_n]:
        deta = hep_eta - o["eta"]
        dphi = hep_phi - o["phi"]
        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
        dR = np.hypot(deta, dphi)
        # prefer same-pdg if any same-pdg is within ΔR<0.05; else pick global min ΔR
        same_pdg = hep_pdg == o["pdg"]
        candidates = np.where(same_pdg & (dR < 0.05))[0]
        if len(candidates) > 0:
            k = candidates[np.argmin(dR[candidates])]
        else:
            k = int(np.argmin(dR))
        hp = hepmc[k]
        hf_str = (f"pdg={o['pdg']:+5d} pT={o['pt']:7.2f} "
                  f"η={o['eta']:+.2f} φ={o['phi']:+.2f}")
        hep_str = (f"pdg={hp['pdg']:+5d} status={hp['status']:>2d} "
                   f"pT={hp['pt']:7.2f} η={hp['eta']:+.2f} φ={hp['phi']:+.2f}")
        tag = ""
        if o["pdg"] != hp["pdg"]: tag += " [pdg≠]"
        if dR[k] > 0.05:         tag += " [ΔR>0.05]"
        print(f"  {hf_str:<56}  {hep_str:<58}  {dR[k]:.3f}{tag}")
    print("=" * 92)


def main():
    h = compute_hepmc_stats(HEPMC_FILE, event_index=0)
    fhf = compute_hf_stats(HF_FILE_IDX, HF_EVENT_ID)
    print_side_by_side(h, fhf)

    print("\n\nFollow-up: HF orphans (vp=1) vs all HepMC particles")
    orphans = collect_hf_orphans(HF_FILE_IDX, HF_EVENT_ID)
    hepmc_all = collect_hepmc_particles(HEPMC_FILE, 0)
    print(f"HF orphans at vp=1: {len(orphans)}   |   HepMC particles total: {len(hepmc_all)}")
    match_hf_orphans_to_hepmc(orphans, hepmc_all, top_n=20)


if __name__ == "__main__":
    main()
