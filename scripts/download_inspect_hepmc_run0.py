"""
Download the HepMC truth file for ColliderML run 0 and dump enough of its
structure to figure out how its event IDs / kinematics map to the
HuggingFace ColliderML particle dataset.

Output:
- {OUT_DIR}/run0.hepmc            — raw HepMC (kept, not deleted)
- stdout summary per event: event_number, Higgs decay channel, daughter
  pT/η/φ, total visible pT, leading-pT particle (pdg, pT, η, φ)
"""

import math
import sys
from pathlib import Path

import requests
import pyhepmc

URL = "https://portal.nersc.gov/cfs/m4958/ColliderML/full_pileup/ggf/v1/runs/0/events.hepmc"
OUT_DIR = Path("/storage/agrp/barakma/PileupODD/data/hepmc_dumps")
OUT_FILE = OUT_DIR / "run0.hepmc"

PDG_NAMES = {
    5: "b", 4: "c", 22: "γ", 23: "Z", 24: "W", 15: "τ", 21: "g",
    13: "μ", 11: "e", 12: "νe", 14: "νμ", 16: "ντ",
}


def download():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        size_mb = OUT_FILE.stat().st_size / (1 << 20)
        print(f"already have {OUT_FILE} ({size_mb:.1f} MB) — skipping download")
        return
    print(f"downloading {URL}")
    r = requests.get(URL, timeout=600, stream=True)
    r.raise_for_status()
    n_bytes = 0
    with open(OUT_FILE, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                n_bytes += len(chunk)
    print(f"wrote {OUT_FILE}  ({n_bytes / (1 << 20):.1f} MB)")


def _name(pid: int) -> str:
    return PDG_NAMES.get(abs(pid), f"pdg={pid}")


def _kin(p) -> tuple[float, float, float]:
    """Return (pT, η, φ) of a pyhepmc particle."""
    px, py, pz = p.momentum.px, p.momentum.py, p.momentum.pz
    pt = math.hypot(px, py)
    p_mag = math.hypot(pt, pz)
    if p_mag == abs(pz):
        eta = math.inf if pz > 0 else -math.inf
    else:
        eta = 0.5 * math.log((p_mag + pz) / (p_mag - pz))
    phi = math.atan2(py, px)
    return pt, eta, phi


def inspect():
    print(f"\nparsing {OUT_FILE} …\n")
    with pyhepmc.open(OUT_FILE) as f:
        for ev_idx, event in enumerate(f):
            eid = event.event_number
            parts = event.particles
            # find the Higgs decay
            higgs_daughters = []
            for p in parts:
                if p.pid == 25 and p.end_vertex:
                    out = [c for c in p.end_vertex.particles_out if c.pid != 25]
                    if out:
                        higgs_daughters = out
                        break

            # status-1 ("stable" in HepMC sense, what gets handed to Geant4)
            stable = [p for p in parts if p.status == 1]
            tot_visible_pt = sum(_kin(p)[0] for p in stable if abs(p.pid) not in (12, 14, 16))
            n_stable = len(stable)

            # leading-pT stable particle
            if stable:
                lead = max(stable, key=lambda p: _kin(p)[0])
                lp_pt, lp_eta, lp_phi = _kin(lead)
                lead_str = f"lead={_name(lead.pid)}({lead.pid:+d}) pT={lp_pt:.2f} η={lp_eta:+.2f} φ={lp_phi:+.2f}"
            else:
                lead_str = "no stable particles"

            print(f"=== event_number={eid}  (file index {ev_idx}) ===")
            print(f"  n_particles total = {len(parts)},  n_status==1 = {n_stable}")
            print(f"  Σ visible pT (no ν) = {tot_visible_pt:.1f} GeV")
            print(f"  {lead_str}")
            if not higgs_daughters:
                print("  no Higgs decay found")
            else:
                names = " → ".join(_name(p.pid) for p in higgs_daughters)
                print(f"  Higgs decay: {names}")
                for d in higgs_daughters:
                    pt, eta, phi = _kin(d)
                    print(f"    pdg={d.pid:+d}  pT={pt:7.2f}  η={eta:+.2f}  φ={phi:+.2f}")
            print()

            if ev_idx >= 9:
                # print only the first 10 events to keep output readable
                remaining = sum(1 for _ in f)
                print(f"… plus {remaining} more events in this file (total "
                      f"{ev_idx + 1 + remaining}).")
                break


def main():
    download()
    inspect()


if __name__ == "__main__":
    main()
