"""
Classify the Higgs decay channel of each HF event purely from its own
vp==1 primary particles. No external truth, no higgs_decays.parquet, no
NERSC HepMC.

Per-event decision rule (most-specific first; first match wins):

  γγ      : ≥2 photons at vp=1 with pT > 20 GeV, opening ΔR > 0.4,
            invariant mass within [100, 150] GeV
  ZZ→4l   : ≥4 isolated charged leptons (e/μ) with pT > 15 GeV at vp=1
  WW→2lν  : exactly 2 isolated leptons (e/μ) at vp=1 with pT > 15 GeV AND
            MET (from vp=1 ν's) > 20 GeV
  μμ      : exactly 2 muons pT > 20 GeV, M(μμ) ∈ [100, 150]
  ee      : exactly 2 electrons pT > 20 GeV, M(ee) ∈ [100, 150]
  ττ      : ≥1 τ⁺ AND ≥1 τ⁻ at vp=1 with pT > 15 GeV
  Zγ      : ≥1 γ pT > 20 GeV + ≥2 same-flavour leptons M(ll) ∈ [70, 110]
  bb̄      : ≥1 B-hadron (|pdg|∈{511,513,521,523,531,533,541,543,5122,5132,5232,5222,5212,5332,5142}) at vp=1 with pT > 15 GeV (cut kills soft ISR g→bb̄ splittings)
  cc̄      : ≥1 D-hadron (|pdg|∈{411,413,421,423,431,433,4122,4132,4232,4222,4212,4332}) at vp=1 with pT > 15 GeV (B-hadron veto via rule order; pT cut kills g→cc̄ splittings)
  gg/qq   : leftover (no specific lepton/photon/heavy-hadron signature)
  other   : event had no vp=1 primaries (shouldn't happen)

Then compares the resulting BR distribution to the SM expectation derived
from higgs_decays.parquet (which we already verified is SM-weighted).
"""

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_higgs_diphoton_events import (  # noqa: E402
    HF_RESOLVE, HS_EVENT_NAME, NUM_HF_REPO_FILES, DEFAULT_COLUMNS,
)

# Sentinel the worker prints on its last line so the manager can pick
# up the per-file channel counts from the subprocess stdout.
RESULT_SENTINEL = "RESULT::"

# Per-file label parquets land here. Each row: (file_idx, event_id, channel).
# Downstream tools (e.g. inspect_hbb_top_bottoms.py) glob this directory
# to filter events by channel and load just those event_ids from HF.
DEFAULT_LABEL_DIR = Path(
    "/storage/agrp/barakma/PileupODD/data/hf_decay_labels"
)

# B-hadrons: B/B*-mesons + Λb / Σb / Ξb / Ωb baryons
B_HADRON_PDGS = {511, 513, 521, 523, 531, 533, 541, 543,
                 5122, 5132, 5232, 5222, 5212, 5332, 5142}
# D-hadrons: D/D*-mesons + charmed baryons
D_HADRON_PDGS = {411, 413, 421, 423, 431, 433,
                 4122, 4132, 4232, 4222, 4212, 4332}
NU_PDGS = {12, 14, 16}

# Selection thresholds — tuned for SM ggf Higgs samples
PT_PHOTON      = 20.0
PT_LEPTON      = 15.0
PT_LEPTON_HIGH = 20.0
PT_TAU         = 15.0
PT_HEAVY       = 10.0
DR_DIPHOTON    = 0.4
M_HIGGS_LOW, M_HIGGS_HIGH = 100.0, 150.0
M_Z_LOW,     M_Z_HIGH     = 70.0,  110.0

CHANNELS = ("γγ", "ZZ→4l", "WW→2lν", "μμ", "ee", "ττ", "Zγ", "bb̄", "cc̄", "gg/qq", "other")


def _mass_two(px1, py1, pz1, E1, px2, py2, pz2, E2):
    Es = E1 + E2; pxs = px1 + px2; pys = py1 + py2; pzs = pz1 + pz2
    m2 = Es * Es - (pxs * pxs + pys * pys + pzs * pzs)
    return math.sqrt(max(m2, 0.0))


def _dR(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    while dphi >  math.pi: dphi -= 2 * math.pi
    while dphi < -math.pi: dphi += 2 * math.pi
    return math.hypot(eta1 - eta2, dphi)


def classify_event(row: dict) -> str:
    """Apply the decision rule to one HF event row."""
    pdg = np.asarray(row["pdg_id"], dtype=np.int64)
    px  = np.asarray(row["px"],     dtype=np.float64)
    py  = np.asarray(row["py"],     dtype=np.float64)
    pz  = np.asarray(row["pz"],     dtype=np.float64)
    E   = np.asarray(row["energy"], dtype=np.float64)
    vp  = np.asarray(row["vertex_primary"], dtype=np.int64)

    mask = vp == 1
    if not mask.any():
        return "other"
    pdg = pdg[mask]; px = px[mask]; py = py[mask]; pz = pz[mask]; E = E[mask]
    pt = np.hypot(px, py)
    
    # Highly optimized and safe eta calculation
    eta = np.arcsinh(pz / np.maximum(pt, 1e-30))
    phi = np.arctan2(py, px)

    # 1) γγ
    g_idx = np.where((pdg == 22) & (pt > PT_PHOTON))[0]
    if len(g_idx) >= 2:
        order = g_idx[np.argsort(pt[g_idx])[::-1]]
        # Loop pairs properly, without an early break that skips valid candidates
        for idx_a in range(len(order)):
            for idx_b in range(idx_a + 1, len(order)):
                a, b = order[idx_a], order[idx_b]
                if _dR(eta[a], phi[a], eta[b], phi[b]) > DR_DIPHOTON:
                    m = _mass_two(px[a], py[a], pz[a], E[a], px[b], py[b], pz[b], E[b])
                    if M_HIGGS_LOW <= m <= M_HIGGS_HIGH:
                        return "γγ"

    # ====================================================================
    # Physics Fix 1: Tracker-style isolation (ignore FSR γ and ν)
    # Prompt leptons from W/Z/H are isolated. Leptons from b/c decays are
    # surrounded by jets. But prompt leptons ALSO radiate FSR photons that
    # sit at ΔR<0.1; if we include them in the cone we kill real prompt
    # leptons too. So sum pT in the cone EXCLUDING photons + neutrinos,
    # and use ratio < 0.25 (a bit looser than the calo-isolation 0.15).
    # ====================================================================
    abs_pdg_all = np.abs(pdg)
    lep_mask = (np.isin(abs_pdg_all, [11, 13])) & (pt > PT_LEPTON)
    iso_lep_mask = np.zeros_like(lep_mask)
    if lep_mask.any():
        for i in np.where(lep_mask)[0]:
            dphi = phi - phi[i]
            dphi = np.where(dphi >  math.pi, dphi - 2 * math.pi, dphi)
            dphi = np.where(dphi < -math.pi, dphi + 2 * math.pi, dphi)
            dr2 = (eta - eta[i]) ** 2 + dphi ** 2
            # Exclude the lepton itself (dr2>1e-5 avoids float-noise),
            # exclude FSR photons (pdg=22) and neutrinos from the sum.
            cone_mask = ((dr2 > 1e-5) & (dr2 < 0.09)
                         & (abs_pdg_all != 22)
                         & ~np.isin(abs_pdg_all, list(NU_PDGS)))
            if np.sum(pt[cone_mask]) / pt[i] < 0.25:
                iso_lep_mask[i] = True
    n_iso_lep = int(iso_lep_mask.sum())

    # MET from neutrinos at vp=1
    nu_mask = np.isin(np.abs(pdg), list(NU_PDGS))
    met = math.hypot(float(px[nu_mask].sum()),
                     float(py[nu_mask].sum())) if nu_mask.any() else 0.0

    # 2) ZZ→4l
    if n_iso_lep >= 4:
        return "ZZ→4l"

    # 3) WW→2lν : 2 ISOLATED OS leptons + MET
    if n_iso_lep == 2 and met > 20.0:
        iso_idx = np.where(iso_lep_mask)[0]
        if pdg[iso_idx[0]] * pdg[iso_idx[1]] < 0:
            return "WW→2lν"

    # 4) μμ (isolated)
    mu_idx = np.where((np.abs(pdg) == 13) & (pt > PT_LEPTON_HIGH) & iso_lep_mask)[0]
    if len(mu_idx) == 2:
        a, b = mu_idx
        if pdg[a] * pdg[b] < 0:
            m = _mass_two(px[a], py[a], pz[a], E[a], px[b], py[b], pz[b], E[b])
            if M_HIGGS_LOW <= m <= M_HIGGS_HIGH:
                return "μμ"

    # 5) ee (isolated)
    e_idx = np.where((np.abs(pdg) == 11) & (pt > PT_LEPTON_HIGH) & iso_lep_mask)[0]
    if len(e_idx) == 2:
        a, b = e_idx
        if pdg[a] * pdg[b] < 0:
            m = _mass_two(px[a], py[a], pz[a], E[a], px[b], py[b], pz[b], E[b])
            if M_HIGGS_LOW <= m <= M_HIGGS_HIGH:
                return "ee"

    # 6) ττ
    has_tau_p = ((pdg == -15) & (pt > PT_TAU)).any()
    has_tau_m = ((pdg ==  15) & (pt > PT_TAU)).any()
    if has_tau_p and has_tau_m:
        return "ττ"

    # 7) Zγ : 1 γ + 2 SFOS ISOLATED leptons near M_Z
    g_loose = np.where((pdg == 22) & (pt > PT_PHOTON))[0]
    if len(g_loose) >= 1 and n_iso_lep >= 2:
        for flav in (11, 13):
            li = np.where((np.abs(pdg) == flav) & iso_lep_mask)[0]
            if len(li) >= 2:
                order = li[np.argsort(pt[li])[::-1]]
                for idx_a in range(len(order)):
                    for idx_b in range(idx_a + 1, len(order)):
                        a, b = order[idx_a], order[idx_b]
                        if pdg[a] * pdg[b] < 0:
                            m = _mass_two(px[a], py[a], pz[a], E[a], px[b], py[b], pz[b], E[b])
                            if M_Z_LOW <= m <= M_Z_HIGH:
                                return "Zγ"

    apdg = np.abs(pdg)

    # ====================================================================
    # Physics Fix 2: reject soft ISR g→bb̄ / g→cc̄ splittings
    # vp=1 includes ALL hard-interaction products including initial-state
    # radiation. ISR gluons occasionally split into soft b/c pairs at
    # pT~few GeV. Real H→bb̄/cc̄ produces hadrons at ~50 GeV, so a 15 GeV
    # cut cleanly separates the two without losing genuine signal.
    # ====================================================================
    # 8) bb̄ : ≥1 B-hadron at vp=1 with pT > 15 GeV
    has_b_hadron = bool((np.isin(apdg, list(B_HADRON_PDGS)) & (pt > 15.0)).any())
    if has_b_hadron:
        return "bb̄"

    # 9) cc̄ : ≥1 D-hadron at vp=1 with pT > 15 GeV (and no B-hadron — falls
    #    through naturally because bb̄ was checked first)
    c_had_mask = np.isin(apdg, list(D_HADRON_PDGS)) & (pt > 15.0)
    if c_had_mask.any():
        return "cc̄"

    # 10) gg/qq leftover
    return "gg/qq"


def classify_file(file_idx: int, n_events: int | None = None) -> list[dict]:
    """Scan one HF parquet, classify every event."""
    url = (f"{HF_RESOLVE}/data/{HS_EVENT_NAME}_particles/"
           f"train-{file_idx:05d}-of-{NUM_HF_REPO_FILES:05d}.parquet")
    print(f"loading {url}")
    lf = pl.scan_parquet(url).select(DEFAULT_COLUMNS["particles"])
    
    # Use head() rather than filtering by ID to cap events cleanly without full scan
    if n_events is not None:
        lf = lf.head(n_events)
        
    df = lf.collect()
    print(f"loaded {df.height} events; classifying ...")
    out = []
    for r in df.iter_rows(named=True):
        ch = classify_event(r)
        out.append({"file_idx": file_idx, "event_id": int(r["event_id"]),
                    "channel": ch})
    return out


def sm_br_table() -> dict[str, float]:
    """
    SM Higgs BRs (MH=125 GeV).

    Physics Fix 2: The topological categories (bb, cc, gg) naturally absorb
    hadronic WW and ZZ decays. The SM expectations must be adjusted to
    account for this physical fall-through so the expected fractions sum
    to 100%.
    """
    br_ww_tot      = 0.2159
    br_ww_2l       = br_ww_tot * (0.108 * 2) ** 2
    br_ww_hadronic = br_ww_tot - br_ww_2l       # ~20.5%

    br_zz_tot      = 0.0264
    br_zz_4l       = br_zz_tot * (0.067 ** 2) * 9
    br_zz_hadronic = br_zz_tot - br_zz_4l       # ~2.5%

    # W -> cs ~33% of hadronic W decays (produces D-hadrons → cc bucket)
    # Z -> bb ~15%, Z -> cc ~12%
    return {
        "bb̄":     0.5792 + (br_zz_hadronic * 0.15),
        "WW→2lν": br_ww_2l,
        "gg/qq":  0.0833 + (br_ww_hadronic * 0.55) + (br_zz_hadronic * 0.73),
        "ττ":     0.0623,
        "cc̄":     0.0287 + (br_ww_hadronic * 0.45) + (br_zz_hadronic * 0.12),
        "ZZ→4l":  br_zz_4l,
        "γγ":     0.0022,
        "Zγ":     0.0016,
        "μμ":     0.00022,
        "ee":     0.0,
        "other":  0.0,
    }


def print_distribution_from_counter(counter: Counter, n: int) -> None:
    sm = sm_br_table()
    print(f"\nclassification of {n} events")
    print(f"{'channel':<10}  {'count':>6}  {'frac':>7}  {'SM BR':>7}  {'Δ':>7}")
    print("-" * 52)
    for ch in CHANNELS:
        c = counter.get(ch, 0)
        frac = c / n if n else 0.0
        ref = sm.get(ch, 0.0)
        print(f"{ch:<10}  {c:>6d}  {frac:>7.2%}  {ref:>7.2%}  {frac-ref:>+7.2%}")
    for k in set(counter) - set(CHANNELS):
        print(f"{k:<10}  {counter[k]:>6d}  {counter[k]/n:>7.2%}  (unexpected key)")


def _label_path(label_dir: Path, file_idx: int) -> Path:
    return label_dir / f"labels_file_{file_idx:05d}.parquet"


def manager_loop(files: list[int], n_events: int | None,
                 label_dir: Path, skip_existing: bool) -> None:
    """Process each file in its own subprocess; aggregate + print after each.

    Each worker subprocess writes its (event_id, channel) labels to
    {label_dir}/labels_file_NNNNN.parquet, and prints a JSON counter line
    tagged with RESULT_SENTINEL. The manager accumulates counts in memory
    and prints the running aggregate. Resumable: completed files are
    skipped on re-run because their parquet already exists.
    """
    label_dir.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    cum_counter: Counter = Counter()
    cum_n = 0

    # Seed the running counter with any previously-classified files so the
    # printed aggregate keeps the same denominator as the per-event parquets.
    if skip_existing:
        for fi in files:
            p = _label_path(label_dir, fi)
            if p.exists():
                ch = pl.read_parquet(p)["channel"].to_list()
                cum_counter.update(ch)
                cum_n += len(ch)

    for i, fi in enumerate(files, 1):
        out_path = _label_path(label_dir, fi)
        if out_path.exists() and skip_existing:
            print(f"\n[{i}/{len(files)}] file {fi}: {out_path.name} exists, "
                  f"skipping (use --force to re-classify)")
            print(f"\n--- running aggregate after file {fi} ---")
            print_distribution_from_counter(cum_counter, cum_n)
            continue

        cmd = [sys.executable, "-u", script,
               "--worker", "--file-idx", str(fi),
               "--output-path", str(out_path)]
        if n_events is not None:
            cmd += ["--n-events", str(n_events)]
        print(f"\n[{i}/{len(files)}] spawning subprocess for file {fi} ...")
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0

        # forward worker stdout/stderr so the user still sees what happened
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)

        if result.returncode != 0:
            print(f"  ! subprocess for file {fi} exited {result.returncode} ({dt:.1f}s)")
            continue

        worker_counts = None
        for line in result.stdout.splitlines():
            if line.startswith(RESULT_SENTINEL):
                try:
                    worker_counts = json.loads(line[len(RESULT_SENTINEL):])
                except json.JSONDecodeError as e:
                    print(f"  ! could not parse worker result line: {e}")
                break
        if worker_counts is None:
            print(f"  ! file {fi} subprocess produced no RESULT line — skipping")
            continue

        cum_counter.update(worker_counts)
        cum_n += sum(worker_counts.values())
        print(f"  ✓ file {fi} done in {dt:.1f}s   wrote {out_path}")
        print(f"\n--- running aggregate after file {fi} ---")
        print_distribution_from_counter(cum_counter, cum_n)


def worker_main(file_idx: int, n_events: int | None,
                output_path: Path) -> None:
    """Worker: classify one file, write labels parquet, print counter line."""
    records = classify_file(file_idx, n_events=n_events)
    pl.DataFrame(records).write_parquet(output_path)
    counter = Counter(r["channel"] for r in records)
    print(f"wrote {output_path} ({len(records)} rows)")
    print(f"{RESULT_SENTINEL}{json.dumps(dict(counter))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, nargs="+", default=list(range(1000)),
                        help="HF file indices to classify (default: 0..59)")
    parser.add_argument("--n-events", type=int, default=None,
                        help="Cap events per file (default: all)")
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR),
                        help=f"Where to write per-file label parquets "
                             f"(default: {DEFAULT_LABEL_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-classify files even if their labels parquet exists")
    parser.add_argument("--worker", action="store_true",
                        help="(internal) Worker mode: classify --file-idx and exit")
    parser.add_argument("--file-idx", type=int, default=None,
                        help="(worker mode) file index to classify")
    parser.add_argument("--output-path", type=str, default=None,
                        help="(worker mode) where to write the labels parquet")
    args = parser.parse_args()

    if args.worker:
        assert args.file_idx is not None and args.output_path is not None, \
            "--worker requires --file-idx and --output-path"
        worker_main(args.file_idx, args.n_events, Path(args.output_path))
        return

    manager_loop(list(args.files), args.n_events,
                 Path(args.label_dir), skip_existing=not args.force)


if __name__ == "__main__":
    main()