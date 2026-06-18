"""
H→bb̄ dijet-mass analysis using model predictions from an H5 file.

For each event present BOTH in the H5 (model predictions) and in the bb̄
labels parquets:

  1. Find the Higgs decay family from raw HF particles via the
     strategy_n_orphan_groups cascade (n=1..4) — same method as
     inspect_b_brothers.py / inspect_hbb_top_bottoms.py.
  2. Pick the two highest-E B-hadrons FROM INSIDE that family — these
     are the truth directions of the two b quarks.
  3. Two parallel jet-clustering passes (anti-kt R=0.4):
        (a) PFLOW jets clustered from the H5 MODEL predictions
            (pred_pt, pred_eta, pred_phi)
        (b) TRUTH jets clustered from the H5 truth-target particles
            (truth_pt, truth_eta, truth_phi)
  4. Ghost-associate each set of jets to the two B-hadrons (ΔR<0.4).
  5. Compute the dijet invariant mass for both passes.
  6. Summarise pred-vs-truth side by side.

The H5's `event_number` field is the global HF event_id, matching the
`event_id` column written by classify_hf_decay_channels.py into the
per-file label parquets.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/storage/agrp/barakma/PileupODD")

from load_higgs_diphoton_events import DEFAULT_COLUMNS, load_events  # noqa: E402
from classify_hf_decay_channels import DEFAULT_LABEL_DIR  # noqa: E402
from inspect_hbb_top_bottoms import (  # noqa: E402
    HIGGS_MASS, JET_R, JET_ALGO, MIN_CONSTITUENTS, MIN_JET_PT, JET_ETA_CUT,
    DR_MATCH,
    cluster_jets_event, match_b_to_jets, _delta_r, _wrap_phi,
    find_higgs_family, top2_b_hadrons_in_family,
    get_neutrinos_vp1, sum_neutrinos_in_cone,
)

DEFAULT_H5 = ("/storage/agrp/barakma/hepattn/src/hepattn/experiments/"
              "odd_pileup_reco/logs/odd_pflow_reco_20260519-T142453/"
              "ckpts/epoch=073-val_loss=13.67910__test_ggft_pu200_test.h5")
NUM_CLASSES = 6   # 0..NUM_CLASSES-2 = real object classes; NUM_CLASSES-1 = "no object"

# ΔR cuts for ghost-matching B-hadrons to jets — one dijet mass per cut.
DR_MATCH_VALUES = (0.4, 0.5, 0.6, 0.7)


# ───────────────────────── H5 loading ─────────────────────────────────────

def load_h5_pflow_and_truth(h5_path: Path) -> dict[int, dict]:
    """Return {event_number → {"pred": (pt,eta,phi), "truth": (pt,eta,phi)}}.

    For each H5 event we extract two sets of "particles":
      - PFLOW (model) — valid if pflow_class < NUM_CLASSES-1
      - TRUTH (target) — valid if object_class < NUM_CLASSES-1
    """
    print(f"reading H5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        event_numbers = f["events"]["event_number"][:]
        pflow_class = f["object_class"]["pflow_class"][:]
        truth_class = f["object_class"]["object_class"][:]
        pred_pt     = f["regression"]["pred_pt"][:]
        pred_eta    = f["regression"]["pred_eta"][:]
        pred_sinphi = f["regression"]["pred_sinphi"][:]
        pred_cosphi = f["regression"]["pred_cosphi"][:]
        truth_pt    = f["regression"]["truth_pt"][:]
        truth_eta   = f["regression"]["truth_eta"][:]
        truth_sinphi = f["regression"]["truth_sinphi"][:]
        truth_cosphi = f["regression"]["truth_cosphi"][:]

    pred_phi  = np.arctan2(pred_sinphi,  pred_cosphi)
    truth_phi = np.arctan2(truth_sinphi, truth_cosphi)

    out: dict[int, dict] = {}
    for i, en in enumerate(event_numbers):
        pmask = pflow_class[i] < (NUM_CLASSES - 1)
        tmask = truth_class[i] < (NUM_CLASSES - 1)
        out[int(en)] = {
            "pred":  (pred_pt[i, pmask].astype(np.float64),
                      pred_eta[i, pmask].astype(np.float64),
                      pred_phi[i, pmask].astype(np.float64)),
            "truth": (truth_pt[i, tmask].astype(np.float64),
                      truth_eta[i, tmask].astype(np.float64),
                      truth_phi[i, tmask].astype(np.float64)),
        }
    return out


# ───────────────────────── matching to bb̄ labels ────────────────────────

def select_bb_events_in_h5(label_dir: Path, h5_event_numbers: set[int],
                           n_keep: int) -> dict[int, list[int]]:
    """Read all labels_file_*.parquet, keep rows where channel == "bb̄" AND
    event_id is in the H5, sort, take first n_keep, return file→[event_ids]."""
    paths = sorted(label_dir.glob("labels_file_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no labels in {label_dir}")
    df = (
        pl.concat([pl.read_parquet(p) for p in paths])
        .filter(pl.col("channel") == "bb̄")
        .filter(pl.col("event_id").is_in(list(h5_event_numbers)))
        .sort("file_idx", "event_id")
        .head(n_keep)
    )
    out: dict[int, list[int]] = defaultdict(list)
    for fi, eid, _ in df.iter_rows():
        out[int(fi)].append(int(eid))
    return dict(out)


# ───────────────────────── per-event analysis ────────────────────────────

def analyze_event(eid: int, raw_row: dict, h5_event: dict) -> dict | None:
    """For one event, identify family + top-2 truth B-hadrons, cluster
    PFLOW + TRUTH H5 particles into jets, ghost-match both, compute dijet
    masses. Returns a result dict or None if the event can't be analysed."""
    family = find_higgs_family(raw_row)
    if family is None:
        print(f"  ✗ event {eid}: no Higgs family identified — skipping")
        return None
    bhads = top2_b_hadrons_in_family(raw_row, family["particle_idxs"])
    if len(bhads) < 2:
        print(f"  ✗ event {eid}: family has {len(bhads)} B-hadron(s) — skipping")
        return None
    print(f"  ✓ event {eid}: family via n={family['strategy']}  "
          f"(M={family['mass']:.4f}, N_family={len(family['particle_idxs'])})")
    for k, b in enumerate(bhads, 1):
        print(f"    bH#{k}  pdg={b['pdg']:+5d}  pT={b['pt']:7.2f}  "
              f"η={b['eta']:+.2f}  φ={b['phi']:+.2f}  E={b['E']:7.2f}")

    result: dict = {"event_id": eid, "strategy": family["strategy"],
                    "family_mass": family["mass"]}

    # Pull truth neutrinos (vp=1, |pdg|∈{12,14,16}) once per event. The same
    # truth-ν 4-momenta are added to both pred and truth jets — that's the
    # "ideal MET-allocated-to-jets" correction for both paths.
    nus = get_neutrinos_vp1(raw_row)

    for label in ("pred", "truth"):
        pt, eta, phi = h5_event[label]
        # Initialise per-ΔR result keys to None so downstream code can rely
        # on every key existing even if this pass skips.
        for dr in DR_MATCH_VALUES:
            result[f"{label}_mjj_dr{dr}"]    = None
            result[f"{label}_mjj_nu_dr{dr}"] = None
        if len(pt) < MIN_CONSTITUENTS:
            print(f"    [{label:5s}] < {MIN_CONSTITUENTS} particles → "
                  f"no jets, skipping")
            continue
        jets = cluster_jets_event(pt, eta, phi)
        for j in jets:
            j["phi"] = _wrap_phi(j["phi"])
        print(f"    [{label:5s}] {len(pt)} H5 particles → {len(jets)} jets")
        # One match + dijet mass per ΔR cut. Jets are clustered ONCE; only
        # the ghost-match radius changes. We also compute a ν-corrected
        # variant: add in-cone (ΔR < JET_R) truth neutrinos to each
        # matched jet's 4-momentum before computing M(jj).
        mass_str    = []
        mass_str_nu = []
        for dr_cut in DR_MATCH_VALUES:
            matches = match_b_to_jets(bhads, jets, dr_cut=dr_cut)
            if (len(matches) == 2 and all(m is not None for m in matches)
                    and matches[0] != matches[1]):
                j1, j2 = jets[matches[0]], jets[matches[1]]
                # visible dijet mass
                E_s  = j1["E"]  + j2["E"]
                px_s = j1["px"] + j2["px"]
                py_s = j1["py"] + j2["py"]
                pz_s = j1["pz"] + j2["pz"]
                mjj = float(math.sqrt(max(
                    E_s * E_s - (px_s * px_s + py_s * py_s + pz_s * pz_s), 0.0
                )))
                result[f"{label}_mjj_dr{dr_cut}"] = mjj
                mass_str.append(f"@ΔR<{dr_cut}: {mjj:7.2f}")

                # ν-corrected: add the in-cone (ΔR<JET_R) ν 4-momentum
                # of each matched jet's axis to that jet.
                nu1 = sum_neutrinos_in_cone(nus, j1["eta"], j1["phi"], dR=JET_R)
                nu2 = sum_neutrinos_in_cone(nus, j2["eta"], j2["phi"], dR=JET_R)
                E_c  = E_s  + nu1["E"]  + nu2["E"]
                px_c = px_s + nu1["px"] + nu2["px"]
                py_c = py_s + nu1["py"] + nu2["py"]
                pz_c = pz_s + nu1["pz"] + nu2["pz"]
                mjj_nu = float(math.sqrt(max(
                    E_c * E_c - (px_c * px_c + py_c * py_c + pz_c * pz_c), 0.0
                )))
                result[f"{label}_mjj_nu_dr{dr_cut}"] = mjj_nu
                mass_str_nu.append(f"@ΔR<{dr_cut}: {mjj_nu:7.2f}")
            else:
                mass_str.append(f"@ΔR<{dr_cut}:    ---")
                mass_str_nu.append(f"@ΔR<{dr_cut}:    ---")
        print(f"       → dijet M(bj,b̄j) per ΔR:        " + "   ".join(mass_str))
        print(f"       → dijet M+ν per ΔR (truth ν):   " + "   ".join(mass_str_nu))
    return result


# ───────────────────────── main ─────────────────────────────────────────

def _stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {}
    return {
        "n":      int(arr.size),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "in_5":   int((np.abs(arr - HIGGS_MASS) < 5).sum()),
        "in_10":  int((np.abs(arr - HIGGS_MASS) < 10).sum()),
        "in_20":  int((np.abs(arr - HIGGS_MASS) < 20).sum()),
    }


def print_summary(results: list[dict]) -> None:
    print(f"\n=== summary across {len(results)} events ===")
    for dr in DR_MATCH_VALUES:
        pred = np.array([r[f"pred_mjj_dr{dr}"]  for r in results
                         if r.get(f"pred_mjj_dr{dr}")  is not None])
        trut = np.array([r[f"truth_mjj_dr{dr}"] for r in results
                         if r.get(f"truth_mjj_dr{dr}") is not None])
        sP, sT = _stats(pred), _stats(trut)
        print(f"\n  ── ΔR < {dr} ──────────────────────────────")
        print(f"  {'':<18}  {'pred jets':>11}  {'truth jets':>11}")
        print(f"  " + "-" * 44)
        for key, label in [("n",      "n events"),
                           ("mean",   "mean M(bb̄)"),
                           ("median", "median"),
                           ("std",    "std"),
                           ("min",    "min"),
                           ("max",    "max")]:
            p = sP.get(key, "—"); t = sT.get(key, "—")
            if isinstance(p, float): p = f"{p:.2f}"
            if isinstance(t, float): t = f"{t:.2f}"
            print(f"    {label:<16}  {str(p):>11}  {str(t):>11}")
        for s, lab in ((sP, "pred"), (sT, "truth")):
            if not s:
                continue
            n = s["n"]
            print(f"    {lab:<7} within ±5/±10/±20 of {HIGGS_MASS}: "
                  f"{s['in_5']}/{n} ({s['in_5']/n:.0%}) / "
                  f"{s['in_10']}/{n} ({s['in_10']/n:.0%}) / "
                  f"{s['in_20']}/{n} ({s['in_20']/n:.0%})")


def plot_dijet_histograms(results: list[dict], out_path: Path,
                          mass_key_fmt: str = "{source}_mjj_dr{dr}",
                          title_suffix: str = "") -> None:
    """Two-panel figure (pred top, truth bottom) with one overlaid step
    histogram per ΔR cut, plus a vertical line at MH = 125 GeV.

    `mass_key_fmt` selects which per-event mass to plot. Defaults to the
    visible dijet mass. Pass "{source}_mjj_nu_dr{dr}" for the ν-corrected
    variant.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 250.0, 51)  # 5 GeV bins
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for ax, source in zip(axes, ("pred", "truth")):
        any_data = False
        for dr, color in zip(DR_MATCH_VALUES, colors):
            key = mass_key_fmt.format(source=source, dr=dr)
            arr = np.array([r[key] for r in results if r.get(key) is not None])
            if arr.size == 0:
                continue
            any_data = True
            label = (f"ΔR < {dr}  (N={arr.size}, mean={arr.mean():.1f}, "
                     f"med={np.median(arr):.1f}, std={arr.std():.1f})")
            ax.hist(arr, bins=bins, histtype="step", linewidth=2.0,
                    color=color, label=label)
        ax.axvline(HIGGS_MASS, color="crimson", linestyle="--", linewidth=1.4,
                   label=f"M_H = {HIGGS_MASS:.1f} GeV")
        ax.set_ylabel("events")
        title = (f"{'PFLOW (pred)' if source == 'pred' else 'truth target'} "
                 f"jets — H→bb̄ dijet mass per ΔR")
        if title_suffix:
            title += f"  {title_suffix}"
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        if not any_data:
            ax.text(0.5, 0.5, f"(no {source} dijet masses)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)

    axes[-1].set_xlabel(r"$M(b_j, \bar b_j)$  [GeV]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nsaved dijet-mass histogram → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, default=DEFAULT_H5,
                        help=f"Model H5 file (default: {DEFAULT_H5})")
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR),
                        help=f"bb̄ labels parquet dir (default: {DEFAULT_LABEL_DIR})")
    parser.add_argument("--n-events", type=int, default=1000,
                        help="How many bb̄ events to analyse (default: 1000)")
    parser.add_argument("--plot-out", type=str,
                        default="/storage/agrp/barakma/PileupODD/analysis/plots/"
                                "hbb_dijet_pred_vs_truth.png",
                        help="Output PNG for pred-vs-truth dijet-mass histogram")
    args = parser.parse_args()

    h5_data = load_h5_pflow_and_truth(Path(args.h5))
    h5_eids = set(h5_data.keys())
    print(f"H5: {len(h5_eids)} events  "
          f"(range {min(h5_eids)}..{max(h5_eids)})")

    file_to_events = select_bb_events_in_h5(
        Path(args.label_dir), h5_eids, args.n_events,
    )
    flat = sorted((fi, eid) for fi, eids in file_to_events.items() for eid in eids)
    print(f"selected {len(flat)} H→bb̄ events (in H5 AND labelled bb̄):")
    for fi, eid in flat:
        print(f"  file {fi:>4}  event {eid}")

    # Load raw HF particles per file so we can identify the Higgs family.
    all_results: list[dict] = []
    for fi in sorted(file_to_events):
        eids = file_to_events[fi]
        print(f"\n--- HF file {fi}: {len(eids)} events ---")
        particles = load_events({fi: eids}, kind="particles",
                                columns=DEFAULT_COLUMNS["particles"])
        raw_by_eid = {int(r["event_id"]): r
                      for r in particles.iter_rows(named=True)}
        for eid in eids:
            raw = raw_by_eid.get(eid)
            if raw is None:
                print(f"  event {eid}: not in HF after load — skipping")
                continue
            res = analyze_event(eid, raw, h5_data[eid])
            if res is not None:
                all_results.append(res)

    print_summary(all_results)
    plot_out = Path(args.plot_out)
    # Visible-only plot (existing behaviour)
    plot_dijet_histograms(all_results, plot_out,
                          mass_key_fmt="{source}_mjj_dr{dr}")
    # ν-corrected plot: same layout, but each jet has its in-cone truth
    # neutrinos added back to its 4-momentum before computing M(jj).
    nu_path = plot_out.with_name(plot_out.stem + "_nu" + plot_out.suffix)
    plot_dijet_histograms(all_results, nu_path,
                          mass_key_fmt="{source}_mjj_nu_dr{dr}",
                          title_suffix="(+ in-cone truth ν)")


if __name__ == "__main__":
    main()
