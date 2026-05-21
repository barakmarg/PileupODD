"""
Download HepMC truth files for runs 0..2098 of the ColliderML full_pileup/ggf/v1
dataset, parse each event's Higgs decay channel, and save all results to a
single parquet using polars.

- HepMC files are NOT cached: download to a temp file, parse, delete.
- Every 50 successful runs the parquet is checkpointed so an interrupted run
  can be resumed: on startup we read the existing parquet, find the max run,
  and continue from the next run.
"""

import argparse
import tempfile
from pathlib import Path

import polars as pl
import pyhepmc
import requests
import tqdm

URL_TEMPLATE = (
    "https://portal.nersc.gov/cfs/m4958/ColliderML/full_pileup/ggf/v1/"
    "runs/{run}/events.hepmc"
)
OUT_PATH = Path("/storage/agrp/barakma/PileupODD/data/higgs_decays.parquet")

PDG_NAMES = {
    5: "b", 22: "gamma", 23: "Z", 24: "W", 15: "tau", 21: "g",
    13: "mu", 11: "e", 12: "nu_e", 14: "nu_mu", 16: "nu_tau",
}

PARQUET_SCHEMA = {
    "run": pl.UInt32,
    "event_id": pl.UInt32,
    "decay": pl.String,
    "out_pids": pl.List(pl.Int64),
}


def _download_to(run: int, dest: Path) -> bool:
    url = URL_TEMPLATE.format(run=run)
    try:
        r = requests.get(url, timeout=180, stream=True)
        if r.status_code != 200:
            print(f"  run {run}: HTTP {r.status_code} — skipping")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"  run {run}: request failed ({e}) — skipping")
        return False


def _parse_run(run: int, path: Path) -> list[dict]:
    rows = []
    with pyhepmc.open(path) as f:
        for event in f:
            eid = event.event_number
            decay_str = "Unknown"
            out_pids: list[int] = []
            for p in event.particles:
                if p.pid == 25 and p.end_vertex:
                    out_pids = [part.pid for part in p.end_vertex.particles_out
                                if part.pid != 25]
                    if out_pids:
                        names = [PDG_NAMES.get(abs(pid), str(pid)) for pid in out_pids]
                        decay_str = " -> ".join(names)
                        break
            rows.append({
                "run": run,
                "event_id": int(eid),
                "decay": decay_str,
                "out_pids": out_pids,
            })
    return rows


def _flush(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    pl.DataFrame(rows, schema=PARQUET_SCHEMA).write_parquet(path, compression="zstd")


def _resume_state(path: Path) -> tuple[list[dict], int]:
    """Return (existing rows as dicts, max run already processed or -1)."""
    if not path.exists():
        return [], -1
    df = pl.read_parquet(path)
    if df.is_empty():
        return [], -1
    max_run = int(df["run"].max())
    rows = df.to_dicts()
    print(f"resuming: {len(rows)} rows already in {path}, max run = {max_run}")
    return rows, max_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--run-end", type=int, default=2098, help="inclusive")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows, max_done = _resume_state(OUT_PATH)
    start = max(args.run_start, max_done + 1)
    if start > args.run_end:
        print("nothing to do — parquet already covers the requested range")
        return

    n_ok = n_fail = 0
    pbar = tqdm.tqdm(range(start, args.run_end + 1), desc="runs")
    with tempfile.TemporaryDirectory(prefix="hepmc_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for run in pbar:
            tmp_file = tmpdir_path / f"run{run}.hepmc"
            ok = _download_to(run, tmp_file)
            if not ok:
                n_fail += 1
                tmp_file.unlink(missing_ok=True)
                continue
            try:
                rows.extend(_parse_run(run, tmp_file))
                n_ok += 1
            except Exception as e:
                print(f"  run {run}: parse error ({e}) — skipping")
                n_fail += 1
            finally:
                tmp_file.unlink(missing_ok=True)

            if args.checkpoint_every and n_ok > 0 and n_ok % args.checkpoint_every == 0:
                _flush(rows, OUT_PATH)
                pbar.set_postfix(events=len(rows), ok=n_ok, fail=n_fail)

    _flush(rows, OUT_PATH)
    print(f"\nruns ok={n_ok}, failed={n_fail}, total events={len(rows)}")
    print(f"wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
