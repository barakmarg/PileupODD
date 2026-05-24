"""
Load only H -> γγ events from the ColliderML ggf_pu200 dataset on
HuggingFace, without downloading the full ~1000-file release.

event_ids in higgs_decays.parquet are PER-RUN (each run restarts at 0),
so we need (run, event_id) pairs to identify an event. The `run` value
maps directly to the HF file index.

ID mapping (the part that took us 5 iterations to nail down):
- higgs_decays.parquet has (run, event_id) where event_id is per-run, with
  64 events per run.
- HF ggf_pu200_particles files use global event_id with 100 events per file.
- So: global_eid = run * 64 + per_run_eid; file_idx = global_eid // 100.

Performance:
- Polars' native `pl.scan_parquet(url)` over the HF resolve URL fetches the
  parquet footer, applies row-group min/max stats on event_id (predicate
  pushdown), and only downloads requested columns from surviving groups
  (projection pushdown). No disk cache.
- Sequential with tqdm; one scan per file pulls ALL γγ events that map to
  that file at once via `event_id.is_in([...])`.
"""

import argparse
import time
from collections import defaultdict

import polars as pl
import tqdm

DECAYS_PATH = "/storage/agrp/barakma/PileupODD/data/higgs_decays.parquet"
REPO_ID = "CERN/ColliderML-Release-1"
HF_RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
HS_EVENT_NAME = "ggf_pu200"
NUM_HF_REPO_FILES = 1000   # ColliderML release uses 1000-file shards
HF_EVENTS_PER_FILE = 100   # HF event_ids are global; file_idx = global_eid // 100
DECAYS_EVENTS_PER_RUN = 64 # higgs_decays event_id is per-run; runs have 64 events each
# Mapping: global_eid = run * DECAYS_EVENTS_PER_RUN + per_run_eid
#          file_idx   = global_eid // HF_EVENTS_PER_FILE

KIND_CHOICES = ("particles", "calo_hits", "tracks")

# Column sets the existing training pipeline downloads from HF — keep parity
# with primary/create_trainning_dataset_pileup.py so we pull the same payload.
DEFAULT_COLUMNS: dict[str, list[str] | None] = {
    "particles": [
        "event_id", "particle_id", "vertex_primary", "pdg_id",
        "energy", "px", "py", "pz", "vx", "vy", "vz", "parent_id",
    ],
    "calo_hits": [
        "event_id", "detector", "total_energy", "x", "y", "z",
        "contrib_particle_ids", "contrib_energies", "contrib_times",
    ],
    "tracks": None,  # the pipeline reads tracks with all columns
}


def find_gamma_gamma_events() -> dict[int, list[int]]:
    """Return {file_idx -> [global event_ids]} for H -> γγ events.

    higgs_decays event_ids are PER-RUN with 64 events per run. HF event_ids
    are global with 100 events per file. So:
        global_eid = run * 64 + per_run_eid
        file_idx   = global_eid // 100
    Events whose global_eid >= NUM_HF_REPO_FILES * HF_EVENTS_PER_FILE have
    no corresponding HF file and are dropped with a warning.
    """
    pairs = (
        pl.read_parquet(DECAYS_PATH)
        .filter(
            (pl.col("out_pids").list.len() == 2)
            & pl.col("out_pids").list.eval(pl.element().abs() == 22).list.all()
        )
        .select("run", "event_id")
        .with_columns(
            (pl.col("run") * DECAYS_EVENTS_PER_RUN + pl.col("event_id"))
            .alias("global_eid")
        )
        .with_columns(
            (pl.col("global_eid") // HF_EVENTS_PER_FILE).alias("file_idx")
        )
        .sort("global_eid")
    )

    max_global = NUM_HF_REPO_FILES * HF_EVENTS_PER_FILE
    in_range = pairs.filter(pl.col("global_eid") < max_global)
    dropped = pairs.height - in_range.height
    if dropped:
        print(f"warning: {dropped} γγ events have global_eid >= {max_global} "
              f"(beyond the {NUM_HF_REPO_FILES} HF files) — dropped")

    out: dict[int, list[int]] = defaultdict(list)
    for file_idx, global_eid in in_range.select("file_idx", "global_eid").iter_rows():
        out[int(file_idx)].append(int(global_eid))
    return dict(out)


def _file_url(file_idx: int, kind: str) -> str:
    return (
        f"{HF_RESOLVE}/data/{HS_EVENT_NAME}_{kind}/"
        f"train-{file_idx:05d}-of-{NUM_HF_REPO_FILES:05d}.parquet"
    )


def _fetch_one(
    file_idx: int,
    event_ids: list[int],
    kind: str,
    columns: list[str] | None,
) -> tuple[int, pl.DataFrame, float]:
    """Native Polars HTTP scan with predicate + projection pushdown to HF."""
    url = _file_url(file_idx, kind)
    t0 = time.perf_counter()
    lf = pl.scan_parquet(url).filter(pl.col("event_id").is_in(event_ids))
    if columns is not None:
        lf = lf.select(columns)
    df = lf.collect()
    return file_idx, df, time.perf_counter() - t0


def load_events(
    file_to_events: dict[int, list[int]],
    kind: str = "particles",
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Sequentially scan one HF parquet per file, filter to matching event rows.

    Each file is scanned only once even if multiple requested events live in
    it — `event_id.is_in([...])` pulls them all in a single pass. tqdm tracks
    both files scanned and total events loaded.
    """
    file_idxs = sorted(file_to_events)
    total_events = sum(len(v) for v in file_to_events.values())
    frames = []
    events_loaded = 0
    pbar = tqdm.tqdm(file_idxs, desc=f"scanning {kind}", unit="file")
    for file_idx in pbar:
        eids = file_to_events[file_idx]
        _, df, t_scan = _fetch_one(file_idx, eids, kind, columns)
        frames.append(df)
        events_loaded += df.height
        pbar.set_postfix(
            events=f"{events_loaded}/{total_events}",
            req=len(eids),
            got=df.height,
            last_s=f"{t_scan:.2f}",
        )
    return pl.concat(frames) if frames else pl.DataFrame()


def sanity_check_first_file(kind: str = "particles") -> None:
    """Scan one HF file (event_id column only) and print id range + timing."""
    url = _file_url(0, kind)
    print(f"sanity: scanning {url}")
    t0 = time.perf_counter()
    info = (
        pl.scan_parquet(url)
        .select(
            pl.len().alias("n_rows"),
            pl.col("event_id").min().alias("eid_min"),
            pl.col("event_id").max().alias("eid_max"),
            pl.col("event_id").n_unique().alias("n_unique_eid"),
        )
        .collect()
    )
    print(f"  -> {info.row(0, named=True)}  ({time.perf_counter() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="particles", choices=KIND_CHOICES,
                        help="Which object table to load (default: particles)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Cap total events loaded (good for a first test)")
    parser.add_argument("--columns", nargs="+", default=None,
                        help="Subset of columns to fetch (default: same set "
                             "as create_trainning_dataset_pileup.py for this kind)")
    parser.add_argument("--sanity", action="store_true",
                        help="Scan one HF file unfiltered and print event_id stats, then exit")
    args = parser.parse_args()

    if args.sanity:
        sanity_check_first_file(args.kind)
        return

    file_to_events = find_gamma_gamma_events()
    total = sum(len(v) for v in file_to_events.values())
    print(f"H -> γγ events in higgs_decays.parquet: "
          f"{total} across {len(file_to_events)} HF files")

    if args.limit is not None:
        trimmed, budget = {}, args.limit
        for file_idx in sorted(file_to_events):
            if budget <= 0:
                break
            take = file_to_events[file_idx][:budget]
            trimmed[file_idx] = take
            budget -= len(take)
        file_to_events = trimmed
        total = sum(len(v) for v in file_to_events.values())
        print(f"capped at {total} events across {len(file_to_events)} files (--limit)\n")

    columns = args.columns if args.columns is not None else DEFAULT_COLUMNS[args.kind]
    print(f"columns to fetch: {columns if columns else 'ALL'}")

    t0 = time.perf_counter()
    df = load_events(file_to_events, kind=args.kind, columns=columns)
    dt = time.perf_counter() - t0
    print(f"\nloaded {df.height} event-rows of {args.kind} in {dt:.1f}s")
    print(f"full df shape: {df.shape}")
    print(f"columns: {df.columns}")
    print(f"event_ids loaded: {df['event_id'].to_list()}")
    print("\npreview (first 7 rows):")
    print(df.head(7))


if __name__ == "__main__":
    main()
