"""
Focused debug: pick the first H -> γγ event, locate its file, and try to
fetch just that one event a few different ways. Print everything so we
can see exactly where the filter is failing.
"""

import time
import polars as pl

DECAYS_PATH = "/storage/agrp/barakma/PileupODD/data/higgs_decays.parquet"
REPO_ID = "CERN/ColliderML-Release-1"
HF_RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
HS_EVENT_NAME = "ggf_pu200"
NUM_HF_REPO_FILES = 1000


def main():
    # Target: event_id=63 in file 00000 (HF event_ids are global; file 0 covers 0..99)
    run = 0
    eid = 63
    print(f"target: file {run:05d}, event_id={eid}")

    url = (f"{HF_RESOLVE}/data/{HS_EVENT_NAME}_particles/"
           f"train-{run:05d}-of-{NUM_HF_REPO_FILES:05d}.parquet")
    print(f"\nfile URL: {url}\n")

    # 2. NO FILTER -- pull only event_id column and inspect
    print("=== step 1: read event_id column unfiltered ===")
    t0 = time.perf_counter()
    eids_df = pl.scan_parquet(url).select("event_id").collect()
    print(f"  fetched in {time.perf_counter() - t0:.1f}s")
    print(f"  schema: {eids_df.schema}")
    print(f"  n_rows: {eids_df.height}")
    print(f"  min={eids_df['event_id'].min()}, max={eids_df['event_id'].max()}")
    print(f"  first 10: {eids_df['event_id'].head(10).to_list()}")
    print(f"  target {eid} in file? {eid in eids_df['event_id'].to_list()}")

    # 3. FILTER via is_in
    print(f"\n=== step 2: filter event_id.is_in([{eid}]) ===")
    t0 = time.perf_counter()
    df_isin = (
        pl.scan_parquet(url)
        .filter(pl.col("event_id").is_in([eid]))
        .select("event_id")
        .collect()
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s, rows: {df_isin.height}")

    # 4. FILTER via equality
    print(f"\n=== step 3: filter event_id == {eid} ===")
    t0 = time.perf_counter()
    df_eq = (
        pl.scan_parquet(url)
        .filter(pl.col("event_id") == eid)
        .select("event_id")
        .collect()
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s, rows: {df_eq.height}")

    # 5. FILTER via casted is_in
    print(f"\n=== step 4: filter is_in with explicit u32 cast ===")
    t0 = time.perf_counter()
    target = pl.Series("t", [eid], dtype=pl.UInt32)
    df_cast = (
        pl.scan_parquet(url)
        .filter(pl.col("event_id").is_in(target))
        .select("event_id")
        .collect()
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s, rows: {df_cast.height}")


if __name__ == "__main__":
    main()
