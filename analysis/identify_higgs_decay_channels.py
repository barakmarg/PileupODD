"""
Per-process event counts from the precomputed Higgs-decay table at
/storage/agrp/barakma/PileupODD/data/higgs_decays.parquet.

Schema:
  run        u32
  event_id   u32
  decay      str          e.g. "b -> b", "Z -> Z", "tau -> tau"
  out_pids   list[i64]    PDG ids of the Higgs daughters

Two views:
  - by `decay` label (as the producer wrote it)
  - by sorted absolute PDG tuple (folds antiparticle pairs and any
    label-formatting differences into one canonical channel)
"""

import sys

import polars as pl

sys.path.insert(0, "/storage/agrp/barakma/PileupODD")
from primary.pdg_mappings import PDG_ID_TO_NAME

DECAYS_PATH = "/storage/agrp/barakma/PileupODD/data/higgs_decays.parquet"


def _name(pdg: int) -> str:
    return PDG_ID_TO_NAME.get(str(pdg), f"pdg={pdg}")


def load_lazy() -> pl.LazyFrame:
    return pl.scan_parquet(DECAYS_PATH)


def counts_by_label(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.group_by("decay")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(engine="streaming")
    )


def counts_by_abs_pdg(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.with_columns(
            pl.col("out_pids").list.eval(pl.element().abs()).list.sort().alias("channel")
        )
        .group_by("channel")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(engine="streaming")
    )


def print_label_table(table: pl.DataFrame) -> None:
    total = int(table["count"].sum())
    print(f"\nby decay label (total events: {total})")
    print(f"{'count':>8}  {'frac':>7}  decay")
    print("-" * 50)
    for decay, count in table.iter_rows():
        print(f"{count:8d}  {count/total:7.2%}  {decay}")


def print_pdg_table(table: pl.DataFrame) -> None:
    total = int(table["count"].sum())
    print(f"\nby abs-pdg channel (total events: {total})")
    print(f"{'count':>8}  {'frac':>7}  channel")
    print("-" * 50)
    for channel, count in table.iter_rows():
        names = " ".join(_name(p) for p in channel)
        pdg_str = ",".join(str(p) for p in channel)
        print(f"{count:8d}  {count/total:7.2%}  {names}  ({pdg_str})")


def main() -> None:
    lf = load_lazy()
    n_events = lf.select(pl.len()).collect().item()
    n_runs = lf.select(pl.col("run").n_unique()).collect().item()
    print(f"{DECAYS_PATH}: {n_events} events across {n_runs} runs")

    print_label_table(counts_by_label(lf))
    print_pdg_table(counts_by_abs_pdg(lf))


if __name__ == "__main__":
    main()
