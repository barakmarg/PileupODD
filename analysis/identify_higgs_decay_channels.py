"""
Identify the Higgs decay channel per event in ggf_pu200.

Why not `primary == True`: that flag marks "input to Geant4", i.e. the
generator final state AFTER parton shower + hadronization. For H -> bb that
is the whole hadron jet (~hundreds of particles), not the Higgs daughters.

Strategy that survives:
- Orphans: particles whose `parent_id` does not appear as a `particle_id`
  in the same event. Among these are the immediate decay products of the
  unstored Higgs (also some mid-shower hadrons whose intermediate parents
  the storage skipped).
- Restrict to the hard-scatter lineage (`vertex_primary == 1`).
- Per event, take the TOP-2 orphans by energy. The Higgs daughters carry
  ~MH/2 ~= 60 GeV each (plus boost); mid-shower orphan hadrons are O(GeV),
  so top-2-by-energy cleanly isolates the Higgs decay products.
- Aggregate the (sorted) pdg pair into a channel histogram.

Memory/perf: LazyFrame + streaming engine end-to-end; the exploded
particle-per-row frame is never materialized in Python.
"""

import sys

import polars as pl
from huggingface_hub import HfFileSystem

sys.path.insert(0, "/storage/agrp/barakma/PileupODD")
from primary.pdg_mappings import PDG_ID_TO_NAME

HS_EVENT_NAME = "ggf_pu200"
NUMBER_OF_HF_REPO_FILES = 1000
COLUMNS = ["event_id", "particle_id", "parent_id", "pdg_id",
           "vertex_primary", "energy", "px", "py"]
LIST_COLS = [c for c in COLUMNS if c != "event_id"]
HARD_SCATTER_VERTEX = 1


def _name(pdg: int) -> str:
    return PDG_ID_TO_NAME.get(str(pdg), f"pdg={pdg}")


def load_lazy(file_index: int = 0) -> pl.LazyFrame:
    fs = HfFileSystem()
    path = (
        f"datasets/CERN/ColliderML-Release-1/data/{HS_EVENT_NAME}_particles/"
        f"train-{file_index:05d}-of-{NUMBER_OF_HF_REPO_FILES:05d}.parquet"
    )
    print(f"loading {path}")
    with fs.open(path, "rb") as f:
        df = pl.read_parquet(f, columns=COLUMNS)
    return df.lazy()


def orphans(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Hard-scatter-lineage particles whose parent is not stored in the same
    event. Adds a `pt` column for downstream ranking. Beam remnants stream
    forward (high energy, sub-GeV pT) so pT cleanly separates them from
    central Higgs daughters (~MH/2 ~= 60 GeV in pT for a non-boosted H).
    """
    flat = lf.explode(LIST_COLS).with_columns(
        (pl.col("px") ** 2 + pl.col("py") ** 2).sqrt().alias("pt")
    )
    hs = flat.filter(pl.col("vertex_primary") == HARD_SCATTER_VERTEX)
    parents_in_event = flat.select("event_id", "particle_id")
    return hs.join(
        parents_in_event,
        left_on=["event_id", "parent_id"],
        right_on=["event_id", "particle_id"],
        how="anti",
    )


def top2_orphans(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Per event: the 2 highest-pT orphans (pdg + pT + energy)."""
    return (
        orphans(lf)
        .group_by("event_id")
        .agg(
            pl.col("pdg_id").sort_by("pt", descending=True).head(2).alias("pdg_top2"),
            pl.col("pt").sort_by("pt", descending=True).head(2).alias("pt_top2"),
            pl.col("energy").sort_by("pt", descending=True).head(2).alias("E_top2"),
        )
    )


def channel_counts(lf: pl.LazyFrame, *, signed: bool) -> pl.DataFrame:
    pdg_col = pl.col("pdg_top2") if signed else pl.col("pdg_top2").list.eval(pl.element().abs())
    per_event = top2_orphans(lf).select(
        "event_id",
        pdg_col.list.sort().alias("channel"),
    )
    return (
        per_event
        .group_by("channel")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(engine="streaming")
    )


def diagnostics(lf: pl.LazyFrame) -> None:
    t2 = top2_orphans(lf).collect(engine="streaming")
    print("\nfirst 10 events: top-2 orphans by pT  [pdg(pT, E) GeV]:")
    for row in t2.head(10).iter_rows(named=True):
        pairs = [
            f"{_name(p)}(pT={pt:.1f}, E={e:.1f})"
            for p, pt, e in zip(row["pdg_top2"], row["pt_top2"], row["E_top2"])
        ]
        print(f"  event {row['event_id']}: " + ", ".join(pairs))

    print("\ntop-1 orphan pT summary (Higgs daughter pT ~ tens of GeV):")
    pt1 = t2.select(pl.col("pt_top2").list.first().alias("pT1"))
    print(pt1.describe())


def print_channel_table(title: str, table: pl.DataFrame) -> None:
    total = int(table["count"].sum())
    print(f"\n{title} (total events: {total})")
    print(f"{'count':>7}  {'frac':>6}  channel")
    print("-" * 60)
    for channel, count in table.iter_rows():
        names = " ".join(_name(p) for p in channel)
        pdg_str = ",".join(str(p) for p in channel)
        print(f"{count:7d}  {count/total:6.2%}  {names}  ({pdg_str})")


def main(file_index: int = 0) -> None:
    lf = load_lazy(file_index)
    diagnostics(lf)
    print_channel_table("abs-pdg channel (top-2 orphans by pT)",
                        channel_counts(lf, signed=False))
    print_channel_table("signed-pdg channel (top-2 orphans by pT)",
                        channel_counts(lf, signed=True))


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
