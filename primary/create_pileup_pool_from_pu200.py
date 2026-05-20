"""
Build a pileup-only event pool from ttbar_pu200 by splitting each PU200 event
into N independent events — one per pileup vertex (vertex_primary > 1).

Each pileup vertex in a PU200 event becomes its own event with a new event_id
constructed as `orig_event_id * 1000 + vertex_primary`. Three dataframes are
emitted (particles, calo_hits, tracks) with the same column layout as the HF
`pileup_only_pu0` dataset, plus an extra `orig_event_id` column.

Run:
    python create_pileup_pool_from_pu200.py --file-index 0
"""

import argparse
import gc
import shutil
from pathlib import Path

import polars as pl
from huggingface_hub import HfFileSystem

EVENT_NAME = "ttbar_pu200"
NUMBER_OF_HF_REPO_FILES = 1000
OUT_DIR = Path("/storage/agrp/barakma/PileupODD/data/pileup_from_ttbar_pu200")

PARTICLE_COLS = [
    'event_id', 'particle_id', 'vertex_primary', 'pdg_id', 'energy',
    'px', 'py', 'pz', 'vx', 'vy', 'vz', 'parent_id',
]
CALO_HITS_COLS = [
    'event_id', 'detector', 'total_energy', 'x', 'y', 'z',
    'contrib_particle_ids', 'contrib_energies', 'contrib_times',
]


def _load_hf(fs: HfFileSystem, kind: str, file_index: int, columns=None) -> pl.DataFrame:
    path = (
        f"datasets/CERN/ColliderML-Release-1/data/{EVENT_NAME}_{kind}/"
        f"train-{file_index:05d}-of-{NUMBER_OF_HF_REPO_FILES:05d}.parquet"
    )
    print(f"  loading {path}")
    with fs.open(path, "rb") as f:
        return pl.read_parquet(f, columns=columns)


def _empty_list_lit(dtype: pl.DataType) -> pl.Expr:
    return pl.lit([], dtype=dtype)


def _align_to_canonical(
    df: pl.DataFrame,
    canonical: pl.DataFrame,
    list_cols: list[str],
    source_schema: dict,
) -> pl.DataFrame:
    """
    Left-join `df` onto `canonical` (event_id, orig_event_id) so every canonical
    event_id is present. Rows missing in df get empty lists of the correct
    nested dtype.
    """
    joined = canonical.join(df.drop('orig_event_id'), on='event_id', how='left')
    fills = []
    for c in list_cols:
        dtype = source_schema[c]
        fills.append(
            pl.when(pl.col(c).is_null())
            .then(_empty_list_lit(dtype))
            .otherwise(pl.col(c))
            .alias(c)
        )
    return joined.with_columns(fills).select(['event_id', 'orig_event_id', *list_cols])


def _build_particles_and_pidmap(
    particles: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Pass A — drop hard-scatter particles in a single sweep using `list.gather`
    (constant predicate, no explode).
    Pass B — explode the HS-free frame; assign new_event_id per (event, vertex).
    Pass C — regroup by new_event_id to get the per-vertex particles output.
    Returns (particles_per_vertex, pid_map).
    """
    list_cols = [c for c in particles.columns if c != 'event_id']

    particles_pu = (
        particles
        .with_columns(
            _idx=pl.col('vertex_primary').list.eval((pl.element() > 1).arg_true())
        )
        .with_columns(
            pl.exclude('event_id', '_idx').list.gather(pl.col('_idx'))
        )
        .drop('_idx')
    )

    exploded = (
        particles_pu
        .explode(list_cols)
        .with_columns(
            orig_event_id=pl.col('event_id'),
            new_event_id=(
                pl.col('event_id').cast(pl.UInt32) * 1000
                + pl.col('vertex_primary').cast(pl.UInt32)
            ).cast(pl.UInt32),
        )
    )

    new_particles = (
        exploded
        .group_by('new_event_id', maintain_order=True)
        .agg([pl.col('orig_event_id').first(), *[pl.col(c) for c in list_cols]])
        .rename({'new_event_id': 'event_id'})
        .select(['event_id', 'orig_event_id', *list_cols])
    )

    pid_map = exploded.select(['event_id', 'particle_id', 'new_event_id', 'orig_event_id'])
    return new_particles, pid_map


def _build_tracks(
    tracks: pl.DataFrame,
    pid_map: pl.DataFrame,
    canonical_events: pl.DataFrame,
) -> pl.DataFrame:
    """
    Explode tracks, inner-join with pid_map on (event_id, majority_particle_id),
    regroup by new_event_id, then fill empty rows for vertices with no tracks.
    """
    list_cols = [c for c in tracks.columns if c != 'event_id']
    schema = dict(tracks.schema)

    joined = (
        tracks
        .explode(list_cols)
        .join(
            pid_map.rename({'particle_id': 'majority_particle_id'}),
            on=['event_id', 'majority_particle_id'],
            how='inner',
        )
        .drop('event_id')
    )

    grouped = (
        joined
        .group_by('new_event_id', maintain_order=True)
        .agg([pl.col('orig_event_id').first(), *[pl.col(c) for c in list_cols]])
        .rename({'new_event_id': 'event_id'})
    )

    return _align_to_canonical(grouped, canonical_events, list_cols, schema)


def _build_calo_hits_chunk_lazy(
    calo_hits_chunk: pl.DataFrame,
    pid_map: pl.DataFrame,
) -> pl.LazyFrame:
    """
    Per-chunk lazy plan: level-1 explode by hit, level-2 explode contribs,
    inner-join with pid_map, regroup per (hit, new_event_id), regroup per event.
    """
    per_hit_cols = ['detector', 'x', 'y', 'z',
                    'contrib_particle_ids', 'contrib_energies', 'contrib_times']

    lf = (
        calo_hits_chunk.lazy()
        .drop('total_energy')  # recomputed downstream
        .explode(per_hit_cols)
        .with_row_index('hit_idx')
        .explode(['contrib_particle_ids', 'contrib_energies', 'contrib_times'])
        .rename({'contrib_particle_ids': 'particle_id'})
        .join(pid_map.lazy(), on=['event_id', 'particle_id'], how='inner')
        .drop('event_id')
    )

    per_hit = (
        lf
        .group_by(
            ['hit_idx', 'new_event_id', 'orig_event_id', 'detector', 'x', 'y', 'z'],
            maintain_order=True,
        )
        .agg([
            pl.col('particle_id').alias('contrib_particle_ids'),
            pl.col('contrib_energies'),
            pl.col('contrib_times'),
            pl.col('contrib_energies').sum().alias('total_energy'),
        ])
    )

    per_event = (
        per_hit
        .group_by('new_event_id', maintain_order=True)
        .agg([
            pl.col('orig_event_id').first(),
            pl.col('detector'),
            pl.col('total_energy'),
            pl.col('x'), pl.col('y'), pl.col('z'),
            pl.col('contrib_particle_ids'),
            pl.col('contrib_energies'),
            pl.col('contrib_times'),
        ])
        .rename({'new_event_id': 'event_id'})
    )
    return per_event


def _build_calo_hits(
    calo_hits: pl.DataFrame,
    pid_map: pl.DataFrame,
    canonical_events: pl.DataFrame,
    chunk_events: int,
    temp_dir: Path,
) -> pl.DataFrame:
    """
    Chunked + streaming construction of calo_hits per new_event_id.
    Each chunk is sunk to its own parquet to keep peak RSS bounded.
    Final step: concat chunks and align to canonical event set.
    """
    list_cols = [c for c in CALO_HITS_COLS if c != 'event_id']
    schema = dict(calo_hits.schema)

    temp_dir.mkdir(parents=True, exist_ok=True)

    n_events = calo_hits.height
    n_chunks = (n_events + chunk_events - 1) // chunk_events
    for k in range(n_chunks):
        start = k * chunk_events
        length = min(chunk_events, n_events - start)
        chunk = calo_hits.slice(start, length)
        print(f"  calo_hits chunk {k + 1}/{n_chunks}: events [{start}, {start + length})")
        plan = _build_calo_hits_chunk_lazy(chunk, pid_map)
        plan.sink_parquet(temp_dir / f"chunk-{k:04d}.parquet", compression='zstd')
        del chunk, plan
        gc.collect()

    print("  concatenating chunks + aligning to canonical events")
    concat = pl.scan_parquet(str(temp_dir / "chunk-*.parquet")).collect()
    aligned = _align_to_canonical(concat, canonical_events, list_cols, schema)
    return aligned


def build_pileup_pool(file_index: int, chunk_events: int = 50) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    # ===== particles =====
    print(f"[file {file_index}] particles")
    particles = _load_hf(fs, 'particles', file_index, PARTICLE_COLS)
    new_particles, pid_map = _build_particles_and_pidmap(particles)
    del particles
    gc.collect()

    canonical_events = new_particles.select(['event_id', 'orig_event_id'])

    out_path = OUT_DIR / f"particles-{file_index:05d}.parquet"
    new_particles.write_parquet(out_path, compression='zstd')
    print(f"  wrote {out_path}  ({new_particles.height} events)")
    del new_particles
    gc.collect()

    # ===== tracks =====
    print(f"[file {file_index}] tracks")
    tracks = _load_hf(fs, 'tracks', file_index)
    new_tracks = _build_tracks(tracks, pid_map, canonical_events)
    del tracks
    gc.collect()

    out_path = OUT_DIR / f"tracks-{file_index:05d}.parquet"
    new_tracks.write_parquet(out_path, compression='zstd')
    print(f"  wrote {out_path}  ({new_tracks.height} events)")
    del new_tracks
    gc.collect()

    # ===== calo_hits =====
    print(f"[file {file_index}] calo_hits  (chunk size {chunk_events})")
    calo_hits = _load_hf(fs, 'calo_hits', file_index, CALO_HITS_COLS)
    temp_dir = OUT_DIR / f"_tmp_calo-{file_index:05d}"
    new_calo_hits = _build_calo_hits(calo_hits, pid_map, canonical_events, chunk_events, temp_dir)
    del calo_hits, pid_map
    gc.collect()

    out_path = OUT_DIR / f"calo_hits-{file_index:05d}.parquet"
    new_calo_hits.write_parquet(out_path, compression='zstd')
    print(f"  wrote {out_path}  ({new_calo_hits.height} events)")
    del new_calo_hits
    gc.collect()

    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[file {file_index}] done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-index', type=int, nargs='+', default=[0],
                        help='HF file index/indices to process')
    parser.add_argument('--chunk-events', type=int, default=50,
                        help='events per chunk in the calo_hits double-explode stage')
    args = parser.parse_args()

    for i in args.file_index:
        build_pileup_pool(i, args.chunk_events)


if __name__ == '__main__':
    main()
