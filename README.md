# colliderml-pflow

Construction of particle-flow training datasets from the CERN **ColliderML-Release-1**
Geant4 simulation of the **Open Data Detector (ODD)**.

Turns sharded simulation output — generator particles with full parentage, reconstructed
tracks, and calorimeter hits with truth contributions — into four flat, ML-ready Parquet
tables per shard, in three modes.

This is the dataset-construction code for the accompanying paper. For the physics
definitions and thresholds, see [docs/METHODS.md](docs/METHODS.md).

---

## Contents

- [The three modes](#the-three-modes)
- [Install](#install)
- [Quick start](#quick-start)
- [Output tables](#output-tables)
- [Configuration reference](#configuration-reference)
- [Reproducing the paper datasets](#reproducing-the-paper-datasets)
- [Running at scale](#running-at-scale)
- [Reproducibility and determinism](#reproducibility-and-determinism)
- [Testing](#testing)
- [How this relates to the `master` branch](#how-this-relates-to-the-master-branch)
- [Package layout](#package-layout)

---

## The three modes

All three share one pipeline. They differ only in which particles become reconstruction
targets, and whether pileup is real or synthetic.

| mode | input | targets built from | pileup |
|---|---|---|---|
| `hard_scatter` | `ttbar_pu200` | `vertex_primary == 1` only | real, in the sample |
| `all_vertices` | `ttbar_pu200` | **every** vertex | real, in the sample |
| `overlay` | `ttbar_pu0` + `pileup_only_pu0` | `vertex_primary == 1` only | **synthetic**, overlaid |

- **`hard_scatter`** — the baseline. Pileup deposits energy but is never something the
  network is asked to reconstruct.
- **`all_vertices`** — the main paper dataset. Pileup particles are targets too, so the
  network must *separate* hard scatter from pileup rather than being handed the
  distinction.
- **`overlay`** — synthetic pileup. Takes pileup-free hard-scatter events and overlays
  `Poisson(μ)` pileup interactions. This decouples the physics process from the pileup
  level, so the same events can be studied at several values of μ, and the pileup content
  of every cluster is known exactly.

> **`overlay` is `hard_scatter` plus an overlay stage, not a third vertex policy.** Its
> hard-scatter side applies the same `vertex_primary == 1` filter. On `ttbar_pu0` input
> that filter is a no-op — every particle already sits on the single primary vertex — but
> it is applied, and the particle→vertex snapshot it feeds is what lets overlaid clusters
> be broken down by originating vertex.

The vertex policy is **derived from `mode`** and is deliberately not a separate config
knob, so a run cannot silently drift from the published datasets.

## Install

The pipeline needs `polars`, `numpy`, `CLUEstering`, and a CUDA device for clustering.

On the Weizmann cluster, everything is already present in the shared `common` environment:

```bash
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
```

Elsewhere:

```bash
pip install -r requirements.txt
# or, as an installed package:
pip install -e '.[clustering,test]'
```

Then run from the repository root (or `pip install -e .` to get the
`colliderml-pflow` entry point on your `PATH`).

No HuggingFace credentials are needed — the dataset is public.

## Quick start

A four-event run to confirm everything works end to end. It takes a couple of minutes
and downloads a few MB, because predicate pushdown fetches only the events requested:

```bash
python -m colliderml_pflow preprocess --config configs/smoke.yaml --set mode=all_vertices
```

Then the full path from raw shards to a model-ready dataset:

```bash
# 1. Build the dataset.
python -m colliderml_pflow preprocess --config configs/ttbar_pu200_all_vertices.yaml

# 2. Compute input normalization statistics over the written shards.
python -m colliderml_pflow norm-stats  --config configs/ttbar_pu200_all_vertices.yaml

# 3. Split into train/val/test by event.
python -m colliderml_pflow split       --config configs/ttbar_pu200_all_vertices.yaml
```

Inspect a resolved configuration without running anything:

```bash
python -m colliderml_pflow preprocess --config configs/smoke.yaml --print-config
```

Override any setting from the command line — no need to edit YAML:

```bash
python -m colliderml_pflow preprocess --config configs/smoke.yaml \
    --set mode=overlay \
    --set dataset.event_name=ttbar_pu0 \
    --set overlay.pileup_level=60 \
    --set runtime.chunk_size=10
```

Values are parsed as YAML, so `null`, `true` and `[0,1,2]` all work. Later `--set` wins.
An unrecognised key is an error, not a silent no-op.

## Output tables

Four Parquet files per shard, written as `<table>-<shard:05d>.parquet`.

Every table is **one row per event**, with `event_id` as a scalar and all other columns
as per-event `List`s. **In-event list order is meaningful**: `particle_idx` and
`cluster_idx` are positional indices *within* an event.

### `target_particles` — the truth objects to reconstruct
`particle_id`, `pdg_id`, `energy`, `eta`, `phi`, `px`, `py`, `pz`, `pt`, `has_track`,
`vertex_primary`, `vx`, `vy`, `vz`, `particle_idx`

### `calo_clusters` — calorimeter nodes seen by the network
`cluster_id`, `total_cluster_energy`, `hcal_energy`, `hcal_fraction`,
`sigma_eta`, `sigma_phi`, `sigma_rho` (hit spread within the cluster),
`number_of_hits`, `energy_hits_std`, `max_hit_energy`,
`cluster_eta`, `cluster_phi`, `cluster_rho` (centroid),
`vertex_primary_indices`, `vertex_primary_energies` (energy broken down by originating
vertex — the pileup label)

### `tracks` — reconstructed tracks (hard-scatter **and** pileup)
`d0`, `z0`, `phi`, `theta`, `qop`, `hit_ids`, `track_id`, `phi_int`, `eta_int`,
`track_tanlambda`, `track_omega`, `pt`, `eta`, `vertex_primary`, `vx`, `vy`, `vz`,
`particle_pt`, `particle_idx`, `particle_id`

`particle_idx >= 0` marks a track matched to a target particle; `-1` marks a pileup or
unmatched track.

In `overlay` mode only, one extra column: `source_pileup_event_id` — null on hard-scatter
tracks, the originating pileup event id on overlaid ones.

### `target_particles_deps` — the incidence ground truth
`particle_idx`, `cluster_idx`, `total_energy_deps_in_cluster`

A sparse (particle, cluster) → energy table: which target particles deposited how much
energy in which clusters. This is the assignment the network learns.

> **`cluster_time` is not produced.** See
> [How this relates to the `master` branch](#how-this-relates-to-the-master-branch).

## Configuration reference

```yaml
mode: all_vertices            # hard_scatter | all_vertices | overlay

dataset:
  repo: CERN/ColliderML-Release-1
  shards_total: 1000          # part of the shard filename
  event_name: ttbar_pu200     # dataset prefix
  file_indices: {range: [0, 100]}   # or [0,1,2], or {map: {shard: [event_id, ...]}}
  max_events_per_file: null   # cap events per shard; null = all

overlay:                      # read only when mode == overlay
  pu_event_name: pileup_only_pu0
  pu_file_indices: [0, 1, 2]  # shared sampling pool for the whole run
  pu_max_events: null         # cap the pool size
  pileup_level: 200           # Poisson mean
  seed: 42                    # shard i uses seed + i
  invisible_pu_prob: 0.0      # diffractive fraction contributing nothing (0.19 measured)
  tof:                        # read-out time window, pileup hits only
    enabled: true
    sigma_ns: 0.185           # per-vertex Gaussian time spread
    window_ns: [-1.0, 10.0]

cuts:
  truth_pt: 1.0               # GeV — min track pT, and truth pT threshold
  truth_eta: 3.0              # max |eta|
  target_pt: 0.3              # GeV — min pT to become a target
  cluster_energy: 0.15        # GeV — min calibrated cluster energy

clustering:
  backend: gpu cuda           # cpu serial | cpu tbb | cpu omp
  dc: 75.88106168184893       # CLUE critical distance
  rhoc: 104.34315216716726    # CLUE seeding density
  dm: 87.0967630118376        # CLUE max assignment distance
  ppbin: 16                   # CLUE points per tile
  deterministic: true         # pin the point order CLUE sees

runtime:
  chunk_size: 100             # events per worker process; <=0 = whole shard
  output_dir: data/${dataset.event_name}_${mode}
  tmp_dir: data/tmp
```

`${a.b}` references are interpolated against the configuration itself.

### `file_indices`

| form | meaning |
|---|---|
| `{range: [0, 100]}` | shards 0–99 (upper bound exclusive) |
| `[7, 42, 91]` | exactly those shards |
| `{map: {7: [700, 701], 42: [4200]}}` | exactly those *events* from those shards |

The `map` form pins individual events, which is how a specific validation set can be
regenerated exactly.

### `chunk_size`

Each chunk runs in its own spawned process. This is about memory, not parallelism —
polars' allocator does not return large transients to the OS within a process, so a
long-lived run creeps upward until it is killed. Ending the process at each chunk
boundary makes the OS reclaim everything. Lower it if a run is killed for memory;
overlaid events are much heavier, hence `chunk_size: 50` in the overlay preset.

## Reproducing the paper datasets

| dataset | config |
|---|---|
| ttbar PU200, all vertices | `configs/ttbar_pu200_all_vertices.yaml` |
| ggF Higgs PU200, all vertices | `configs/ggf_pu200_all_vertices.yaml` |
| ttbar PU200, hard scatter only | `configs/ttbar_pu200_hard_scatter.yaml` |
| ttbar PU0 + synthetic PU200 | `configs/ttbar_pu0_overlay_pu200.yaml` |

```bash
python -m colliderml_pflow submit --config configs/ttbar_pu200_all_vertices.yaml --dry-run
python -m colliderml_pflow submit --config configs/ttbar_pu200_all_vertices.yaml
```

## Running at scale

A full dataset is 100+ shards, far more than one walltime allowance. The `submit`
subcommand splits the shard range into groups and queues one PBS job per group:

```bash
# Inspect the qsub command lines without submitting.
python -m colliderml_pflow submit --config configs/ttbar_pu200_all_vertices.yaml --dry-run

# Queue them.
python -m colliderml_pflow submit --config configs/ttbar_pu200_all_vertices.yaml \
    --group-size 4 --queue N --log-dir logs

# Or run the groups locally, one fresh subprocess at a time.
python -m colliderml_pflow submit --config configs/ttbar_pu200_all_vertices.yaml --local
```

Each job re-reads the YAML and receives its own `dataset.file_indices`, so jobs are
independent and safely re-runnable. Overlay defaults to `--group-size 3`, matching the
pileup pool's 3-shard blocks so the pool is loaded once per group.

## Reproducibility and determinism

**Calorimeter clustering is stochastic.** CLUE hands out cluster ids in discovery order,
and its CUDA backend reduces nondeterministically. Two runs over byte-identical input
therefore produce different cluster *labels* and slightly different cluster boundaries.

What that means in practice, measured on a two-event `ttbar_pu200` fixture:

| table | reproducible? |
|---|---|
| `target_particles` | **yes, exactly** — produced before clustering |
| `tracks` | **yes, exactly** |
| `calo_clusters` | labels vary; counts exact, energies stable to ~0.004% |
| `target_particles_deps` | link count varies by ~0.02%, total energy stable to ~0.001% |

Structural quantities — number of clusters, number of contributing vertices, number of
target particles receiving energy — came out **exactly equal** across every run measured.

This branch pins three ordering-dependent steps that `master` left unpinned, each of
which was an avoidable source of variation:

1. **The point order CLUE sees** (`clustering.deterministic`, default `true`). Polars'
   `group_by` and `sort` are order-unstable by default; with them pinned, the CPU
   backends become bit-reproducible. The CUDA backend remains nondeterministic
   internally — for bit-exact output, use `--set clustering.backend='cpu serial'`.
2. **The pileup sampling pool order.** `master` fed `numpy`'s sampler a pool whose order
   came from an order-unstable `unique()`, so **the overlay drew a different pileup
   sample on every run despite taking a `seed`**. Both id arrays are now sorted before
   sampling, so the sample is genuinely a function of the seed and the pool contents.
3. **The overlaid pileup track order**, previously straight out of an unordered join,
   now sorted on `(source_pileup_event_id, track_id)`.

Overlay seeding: shard `i` uses `seed + i`, and each chunk derives its own seed by
hashing `(shard_seed, chunk_index)` through `numpy.random.SeedSequence`. Hashing rather
than adding matters — `seed + chunk` collides whenever `a + n == b + m`, which would
silently reuse a pileup sample across shards.

Note that chunked and unchunked overlay runs draw *different* samples by construction,
since the sample map is drawn per chunk. Comparing two overlay runs requires the same
`chunk_size`.

## Testing

```bash
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common

pytest tests/ -q                    # everything (~8 min; needs network + GPU)
pytest tests/ -q -m "not slow"      # fast unit tests only (<1 s)
```

| file | what it covers | needs |
|---|---|---|
| `test_config.py` | config validation, mode→vertex-policy contract, presets | — |
| `test_reproducibility.py` | pileup sampling is seed-reproducible and order-invariant | — |
| `test_smoke.py` | all four subcommands end to end, output schema | network, GPU |
| `test_equivalence.py` | agreement with the `master` scripts, per mode | network, GPU |

`test_equivalence.py` is the correctness gate. For each mode it runs the original
implementation and this one in separate subprocesses over identical cached input, then:

- requires **exact equality** for `target_particles` and `tracks` — every column, dtype,
  and in-event list position;
- compares the cluster-dependent tables through **label-invariant physics**: cluster
  count, total and HCal energy, energy grouped by physical `vertex_primary`, and
  deposited energy grouped by the stable `particle_idx`;
- includes a **self-calibrating** check that runs `master` *twice* and requires this
  branch to be no further from `master` than `master` is from itself.

The first run downloads a small fixture to `tests/fixtures/` and caches it; later runs
need no network. Delete the directory to refetch.

## How this relates to the `master` branch

This branch consolidates three ~90%-duplicated scripts
(`create_trainning_dataset_pileup.py`, `..._research.py`,
`create_training_dataset_pileup_overlay.py`, 1615 / 1709 / 2065 lines) into one
configurable package. Nine helper functions were byte-identical triplicates; the shared
`preprocessing.py` was 2844 lines of which 1512 were unreachable.

Behaviour is preserved except where noted here.

### `cluster_time` is no longer produced

`master` computed a `cluster_time` column for `hard_scatter` and `all_vertices` (an
energy-weighted hit time with ATLAS TileCal resolution smearing plus a per-event 0.17 ns
shift). Its overlay script already omitted it. This branch drops it in **all** modes,
giving one uniform schema.

> **This is a breaking change downstream.** `cluster_time` is a live network input in
> three `hepattn` experiments — `odd_pileup_reco_pu_cond/pflow_data.py:229,764,854`,
> `odd_pileup_maskformer/pflow_data.py:199,524,573`, and
> `odd_pileup_reco/configs/odd_var_transform.yaml:127` — and both existing all-vertices
> paper datasets contain it. **Datasets regenerated from this branch will not load in
> those experiments, and are not compatible with checkpoints trained with that input,
> until `cluster_time` is removed from those field lists and transform configs.** That
> change is outside this repository.
>
> A side benefit: `contrib_times` was only ever read to build `cluster_time`, so the
> non-overlay modes now read one large column less per shard.

### Other intentional changes

- **Predicate-pushdown loading everywhere.** `master` used it only for `hard_scatter`;
  the other two downloaded whole shards (a PU200 `calo_hits` shard is 2.1 GB) and sliced
  in memory. Same bytes reach the same frames — the `all_vertices` equivalence test
  validates exactly this — but `max_events_per_file` now genuinely limits I/O.
- **Ordering pinned** in three places, as described in
  [Reproducibility and determinism](#reproducibility-and-determinism).
- **One `filter_orphans_and_reindex`.** `master`'s overlay version has a guarded
  pileup-track branch that falls back to the original behaviour when
  `source_pileup_event_id` is absent, so it is a strict superset and serves every mode.
- **Normalization spec unified.** `master` had two variants that disagreed on whether
  `number_of_hits` gets a `sqrt` transform. The published statistics came from the
  streaming version (no transform), so that is what is kept; the divergent in-memory
  twin is not carried over.
- **`preprocessing.py` pruned** to its 14 reachable functions, dropping the `torch`,
  `fastjet`, `awkward` and `sklearn.cluster` dependencies with it. Function bodies are
  otherwise unmodified.
- **Not carried over:** the `add/update_*_vertex_info` backfill helpers (superseded by
  the pipeline, which writes vertex info directly) and the `create_pileup_pool_from_pu200.py`
  overlay variant that built its pool from real PU200 vertices. Both remain on `master`.

## Package layout

```
colliderml_pflow/
  cli.py             preprocess | norm-stats | split | submit
  config.py          typed config, YAML loading, --set overrides, validation
  hf_io.py           HuggingFace reads (predicate pushdown, retry)
  pipeline.py        Stage A (prepare_source) and Stage C (run_tail)
  overlay.py         Stage B: sampling, ToF cut, calo + track merge
  clustering.py      voxelisation + CLUE
  aggregate.py       cluster features, orphan filtering and reindexing
  preprocessing.py   particle-level primitives (masks, targets, backtracking)
  calibration.py     detector calibration and voxel-size tables
  pdg.py             PDG-id tables
  runner.py          shard orchestration, chunked spawn workers
  normalization.py   streaming KLL-sketch input statistics
  splits.py          train/val/test splitting by event
  submit.py          PBS job splitting and submission
configs/             one preset per paper dataset, plus smoke.yaml
docs/METHODS.md      physics definitions, selections, thresholds
tests/               see Testing
```

The pipeline factors into three stages, shared across modes:

- **Stage A** (`pipeline.prepare_source`) — per input source: Float32 cast, extrapolated
  track features, track selection, particle kinematics. The *primary* source additionally
  applies the vertex policy and builds the target-particle masks; the *pileup* source runs
  a reduced version and is the only one to precompute hit times for the ToF cut.
- **Stage B** (`overlay.run_overlay`) — overlay mode only.
- **Stage C** (`pipeline.run_tail`) — shared by every mode: cluster, drop sub-threshold
  clusters, attribute energy back to target particles, emit the four tables.
