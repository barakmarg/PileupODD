# colliderml-pflow

Construction of particle-flow training datasets from the CERN **ColliderML-Release-1**
Geant4 simulation of the **Open Data Detector (ODD)**.

Turns sharded simulation output — generator particles with full parentage, reconstructed
tracks, and calorimeter hits with truth contributions — into four flat, ML-ready Parquet
tables per shard, in three modes.

This is the dataset-construction code for the accompanying paper. For the physics
definitions and thresholds, see [docs/METHODS.md](docs/METHODS.md).

> [!IMPORTANT]
> **A CUDA GPU is required. Only `clustering.backend: gpu cuda` produces usable data.**
> CLUEstering's CPU backends are broken: they emit infinite coordinate and energy values,
> and they suffer mode collapse, lumping most hits into a single cluster. Any dataset built
> with a `cpu *` backend is invalid. Selecting one emits a `RuntimeWarning`; they are kept
> selectable only for exercising code paths that do not depend on cluster content.

---

## Contents

- [The three modes](#the-three-modes)
- [Install](#install)
- [Quick start](#quick-start)
- [Output tables](#output-tables)
- [Dataset layout and splitting](#dataset-layout-and-splitting)
- [Configuration reference](#configuration-reference)
- [Reproducing the paper datasets](#reproducing-the-paper-datasets)
- [Running at scale](#running-at-scale)
- [Reproducibility and determinism](#reproducibility-and-determinism)
- [Testing](#testing)
- [Verification against the existing datasets](#verification-against-the-existing-datasets)
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

### Weizmann cluster: one script

```bash
git clone <this repo> && cd <this repo>
./setup_cluster.sh --smoke
```

That activates the shared conda environment, installs any missing Python dependency,
builds and installs CLUEstering with its CUDA backend, and then runs a one-event
end-to-end check. Total time is under a minute when CLUEstering is already built, plus a
few minutes for the compile the first time.

It is safe to re-run — every step checks whether it is already satisfied. `--force`
rebuilds CLUEstering anyway; omit `--smoke` to skip the check and just print the command.
The script fails loudly rather than falling back if the CUDA backend cannot be built,
since a CPU-only install cannot produce a valid dataset.

The rest of this section is what the script does, for anyone setting up by hand or working
off-cluster.

### 1. Python environment

On the Weizmann cluster, the shared `common` environment already provides polars, numpy,
pyarrow, PyYAML, tqdm, datasketches and psutil:

```bash
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common
```

Elsewhere:

```bash
pip install -r requirements.txt
```

### 2. CLUEstering (must be compiled)

The shared `common` environment does **not** include CLUEstering — it is a user-site
install. Check whether you already have it, and crucially whether it has the **CUDA**
backend:

```bash
python -c "
import CLUEstering, pathlib
lib = pathlib.Path(CLUEstering.__file__).parent / 'lib'
print(CLUEstering.__file__)
print(sorted(p.name.split('.')[0] for p in lib.glob('*.so')))
"
```

You need `CLUE_GPU_CUDA` in that list. If the import fails, or only the `CLUE_CPU_*`
backends are listed, build it — a CPU-only install cannot produce a valid dataset.

**Build requirements**

| requirement | note |
|---|---|
| CMake ≥ 3.16 | 3.26.5 in the `common` env |
| C++20 compiler | g++ 11.4.1 works |
| CUDA toolkit (`nvcc`) | **required for the `gpu cuda` backend**; `/usr/local/cuda/bin/nvcc` on the cluster |
| network access | alpaka 2.1.0 is downloaded at configure time via CMake `FetchContent` |
| git submodules | `extern/pybind11` must be checked out |

**Build and install**

A checkout already exists on the cluster at `/storage/agrp/barakma/CLUEstering`
(upstream: `https://gitlab.cern.ch/kalos/CLUEstering.git`, built at tag 2.9.0). Copy it
somewhere writable rather than building in place:

```bash
source /usr/wipp/conda/24.5.0u/bin/activate /usr/wipp/conda/24.5.0u/envs/common

# Get the source. Either copy the existing checkout ...
cp -r /storage/agrp/barakma/CLUEstering ~/CLUEstering
cd ~/CLUEstering
# ... or clone it fresh:
#   git clone https://gitlab.cern.ch/kalos/CLUEstering.git ~/CLUEstering
#   cd ~/CLUEstering && git checkout 2.9.0

# pybind11 is a submodule and the build fails without it.
git submodule update --init --recursive

# nvcc MUST be visible now: CMake probes for it with check_language(CUDA) and
# silently omits the CUDA backend if it is absent. See the warning below.
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version    # confirm before continuing

# setup.py drives CMake itself (cmake -B build -DBUILD_PYTHON=ON, then --build).
# This compiles four backends and takes several minutes.
pip install --user .
```

> **The most common failure is silent.** `CLUEstering/BindingModules/CMakeLists.txt` calls
> `check_language(CUDA)` and only adds the CUDA target if a compiler is found. Build
> without `nvcc` on `PATH` and everything succeeds, but you get CPU backends only — and
> then every run configured with `backend: gpu cuda` fails at clustering time. Re-run the
> check above after installing and confirm `CLUE_GPU_CUDA` is present.

CLUEstering pulls in `scikit-learn`, `matplotlib` and `pandas` as its own dependencies.
This package does not use them.

### 3. Verify

One event, end to end, in under a minute — proves the HuggingFace read, the spawned
worker, CUDA clustering, the aggregation and the parquet write all work:

```bash
python -m colliderml_pflow preprocess --config configs/quick_check.yaml
```

It should end with `[ALL SHARDS DONE]` and leave four parquet files in `data/quick_check/`.
For a fuller check across all three modes, use `configs/smoke.yaml` (4 events).

Optionally install the package to get the `colliderml-pflow` entry point on your `PATH`:

```bash
pip install -e .
```

No HuggingFace credentials are needed — the dataset is public.

## Quick start

The fastest end-to-end check is one event, in under a minute — predicate pushdown fetches
only the event requested, not the shard:

```bash
python -m colliderml_pflow preprocess --config configs/quick_check.yaml
```

A fuller four-event check, which you can point at any of the three modes:

```bash
python -m colliderml_pflow preprocess --config configs/smoke.yaml --set mode=all_vertices
```

Then the full path from raw shards to a model-ready dataset — two steps, because
train/validation/test splitting happens in the training dataloader, not here (see
[Dataset layout and splitting](#dataset-layout-and-splitting)):

```bash
# 1. Build the dataset.
python -m colliderml_pflow preprocess --config configs/ttbar_pu200_all_vertices.yaml

# 2. Compute input normalization statistics over the written shards.
python -m colliderml_pflow norm-stats  --config configs/ttbar_pu200_all_vertices.yaml
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

## Dataset layout and splitting

A dataset is **one flat directory of shards** — no `train/`, `val/` or `test/`
subdirectories:

```
data/ttbar_pu200_all_vertices/
  target_particles-00000.parquet   calo_clusters-00000.parquet
  tracks-00000.parquet             target_particles_deps-00000.parquet
  target_particles-00001.parquet   ...
  normalization_stats.yaml
```

**This package does not split the data, by design.** The training dataloader
(`hepattn`'s `pflow_data.py`) does it at load time: it globs
`target_particles-*.parquet` from a single directory, shuffles the shard list
deterministically from its `seed`, and slices it by `train_split` / `val_split` /
`test_split`. So the split is **by whole shard file**, chosen at training time.

Writing splits here would be wrong in three ways: it splits at the wrong granularity
(per event, not per shard), produces a directory layout nothing reads, and triples the
dataset on disk. Configure the split on the dataloader instead:

```yaml
data:
  unify_path: /path/to/data/ttbar_pu200_all_vertices
  enable_split: true
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  seed: 42
```

One consequence for this package: every table in a shard must cover exactly the same
events, or a shard-level split would tear an event apart. That is asserted in
`tests/test_smoke.py`.

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
  backend: gpu cuda           # the only usable backend; cpu * are broken, see above
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
independent and safely re-runnable. Because shards are independent, a failed job can be
re-run on its own range without touching the rest. Overlay defaults to `--group-size 3`, matching the
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
   `group_by` and `sort` are order-unstable by default, so the order in which points
   reached CLUE — and hence the labels it assigned — varied between runs. Pinning it
   removes that source of variation. The CUDA backend is still nondeterministic
   internally, so cluster labels continue to vary; bit-exact output is not achievable on
   GPU, and the CPU backends are not a usable alternative (see the notice at the top).
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
pytest tests/ -q -m "not slow"      # fast unit tests only (<1 s), no GPU needed
```

The GPU-marked tests need a CLUEstering build with the CUDA backend — see
[Install](#install). There is no CPU fallback: the fast tests avoid clustering entirely
rather than clustering on a CPU backend, because CPU results are not meaningful.

| file | what it covers | needs |
|---|---|---|
| `test_config.py` | config validation, mode→vertex-policy contract, presets | — |
| `test_reproducibility.py` | pileup sampling is seed-reproducible and order-invariant | — |
| `test_smoke.py` | all three subcommands end to end, output schema | network, GPU |
| `test_equivalence.py` | agreement with the `master` scripts, per mode | network, GPU |

Separately, `tools/compare_to_reference.py` checks output against the datasets already on
disk -- see [Verification against the existing datasets](#verification-against-the-existing-datasets).

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

## Verification against the existing datasets

Two independent checks. `tests/test_equivalence.py` compares **code against code** --
master's implementation and this one, run in the same session on identical cached input.
That isolates the port, but says nothing about the datasets already on disk.
`tools/compare_to_reference.py` closes that gap: it regenerates events with this branch and
compares them **value by value against the stored datasets**.

```bash
python tools/compare_to_reference.py \
    --reference /storage/agrp/barakma/PileupODD/data/dihiggs_pu200_all_vertices_paper \
    --mode all_vertices --event-name dihiggs_pu200 --shard 0 --n-events 3
```

### Why a few events is enough

Every stage is per-event: clustering runs per event, the masks and target selection are
per-event expressions, and `particle_idx` / `cluster_idx` are dense indices *within* an
event. Regenerating 3 events of a shard therefore yields exactly what a full-shard run
yields for those 3 events -- so a genuine value-level comparison costs minutes, not hours.

### What is compared, and how it is keyed

Never by row position. Row order and the per-event index columns are not stable across
runs, so everything is joined on stable physical identifiers:

| table | key | comparison |
|---|---|---|
| `target_particles` | `(event_id, particle_id)` | every physics column, **exact** |
| `tracks` | `(event_id, track_id)` | every column, **exact** |
| `target_particles_deps` | `(event_id, particle_id)`, after resolving `particle_idx` back through `target_particles` | particle set exact; summed deposit energy per particle |
| `calo_clusters` | -- | label-invariant aggregates only |

`cluster_id` and `cluster_idx` are labels handed out in CLUE's discovery order. They mean
nothing across runs, so nothing can be joined on them -- hence the aggregate treatment of
the two cluster-dependent tables (cluster count, total and HCal energy, hit count, and
energy per physical `vertex_primary`).

### Results

| reference dataset | mode | verdict |
|---|---|---|
| `data/dihiggs_pu200_all_vertices_paper` | `all_vertices` | **MATCH** |
| `data/ttbar_pu200` | `hard_scatter` | **MATCH** |
| `data/ttbar_pu0_overlay_pu200` | `overlay` | **MATCH** on the comparable entries -- see below |

On 3 events of `dihiggs_pu200_all_vertices_paper`:

- `target_particles` -- 8452 particles, identical set, **0 mismatches** across `pdg_id`,
  `energy`, `eta`, `phi`, `px`, `py`, `pz`, `pt`, `has_track`, `vertex_primary`, `vx`,
  `vy`, `vz` (worst relative difference exactly `0.0`);
- `tracks` -- 2619 tracks, identical set, **0 mismatches** across all 17 columns;
- `target_particles_deps` -- identical particle set; 9 of 8413 summed energies differ;
- `calo_clusters` -- worst aggregate deviation **0.026%**.

On 3 events of `ttbar_pu200`: 474 target particles and 2084 tracks, **0 mismatches** on
every column; 1 of 474 deposit energies differs; worst cluster aggregate **0.027%**.

### Calibrating what "match" means

A stored dataset is one stochastic draw, not the answer, so the deposit differences need a
baseline. Re-running **master's own code** on the same 3 dihiggs events and comparing it
against master's own stored output:

| comparison | particles differing >0.5% | worst rel |
|---|---|---|
| stored reference vs **master re-run** | **9** / 8413 | 9.25e-02 |
| stored reference vs **this branch** | **7** / 8413 | 9.25e-02 |
| master re-run vs this branch | 6 / 8413 | 4.67e-02 |

Master differs from its own stored output by *more* than this branch does, and the worst
case is the same particle in both -- it is the stored draw that is the outlier there. The
mechanism is the cluster energy cutoff: a cluster sitting near the 0.15 GeV threshold is
kept in one run and dropped in the next, moving that whole cluster's energy off the
particles that fed it. `--max-differing-frac` (default 0.5%) bounds how many particles may
be affected.

### Overlay: comparing only the entries that can be compared

Two things in `data/ttbar_pu0_overlay_pu200` are not reproducible, so the comparison filters
them out rather than pretending otherwise.

**The pileup draw.** Master's sampler walked an order-unstable pool, so its exact pileup
content cannot be recovered by any code (see
[Reproducibility and determinism](#reproducibility-and-determinism)). Every pileup track,
and every quantity summed over pileup, therefore differs by construction.

**One contaminated column.** That dataset was written 29 May - 7 Jun;
`filter_orphans_and_reindex` was fixed afterwards (commit `c3b2171`, 25 Jun) so that pileup
tracks whose event-local `majority_particle_id` collided with a hard-scatter target's are
marked `-1` instead of being wired to that target. Master's comment puts the effect at
**~45% of incidence links**; measured on 20 events of the stored dataset, **964 of 16139
pileup tracks (6.0%) carry a spurious `particle_idx >= 0`**. This branch produces 0.

The fix is narrowly scoped, which is what makes a meaningful comparison still possible. It
rewrote only the `tracks_mappings` expression, so `tracks.particle_idx` is the **only**
contaminated column — and only on pileup rows, since the fix's `otherwise` branch is the
original expression. `valid_ids` is built from `majority_particle_id`, which the fix did not
touch, so the surviving target-particle set and `target_particles_deps` are unaffected by
the bug.

So the overlay comparison keeps:

| table | compared | excluded |
|---|---|---|
| `target_particles` | all physics columns, exact | — |
| `tracks` | hard-scatter rows, **all** columns incl. `particle_idx` | pileup rows |
| `calo_clusters` | hard-scatter vertex energy | totals (pileup-dependent) |
| `target_particles_deps` | particle set; energies reported, not gated | — |

Result on 2 events of shard 6:

- `target_particles` — identical set, **0 mismatches** on all 13 physics columns;
- `tracks` — 98 hard-scatter tracks, identical set, **0 mismatches** on all 18 columns,
  `particle_idx` included: the contaminated column agrees exactly where the fix is a no-op;
- `calo_clusters` — hard-scatter vertex energy agrees to **8.7e-04** relative;
- `target_particles_deps` — identical particle set (315 both sides); 48 of 315 summed
  energies differ, as expected from a different pileup draw landing on the same clusters.

The stored overlay dataset should still be **regenerated** rather than kept, because of the
incidence bug. The comparison above establishes that this branch reproduces everything about
it that was correct.

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
  the pipeline, which writes vertex info directly), the `create_pileup_pool_from_pu200.py`
  overlay variant that built its pool from real PU200 vertices, and
  `split_train_val_test()` — splitting belongs to the dataloader, see
  [Dataset layout and splitting](#dataset-layout-and-splitting). All remain on `master`.
  Dropping the splitter also removed the `scikit-learn` dependency.

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
  submit.py          PBS job splitting and submission
setup_cluster.sh     cluster setup: conda env + CLUEstering build + end-to-end check
configs/             one preset per paper dataset, plus smoke.yaml and quick_check.yaml
tools/               compare_to_reference.py -- value-level check against stored datasets
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
