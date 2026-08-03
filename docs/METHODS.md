# Training-Dataset Preprocessing — Methods

How the particle-flow training dataset is produced from simulated detector output. States the
exact definitions, selection rules and numerical thresholds, and annotates every step with the
function and module that implements it, so each claim can be checked against the code.

All module paths are relative to `colliderml_pflow/`. The consolidated step → function index is
in [§12](#12-implementation-map).

> **Driver:** `preprocess_events()` in [`pipeline.py`](../colliderml_pflow/pipeline.py), wired up
> per shard by `run_preprocessing()` in [`runner.py`](../colliderml_pflow/runner.py).

---

## Table of contents

1. [Overview](#1-overview)
2. [Input data & pipeline](#2-input-data--pipeline)
3. [The three modes](#3-the-three-modes)
4. [Hard scatter vs pileup labelling](#4-hard-scatter-vs-pileup-labelling)
5. [Stable particles](#5-stable-particles)
6. [Target particle definition](#6-target-particle-definition)
7. [Calorimeter clustering](#7-calorimeter-clustering)
8. [Particle energy deposits (incidence)](#8-particle-energy-deposits-incidence)
9. [Synthetic pileup overlay](#9-synthetic-pileup-overlay)
10. [Network input variables](#10-network-input-variables)
11. [Parameter & threshold reference](#11-parameter--threshold-reference)
12. [Implementation map](#12-implementation-map)

---

## 1. Overview

The dataset originates from a **Geant4 simulation of the Open Data Detector (ODD)** — the CERN
*ColliderML-Release-1* sample. Events are available pileup-free (`_pu0`), overlaid with **200
pileup interactions per bunch crossing** (`_pu200`), and as **pileup-only** samples
(`pileup_only_pu0`). Each event provides three record types: **generator/simulation particles**
(with full parentage and production vertices), **reconstructed tracks**, and **calorimeter hits**
(cells carrying energy plus the truth list of which particles deposited in them).

Preprocessing turns this into four flat, ML-ready Parquet tables per shard. It does three
conceptually distinct jobs:

1. **Define the truth target particles** the network must reconstruct — a physically motivated
   subset of the simulated particles ([§6](#6-target-particle-definition)).
2. **Cluster the calorimeter hits** into energy clusters that serve as the calorimeter "nodes"
   seen by the network ([§7](#7-calorimeter-clustering)).
3. **Attribute calorimeter energy** back to the target particles, building the per-(cluster,
   particle) deposit table that becomes the network's incidence (assignment) ground truth
   ([§8](#8-particle-energy-deposits-incidence)).

The output feeds the three-stream reconstruction model (documented separately); this note covers
only dataset construction.

---

## 2. Input data & pipeline

**Inputs.** One set of sharded Parquet files per event type. Each shard holds, per event,
list-valued columns:

| Record | Fields used |
|---|---|
| **particles** | `particle_id`, `parent_id`, `pdg_id`, `vertex_primary`, `energy`, `px,py,pz`, production vertex `vx,vy,vz` |
| **tracks** | `majority_particle_id`, `d0`, `z0`, `phi`, `theta`, `qop`, `track_id`, `hit_ids` |
| **calo_hits** | per-cell `x,y,z`, `detector`, `total_energy`, truth contributions `contrib_particle_ids`, `contrib_energies` — plus `contrib_times`, read **only** for the pileup side of an overlay run ([§9](#9-synthetic-pileup-overlay)) |

Shards are read by predicate pushdown, so only the events being processed are fetched
(`hf_io.scan_events()`). All `Float64` columns are down-cast to `Float32` on load, for memory and
speed (`pipeline.prepare_source()`).

**Pipeline stages.** The work factors into three stages. Every mode shares A and C; only
`overlay` has a stage B.

```
 raw particles / tracks / calo_hits          (per source: primary, and pileup if overlaying)
        │
        │  ── STAGE A ── pipeline.prepare_source()
        │  Float64 → Float32 cast
        ▼
 track kinematics + extrapolation            ── calculate_extrapolated_features_polars()
        │
        ▼
 track selection: pt > 1.0 GeV, |eta| < 3.0, originating vertex joined on
        │
        ▼
 snapshot particle → vertex_primary  (BEFORE any vertex filter; feeds §8 step 3)
        │
        ▼
 vertex policy: keep vertex_primary == 1, or keep all      (see §3)
        │
        ▼
 particle masks: orphan, created_inside_calo, has_track, enter_calo, eta/phi/pT
        │         ── add_orphan_mask(), add_created_inside_calo_mask(),
        │            add_particle_have_track_mask(), add_eta_and_phi_and_pt(),
        │            get_particles_id_parent_of_inside_calo_particles_maskv3()
        ▼
 TARGET PARTICLE selection                   ── set_target_particles_maskv4()
        │
        │  ── STAGE B (overlay mode only) ── overlay.run_overlay()
        │  Poisson sampling, time-of-flight cut, cell-wise energy merge, track append
        ▼
        │  ── STAGE C ── pipeline.run_tail()
 CLUE calorimeter clustering                 ── clue_clustering()  (voxelize_hits + CLUEstering)
        │
        ▼
 drop sub-threshold clusters and CLUE outliers ── apply_cluster_energy_cutoff()
        │
        ▼
 cluster features                            ── create_calo_clusters()
        │
        ▼
 energy deposits:                            ── cluster_contrib_energy()
   • per-target incidence                    ── cluster_purity() (+ backtrack_to_target())
   • per-cluster HS vs pileup energy         ── cluster_vertex_primary_deps()
        │
        ▼
 drop orphan targets + reindex                ── filter_orphans_and_reindex()
        │
        ▼
 4 Parquet tables (target_particles, calo_clusters, tracks, target_particles_deps)
```

**Global cuts** (`config.Cuts`, values from `configs/*.yaml`): truth-particle transverse momentum
`truth_pt = 1.0` GeV, pseudorapidity acceptance `truth_eta = 3.0`, target-particle momentum
`target_pt = 0.3` GeV, and minimum calibrated cluster energy `cluster_energy = 0.15` GeV.

---

## 3. The three modes

The modes differ in which particles become targets, and whether pileup is real or synthetic.

| mode | input sample | vertex policy | pileup |
|---|---|---|---|
| `hard_scatter` | `ttbar_pu200` | `vertex_primary == 1` | real, present in the sample |
| `all_vertices` | `ttbar_pu200` | all vertices kept | real, present in the sample |
| `overlay` | `ttbar_pu0` + `pileup_only_pu0` | `vertex_primary == 1` | synthetic, overlaid ([§9](#9-synthetic-pileup-overlay)) |

In `hard_scatter`, only the hard-scatter interaction is reconstructed; pileup deposits energy but
is never a target. In `all_vertices` the vertex filter is skipped, so pileup particles are targets
too and the network must separate hard scatter from pileup rather than being handed the
distinction.

`overlay` is `hard_scatter` **plus** stage B, not a third vertex policy: its hard-scatter side
applies the same `vertex_primary == 1` filter. On `ttbar_pu0` input that filter is a no-op — every
particle already belongs to the single primary vertex — but it is applied, and the particle→vertex
snapshot it feeds is what allows overlaid clusters to be decomposed by originating vertex.

The policy is derived from `mode` (`config.Config.keep_all_vertices`) rather than configured
independently, so a run cannot silently diverge from the published datasets.

---

## 4. Hard scatter vs pileup labelling

Every simulated particle carries a **`vertex_primary`** index identifying the interaction vertex
it came from:

- **`vertex_primary == 1` ⇒ hard scatter (HS)** — the physics interaction of interest;
- **`vertex_primary != 1` ⇒ pileup** — one of the ~200 background interactions.

Where the vertex filter applies, target selection runs **only on hard-scatter particles**. Pileup
particles are not discarded outright: a `particle_id → vertex_primary` map is snapshotted
**before** the filter (`pipeline.prepare_source()`), so the pileup energy contaminating each
cluster can still be quantified afterwards (`cluster_vertex_primary_deps()`, see
[§8](#8-particle-energy-deposits-incidence) step 3). Taking that snapshot pre-filter is essential —
after filtering, the vertex origin of pileup depositors would be unrecoverable.

---

## 5. Stable particles

A particle is treated as **stable** — propagating to and showering in the calorimeter rather than
decaying first — from a fixed list of PDG codes, *not* from a generator status word: the sample
carries no `generatorStatus` field.

- **Promptly decaying / unstable** (`particles_decaying_immediately`, exposed as
  `unstable_pdg_ids_df` in [`pdg.py`](../colliderml_pflow/pdg.py)): quarks, gluon, W/Z/H, π⁰,
  K⁰_S, τ, charm and bottom hadrons, hyperons (Λ, Σ, Ξ, Ω), and resonances.
- Everything else — e^±, neutrinos, μ^±, photon, K⁰_L, π^±, K^±, neutron, proton, their
  antiparticles, and nuclei — is treated as stable.

The distinction matters because an unstable particle deposits calorimeter energy only *through its
decay products*. Were it kept as a target, its daughters' shower energy would be wrongly
attributed to it. The selection therefore removes untracked unstable particles and re-attributes
their deposits to stable ancestors or daughters ([§6](#6-target-particle-definition), rules 4–5).

---

## 6. Target particle definition

The **target particles** are the truth objects the network is trained to reconstruct. They are
selected by `set_target_particles_maskv4()`
([`preprocessing.py`](../colliderml_pflow/preprocessing.py)). A particle is a target **iff all**
of the following hold:

| # | Rule | Implemented by |
|---|---|---|
| 1 | **Not a neutrino:** `\|pdg_id\| ∉ {12, 14, 16}` | filter in `set_target_particles_maskv4()` |
| 2 | **Visible:** it `enter_calo` **or** `has_track` — it reaches the calorimeter or leaves a track | masks below, combined in `set_target_particles_maskv4()` |
| 3 | **Not shadowed:** it has **no ancestor that itself has a track**, unless the particle *is* the tracked one — so each tracked particle, and each calorimeter-facing neutral, is represented exactly once | `map_to_nearest_ancestor_with_track()` + filter |
| 4 | **Stability:** keep if `has_track` **or** `pdg_id ∉ unstable_list` — untracked unstable particles are dropped so their daughters' shower energy is not mis-assigned | `unstable_pdg_ids_df` filter |
| 5 | **Back-track untracked targets** to their **target root**, the greatest still-valid stable ancestor | `backtrack_to_target_roots()` |
| 6 | **Kinematic acceptance** (below) | pT/η filter in `set_target_particles_maskv4()` |

**Kinematic acceptance (rule 6).** A candidate is accepted if *either*:

```
(pt_truth > 1.0 GeV)  AND  (pt_target > 0.3 GeV)  AND  (|eta_truth| < 3.0)
        OR
(has a track)  AND  (track_pt > 0.3 GeV)  AND  (|track_eta| < 3.0)
```

`pt_truth`/`eta_truth` are the kinematics of the truth root ancestor (found by
`backtrack_to_target()`), `pt_target` those of the target particle itself, and `track_pt`/
`track_eta` those of the matched track. The tracked branch keeps low-energy *charged* particles
that nonetheless leave a good track.

**Supporting masks** (all in [`preprocessing.py`](../colliderml_pflow/preprocessing.py)):

- **`has_track`** — the particle's `particle_id` appears as some track's `majority_particle_id`
  (`add_particle_have_track_mask()`).
- **`created_inside_calo`** — the production vertex lies *outside* the tracker volume, i.e.
  `¬( (vx² + vy²) < 1080² mm² AND |vz| < 3030 mm )` (`add_created_inside_calo_mask()`).
- **`enter_calo`** — the particle is the first ancestor *outside* the calorimeter of a cell that
  received energy: the object that actually entered the calorimeter and started the shower
  (`get_particles_id_parent_of_inside_calo_particles_maskv3()` →
  `map_calo_depositors_to_first_outside_ancestor()`).
- **η, φ, pT** are derived from `(px, py, pz)` by `add_eta_and_phi_and_pt()`; missing-parent flags
  by `add_orphan_mask()`.

**Worked intuition.** A charged pion that leaves a track is a target via the tracked branch. A
photon that converts only in the calorimeter has no track but does `enter_calo`, is stable, and
passes the truth-pT cut — it is a target. A short-lived K⁰_S decaying in the tracker is *not* a
target; instead its visible, stable decay products that entered the calorimeter are, after the
back-tracking of rules 4–5.

---

## 7. Calorimeter clustering

Calorimeter cells are grouped into **clusters** acting as the calorimeter input objects (nodes) of
the network. Performed by `clue_clustering()`
([`clustering.py`](../colliderml_pflow/clustering.py)).

**Step 1 — Voxelization** (`voxelize_hits()`). Cells are down-sampled onto a regular spatial grid
per sub-detector, reducing hit multiplicity and equalising the very different ECal and HCal cell
sizes. Voxel edge lengths (`voxel_config`,
[`calibration.py`](../colliderml_pflow/calibration.py)):

| Sub-detector (`detector` id) | Voxel size |
|---|---|
| ECal endcap (9, 11) | 25 mm |
| ECal barrel (10) | 60 mm |
| HCal barrel/endcap (12, 13, 14) | 60 mm |

Within each voxel, cell energies are summed and positions averaged.

**Step 2 — Energy calibration.** Cell energy is calibrated as
`energy = total_energy × calib_factor × 1000` (MeV), with per-sub-detector factors from the
`CALIBRATION` table ([`calibration.py`](../colliderml_pflow/calibration.py)):

| Sub-detector | `calib_factor` |
|---|---|
| ECal barrel (10) | 37.5 |
| ECal endcap (9, 11) | 38.7 |
| HCal barrel (13) | 45.0 |
| HCal endcap (12, 14) | 46.9 |

**Step 3 — CLUE clustering.** Calibrated voxels are clustered **per event in full 3D `(x, y, z)`
with energy as the point weight**, using the CLUE algorithm (CLUEstering library). CLUE is
density-based: for each point it computes a local energy density within radius `dc`, links each
point to its nearest neighbour of higher density within `dm`, promotes points whose density
exceeds `rhoc` to cluster **seeds**, and assigns the rest by following nearest-higher chains.
Points that are neither seeds nor reachable are labelled outliers (`cluster_id = −1`).

| Parameter | Value | Meaning |
|---|---|---|
| `dc` | 75.881 | local-density radius |
| `rhoc` | 104.343 | seed (critical) density threshold |
| `dm` | 87.097 | max distance to nearest-higher |
| `ppbin` | 16 | points-per-bin tiling parameter |

Voxel-level `cluster_id` and cluster centroid are mapped back onto the **original** cells via the
shared voxel index, so every cell — and hence every truth contribution — carries a cluster label.

**Step 4 — Cluster selection** (`apply_cluster_energy_cutoff()`,
[`pipeline.py`](../colliderml_pflow/pipeline.py)). Outlier cells (`cluster_id < 0`) are dropped,
as are all cells of any cluster whose total calibrated energy is at or below
`cluster_energy = 0.15` GeV.

**Step 5 — Cluster features.** Per-cluster quantities are aggregated by `create_calo_clusters()`
([`aggregate.py`](../colliderml_pflow/aggregate.py)); the variables are listed in
[§10.2](#102-cluster-variables).

**A note on determinism.** CLUE assigns cluster ids in discovery order and its CUDA backend
reduces nondeterministically, so cluster *labels* are not reproducible between runs even on
identical input. Cluster *counts* and total energies are stable to a few parts in 10⁵. The point
order reaching CLUE is pinned (`clustering.deterministic`, default on); with a CPU backend that
makes clustering bit-reproducible. See the README for measured figures.

---

## 8. Particle energy deposits (incidence)

The network's assignment target ("incidence") requires knowing **how much calibrated energy each
target particle deposited in each cluster**. This is built from the Geant4 truth contribution
lists attached to every calorimeter cell.

**Step 1 — Per-(cluster, particle) energy** (`cluster_contrib_energy()`,
[`preprocessing.py`](../colliderml_pflow/preprocessing.py)). Each cell carries parallel lists
`contrib_particle_ids` and `contrib_energies`. These are double-exploded (cell → contribution),
calibrated (`× calib_factor`), and summed to give the calibrated energy each *individual* particle
deposited in each cluster.

**Step 2 — Attribute to targets** (`cluster_purity()` with `backtrack_to_target()`). Each
contributing particle is walked up its parentage chain until it reaches a **target particle** — its
"ultimate ancestor" in the target set. Energies are summed per (event, cluster, target), yielding
internally `total_energy_deps_in_cluster` (that target's energy in that cluster),
`total_energy_deps` (its total across clusters) and `purity = in_cluster / total`. Of these, **only
`total_energy_deps_in_cluster`** is written, paired with the target's `particle_idx` and the
`cluster_idx` — a sparse list of `(particle, cluster, energy)` triplets. This is exactly the
information from which the model builds its column-normalised particle↔node incidence matrix.

**Step 3 — Per-cluster hard-scatter vs pileup split** (`cluster_vertex_primary_deps()`). The same
calibrated contribution energies are summed by `vertex_primary` instead of by particle, producing
for every cluster the parallel lists `vertex_primary_indices` / `vertex_primary_energies`. The
entry for vertex 1 is the cluster's **hard-scatter energy**; the remainder is pileup. Downstream
this gives each cluster's HS energy and HS energy fraction, used as the calorimeter HS/pileup
classification target. This step depends on the pre-filter snapshot from
[§4](#4-hard-scatter-vs-pileup-labelling).

**Step 4 — Orphan removal** (`filter_orphans_and_reindex()`,
[`aggregate.py`](../colliderml_pflow/aggregate.py)). Target particles ending up with *neither* a
track *nor* any cluster deposit carry no learnable signal and are removed — nothing in the detector
records them, so the network cannot be asked to find them. Particle and cluster indices are then
reindexed to a contiguous per-event range. Typically ~0.2% of target energy is removed this way.

In overlay mode this step also handles a subtlety: `particle_id` is event-local and reused across
source events, so after overlay a pileup track whose `majority_particle_id` collides with a
hard-scatter target's id would be spuriously wired to that target. Pileup tracks, identified by a
non-null `source_pileup_event_id`, are therefore forced to the `particle_idx = −1` sentinel, which
makes `particle_idx >= 0` a clean "hard-scatter track" flag.

---

## 9. Synthetic pileup overlay

`overlay` mode builds a PU200-like sample without simulating PU200: each pileup-free hard-scatter
event has `N ~ Poisson(μ)` pileup events drawn from a shared pool overlaid onto it. This decouples
the physics process from the pileup level — the same hard-scatter events can be studied at several
values of μ — and makes the pileup content of every cluster exactly known. Implemented in
[`overlay.py`](../colliderml_pflow/overlay.py).

**Step 1 — Sampling** (`build_sample_map()`). For each hard-scatter event, draw `N ~ Poisson(μ)`
and choose N *distinct* pileup events from the pool. Distinctness holds within a hard-scatter
event; the same pileup event may be reused across different ones, which is why a pool of a few
hundred events suffices. The pool is enumerated from the pileup **particles**, not its calorimeter
hits, so vertices that deposited no energy are still sampled at their Poisson rate and correctly
contribute nothing.

**Step 2 — Invisible vertices.** A fraction of real interactions are diffractive and leave no
detector signature. `invisible_pu_prob` thins the draw as `K ~ Binomial(N, 1 − p)` — applied as a
thinning rather than by sampling and discarding, so no work is wasted. The measured value is 0.19;
the shipped preset uses 0.0.

**Step 3 — Time-of-flight cut** (`overlay_calo_hits()`). In a real bunch crossing, pileup
interactions are spread in time, so hits arriving outside the read-out window are never recorded.
Every simulated PU0 event instead sits at `t = 0`, so naive overlay inflates pileup energy by
roughly 8%. Each sampled pileup vertex is therefore given a Gaussian time offset
`Δt ~ N(0, σ)` with `σ = 0.185` ns, and each of its hits is kept only if

```
t_corr = t_hit + Δt − |r| / c      lies within  [−1.0, 10.0] ns
```

where `t_hit` is the hit's energy-weighted mean contribution time (precomputed by
`pipeline._precompute_hit_times()`), `|r| = √(x²+y²+z²)`, and `c = 299.792458` mm/ns. The cut is
applied to the **pileup side only**: hard-scatter hits are at `t = 0` in simulation and were
already windowed there.

**Step 4 — Energy merge** (`overlay_calo_hits()`). Cells are matched on
`(event_id, detector, x, y, z)` with coordinates rounded to 3 decimals and merged with a full
outer join, so pileup-only cells survive as new hits. Truth contributions are deliberately **not**
carried over from pileup — only its energy is added. Hard-scatter `contrib_particle_ids` /
`contrib_energies` pass through untouched, and pileup-only cells get empty lists. This is what
keeps downstream truth attribution pointing only at hard-scatter particles while the pileup appears
as unattributed energy: precisely the reconstruction problem the model must solve.

**Step 5 — Track merge** (`overlay_tracks()`). Sampled pileup tracks are appended to each
hard-scatter event's track list, hard-scatter tracks first, with a `source_pileup_event_id` column
that is null on hard-scatter rows. See [§8](#8-particle-energy-deposits-incidence) step 4 for how
that column prevents mis-attribution.

**Seeding.** Shard `i` uses `seed + i`; each chunk derives its own seed by hashing
`(shard_seed, chunk_index)` through `numpy.random.SeedSequence`. Hashing rather than adding matters:
`seed + chunk` collides whenever `a + n == b + m`, which would silently reuse a pileup sample
across shards. Both id arrays are sorted before sampling, so the sample is a function of the seed
and pool contents alone. Chunked and unchunked runs draw different samples by construction, since
the map is drawn per chunk.

---

## 10. Network input variables

### 10.1 Track variables

Computed by `calculate_extrapolated_features_polars()`
([`preprocessing.py`](../colliderml_pflow/preprocessing.py)) with magnetic field `B = 3.0` T and
calorimeter-face geometry `R_cal = 1080` mm, `Z_cal = 3030` mm.

| Variable | Definition | Notes |
|---|---|---|
| `d0` | transverse impact parameter | raw track fit |
| `z0` | longitudinal impact parameter | raw track fit |
| `phi` | azimuthal angle at perigee | raw track fit |
| `theta` | polar angle at perigee | raw track fit |
| `qop` | charge / momentum | raw track fit |
| `pt` | `p · sin(theta)`, with `p = \|1/qop\|` | transverse momentum |
| `eta` | `−ln tan(theta/2)` | pseudorapidity |
| `track_tanlambda` | `cot(theta)` | R–z slope |
| `track_omega` | `charge / R_curv`, with `R_curv = pt / (0.0003 · B)` | signed curvature [1/mm] |
| `phi_int`, `eta_int` | helix extrapolation to the calorimeter face (barrel at `R_cal`, else endcap at `Z_cal`) | the track's calorimeter impact position |

`qop` is clamped away from zero in the curvature and momentum formulas to avoid division blow-ups.

Each track additionally carries its originating particle's production vertex `vx, vy, vz` and true
`particle_pt`, joined on in stage A.

### 10.2 Cluster variables

Computed by `create_calo_clusters()` ([`aggregate.py`](../colliderml_pflow/aggregate.py)) from the
calibrated cells of each cluster. Let `cal_E = total_energy × calib_factor` per cell, and let the
cluster centroid from CLUE be `(cx, cy, cz)`.

| Variable | Definition |
|---|---|
| `cluster_phi` | `atan2(cy, cx)` |
| `cluster_eta` | `arcsinh( cz / √(cx² + cy²) )` |
| `cluster_rho` | `√(cx² + cy²)` |
| `total_cluster_energy` | `Σ cal_E` over the cluster's cells |
| `hcal_energy` | `Σ cal_E` over HCal cells only |
| `hcal_fraction` | `hcal_energy / total_cluster_energy` |
| `sigma_eta` | standard deviation of the cells' `hit_eta = arcsinh(z/ρ)` |
| `sigma_phi` | standard deviation of the cells' `hit_phi = atan2(y, x)` |
| `sigma_rho` | standard deviation of the cells' `hit_rho = √(x² + y²)` |
| `number_of_hits` | number of cells in the cluster |
| `energy_hits_std` | standard deviation of the cells' `cal_E` |
| `max_hit_energy` | maximum single-cell `cal_E` |

`sigma_eta/phi/rho` are the calorimeter **shower-shape widths**; with `hcal_fraction` and
`max_hit_energy` they let the network distinguish electromagnetic from hadronic, and broad from
narrow, showers.

**Timing variables are not produced.** A per-cluster time was computed on the `master` branch for
two of the three modes but never for overlay; it is dropped here so all modes share one schema. See
the README for the downstream consequences.

### 10.3 Normalization

Input scaling statistics are accumulated in a single streaming pass over the written shards
(`normalization.generate_normalization_stats()`), using KLL sketches for quantiles so memory stays
bounded. Each feature is recorded with its scaling scheme (`min_max_sym` or `std`), an optional
pre-transform (`sqrt` for energy-like quantities), and its mean, standard deviation, q25/q75/q95/q99
and min/max.

---

## 11. Parameter & threshold reference

| Parameter | Value | Role | Where |
|---|---|---|---|
| `cuts.truth_pt` | 1.0 GeV | min truth-ancestor pT for a target; min track pT | `set_target_particles_maskv4()`, `prepare_source()` |
| `cuts.truth_eta` | 3.0 | \|η\| acceptance | `set_target_particles_maskv4()`, `prepare_source()` |
| `cuts.target_pt` | 0.3 GeV | min target / track pT | `set_target_particles_maskv4()` |
| `cuts.cluster_energy` | 0.15 GeV | min calibrated cluster energy | `apply_cluster_energy_cutoff()` |
| tracker radius | 1080 mm | `created_inside_calo` boundary | `add_created_inside_calo_mask()` |
| tracker half-length | 3030 mm | `created_inside_calo` boundary | `add_created_inside_calo_mask()` |
| `B_field` | 3.0 T | track curvature / extrapolation | `calculate_extrapolated_features_polars()` |
| `R_cal`, `Z_cal` | 1080 mm, 3030 mm | calorimeter face for extrapolation | `calculate_extrapolated_features_polars()` |
| CLUE `dc`/`rhoc`/`dm`/`ppbin` | 75.881 / 104.343 / 87.097 / 16 | clustering | `clue_clustering()` |
| voxel sizes | 25 mm (ECal endcap), 60 mm (rest) | voxelization | `voxel_config` |
| calibration factors | 37.5 / 38.7 / 45.0 / 46.9 | ECal barrel / ECal endcap / HCal barrel / HCal endcap | `CALIBRATION` |
| HS vertex id | `vertex_primary == 1` | hard-scatter label | `prepare_source()` |
| `overlay.pileup_level` | 200 | Poisson mean for synthetic pileup | `build_sample_map()` |
| `overlay.invisible_pu_prob` | 0.0 (0.19 measured) | diffractive fraction contributing nothing | `build_sample_map()` |
| `overlay.tof.sigma_ns` | 0.185 ns | per-vertex bunch-crossing time spread | `overlay_calo_hits()` |
| `overlay.tof.window_ns` | [−1.0, 10.0] ns | read-out acceptance window | `overlay_calo_hits()` |
| speed of light | 299.792458 mm/ns | flight-time correction | `overlay.TOF_C_MM_NS` |

---

## 12. Implementation map

| Step / quantity | Function | Module |
|---|---|---|
| Top-level run over shards | `run_preprocessing()` | `runner.py` |
| Per-batch driver | `preprocess_events()` | `pipeline.py` |
| Stage A: per-source preparation | `prepare_source()` | `pipeline.py` |
| Stage C: cluster + aggregate | `run_tail()` | `pipeline.py` |
| η, φ, pT from momentum | `add_eta_and_phi_and_pt()` | `preprocessing.py` |
| Missing-parent flag | `add_orphan_mask()` | `preprocessing.py` |
| `has_track` mask | `add_particle_have_track_mask()` | `preprocessing.py` |
| `created_inside_calo` mask | `add_created_inside_calo_mask()` | `preprocessing.py` |
| `enter_calo` mask | `get_particles_id_parent_of_inside_calo_particles_maskv3()` → `map_calo_depositors_to_first_outside_ancestor()` | `preprocessing.py` |
| Nearest tracked ancestor | `map_to_nearest_ancestor_with_track()` | `preprocessing.py` |
| **Target selection** | `set_target_particles_maskv4()` | `preprocessing.py` |
| Back-track to target root | `backtrack_to_target_roots()` | `preprocessing.py` |
| Back-track to truth/target | `backtrack_to_target()` | `preprocessing.py` |
| Unstable PDG list | `unstable_pdg_ids_df` | `pdg.py` |
| Track kinematics + extrapolation | `calculate_extrapolated_features_polars()` | `preprocessing.py` |
| Hit voxelization | `voxelize_hits()`, `voxel_config` | `clustering.py`, `calibration.py` |
| Energy calibration table | `CALIBRATION` | `calibration.py` |
| CLUE clustering | `clue_clustering()` (CLUEstering library) | `clustering.py` |
| Cluster energy cutoff | `apply_cluster_energy_cutoff()` | `pipeline.py` |
| Cluster features | `create_calo_clusters()` | `aggregate.py` |
| Per-(cluster, particle) energy | `cluster_contrib_energy()` | `preprocessing.py` |
| Deposits / incidence | `cluster_purity()` | `preprocessing.py` |
| Per-cluster HS vs pileup energy | `cluster_vertex_primary_deps()` | `preprocessing.py` |
| Orphan removal + reindex | `filter_orphans_and_reindex()` | `aggregate.py` |
| Pileup sampling | `build_sample_map()` | `overlay.py` |
| ToF cut + cell energy merge | `overlay_calo_hits()` | `overlay.py` |
| Pileup hit-time precompute | `_precompute_hit_times()` | `pipeline.py` |
| Track merge | `overlay_tracks()` | `overlay.py` |
| Shard reads (predicate pushdown) | `scan_events()`, `load_triplet()`, `load_pileup_pool()` | `hf_io.py` |
| Normalization statistics | `generate_normalization_stats()` | `normalization.py` |
| Batch job splitting | `group_shards()`, `qsub_commands()` | `submit.py` |

---

*All values and rules above were read from the source in `colliderml_pflow/`. Agreement with the
original `master`-branch implementation is verified per mode by `tests/test_equivalence.py`; see
the README for what is compared exactly and what is compared through label-invariant quantities.*
