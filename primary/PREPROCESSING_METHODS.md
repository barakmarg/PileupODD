# Training-Dataset Preprocessing — Methods Summary

This document describes, at a level suitable for a paper methods section, how the training dataset
for the pileup particle-flow study is produced from simulated detector output. It states the exact
definitions, selection rules, and numerical thresholds, and — so each claim can be verified against
the code — annotates every step with the **function (and file)** that implements it.

The full reference index of step → function is collected in [§11](#11-implementation-map).

> **Primary driver:** `preprocess_for_model()` in `create_trainning_dataset_pileup.py`.
> All file references are relative to `PileupODD/primary/`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Input Data & Pipeline](#2-input-data--pipeline)
3. [Hard Scatter vs Pileup Labelling](#3-hard-scatter-vs-pileup-labelling)
4. [Stable Particles](#4-stable-particles)
5. [Target Particle Definition](#5-target-particle-definition)
6. [Calorimeter Clustering](#6-calorimeter-clustering)
7. [Particle Energy Deposits (Incidence)](#7-particle-energy-deposits-incidence)
8. [Network Input Variables](#8-network-input-variables)
9. [Output Tables](#9-output-tables)
10. [Parameter & Threshold Reference](#10-parameter--threshold-reference)
11. [Implementation Map](#11-implementation-map)

---

## 1. Overview

The dataset originates from a **Geant4 simulation of the Open Data Detector (ODD)** — the CERN
*ColliderML-Release-1* sample — for `t\bar{t}` events overlaid with **200 pileup interactions per
bunch crossing (PU200)**. Each simulated event provides three record types: **generator/simulation
particles** (with full parentage and production vertices), **reconstructed tracks**, and
**calorimeter hits** (cells carrying energy plus the truth list of which particles deposited in
them).

Preprocessing turns this raw simulation output into four flat, ML-ready Parquet tables per shard.
It performs three conceptually distinct jobs:

1. **Define the truth target particles** the network must reconstruct — a physically motivated
   subset of the hard-scatter particles (§5).
2. **Cluster the calorimeter hits** into energy clusters that serve as the calorimeter "nodes" seen
   by the network (§6).
3. **Attribute calorimeter energy** back to the target particles, building the per-(cluster,
   particle) deposit table that becomes the network's incidence (assignment) ground truth (§7).

The output feeds the three-stream reconstruction model (separate documentation); this note covers
only the dataset construction.

---

## 2. Input Data & Pipeline

**Inputs** (one set of sharded Parquet files per event type; loaded by the chunk workers and the
top-level runner). Each shard holds, per event, list-valued columns:

| Record | Key fields used |
|---|---|
| **particles** | `particle_id`, `parent_id`, `pdg_id`, `vertex_primary`, `energy`, `px,py,pz`, production vertex `vx,vy,vz` |
| **tracks** | `majority_particle_id`, `d0`, `z0`, `phi`, `theta`, `qop` |
| **calo_hits** | per-cell `x,y,z`, `detector`, `total_energy`, and truth contributions `contrib_particle_ids`, `contrib_energies` (`contrib_times` exists but is unused here) |

All `Float64` columns are down-cast to `Float32` on load for memory/speed
(*`preprocess_for_model()`*, `create_trainning_dataset_pileup.py`).

**Pipeline stages** (in execution order, all orchestrated by *`preprocess_for_model()`*):

```
 raw particles / tracks / calo_hits
        │
        │  Float64 → Float32 cast
        ▼
 track kinematics + extrapolation        ── calculate_extrapolated_features_polars()
        │
        ▼
 particle masks:  has_track, created_inside_calo, enter_calo, η/φ/pT, parentage
        │         ── add_particle_have_track_mask(), add_created_inside_calo_mask(),
        │            get_particles_id_parent_of_inside_calo_particles_maskv3(),
        │            add_eta_and_phi_and_pt(), add_orphan_mask()
        ▼
 keep hard-scatter particles (vertex_primary == 1)
        │
        ▼
 TARGET PARTICLE selection               ── set_target_particles_maskv4()
        │
        ▼
 CLUE calorimeter clustering             ── clue_clustering()  (voxelize_hits + CLUEstering)
        │
        ▼
 cluster features                        ── create_calo_clusters()
        │
        ▼
 energy deposits:                         ── cluster_contrib_energy()
   • per-target purity / incidence       ── cluster_purity()   (+ backtrack_to_target())
   • per-cluster HS vs pileup energy      ── cluster_vertex_primary_deps()
        │
        ▼
 drop orphan targets + reindex           ── filter_orphans_and_reindex()
        │
        ▼
 4 Parquet tables (target_particles, calo_clusters, tracks, target_particles_deps)
```

**Global event-level cuts** (driver defaults): truth-particle transverse momentum
`truth_pt_cut = 1.0` GeV, pseudorapidity acceptance `truth_eta_cut = 3.0`, target-particle momentum
`target_pt_cut = 0.3` GeV, and a minimum calibrated cluster energy `clusters_cutoff` (default
**0.15** GeV in the production runner; clusters below it are discarded — filter in
*`preprocess_for_model()`*).

---

## 3. Hard Scatter vs Pileup Labelling

Every simulated particle carries a **`vertex_primary`** index identifying which interaction vertex
it came from. The convention here is:

- **`vertex_primary == 1` ⇒ hard scatter (HS)** — the physics interaction of interest;
- **`vertex_primary != 1` ⇒ pileup** — one of the ~200 background interactions.

Target-particle selection is run **only on the hard-scatter particles**: the HS particles are
filtered out first (a `vertex_primary == 1` gather in *`preprocess_for_model()`*). Pileup particles
are *not* discarded outright — they are retained in a separate `particle_id → vertex_primary` map so
that the pileup energy contaminating each calorimeter cluster can later be quantified
(*`cluster_vertex_primary_deps()`*, `preprocessing.py`; see §7).

---

## 4. Stable Particles

A particle is treated as **stable** (i.e. it propagates to and showers in the calorimeter rather
than decaying first) based on a fixed list of PDG codes, *not* on a generator status word (the
sample carries no `generatorStatus` field).

- **Stable** (`STABLE_PDG_IDS` in `pdg_mappings.py`): e^±, ν_e, μ^±, ν_μ, ν_τ, photon (22),
  K^0_L (130), π^± (211), K^± (321), neutron (2112), proton (2212) and their antiparticles, plus
  nuclei/ions (PDG ≥ 10^9).
- **Promptly decaying / unstable** (`particles_decaying_immediately`, exposed as
  `unstable_pdg_ids_df` in `pdg_mappings.py`): quarks, gluon, W/Z/H, π^0, K^0_S, τ, charm and
  bottom hadrons, hyperons (Λ, Σ, Ξ, Ω), and resonances.

The stability list matters because an unstable particle deposits energy in the calorimeter only
*through its decay products*. If such a particle were kept as a target, the shower energy of its
daughters would be wrongly attributed to it. The selection therefore removes untracked unstable
particles and re-attributes their deposits to stable ancestors/daughters (§5, rules 4–5).

---

## 5. Target Particle Definition

The **target particles** are the truth objects the network is trained to reconstruct. They are
selected by *`set_target_particles_maskv4()`* in `preprocessing.py` (invoked from
*`preprocess_for_model()`*), operating on the hard-scatter particles. A particle is a target **iff
all** of the following hold:

| # | Rule | Implemented by |
|---|---|---|
| 1 | **Not a neutrino:** `\|pdg_id\| ∉ {12, 14, 16}` | filter in `set_target_particles_maskv4()` |
| 2 | **Visible:** it `enter_calo` **or** `has_track` (reaches the calorimeter or leaves a track) | masks below, combined in `set_target_particles_maskv4()` |
| 3 | **Not shadowed:** it has **no ancestor that itself has a track**, unless the particle *is* the tracked one — i.e. each tracked particle, and each calorimeter-facing neutral, is represented exactly once | `map_to_nearest_ancestor_with_track()` + filter in `set_target_particles_maskv4()` |
| 4 | **Stability:** keep if `has_track` **or** `pdg_id ∉ unstable_list` — untracked unstable particles are dropped so their daughters' shower energy is not mis-assigned | `unstable_pdg_ids_df` filter in `set_target_particles_maskv4()` |
| 5 | **Back-track untracked targets** to their **target root** (the greatest still-valid stable ancestor) | `backtrack_to_target_roots()` |
| 6 | **Kinematic acceptance** (see formula below) | pT/η filter in `set_target_particles_maskv4()` |

**Kinematic acceptance (rule 6).** A candidate is accepted if *either*:

```
(pt_truth > 1.0 GeV)  AND  (pt_target > 0.3 GeV)  AND  (|eta_truth| < 3.0)
        OR
(has a track)  AND  (track_pt > 0.3 GeV)  AND  (|track_eta| < 3.0)
```

where `pt_truth`/`eta_truth` are the kinematics of the truth root ancestor (found by
*`backtrack_to_target()`*), `pt_target` is the kinematics of the target particle itself, and
`track_pt`/`track_eta` are the matched track's kinematics. The tracked branch ensures low-energy
*charged* particles that nonetheless leave a good track are kept.

**Supporting masks** (all in `preprocessing.py`):

- **`has_track`** — a particle is "tracked" if its `particle_id` appears as a track's
  `majority_particle_id` (*`add_particle_have_track_mask()`*).
- **`created_inside_calo`** — the particle's production vertex lies *outside* the tracker volume,
  i.e. `¬( (vx² + vy²) < 1080² mm²  AND  |vz| < 3030 mm )` (*`add_created_inside_calo_mask()`*).
- **`enter_calo`** — the particle is the first ancestor *outside* the calorimeter of a cell that
  received energy, i.e. the object that actually entered the calorimeter and started the shower
  (*`get_particles_id_parent_of_inside_calo_particles_maskv3()`* →
  *`map_calo_depositors_to_first_outside_ancestor()`*).
- **η, φ, pT** are added from `(px, py, pz)` by *`add_eta_and_phi_and_pt()`*; missing-parent flags
  by *`add_orphan_mask()`*.

**Worked intuition.** A hard-scatter charged pion that leaves a track is a target via the tracked
branch. A hard-scatter photon that converts only in the calorimeter has no track but `enter_calo`,
is stable, and passes the truth-pT cut — it is a target. A short-lived K^0_S that decays in the
tracker is *not* a target; instead its visible, stable decay products (which entered the
calorimeter) are, after the back-tracking of rules 4–5.

---

## 6. Calorimeter Clustering

Calorimeter cells are grouped into **clusters** that act as the calorimeter input objects (nodes)
of the network. Clustering is performed by *`clue_clustering()`* (`clue_clustering.py`).

**Step 1 — Voxelization** (*`voxelize_hits()`*, `downsample.py`). Cells are first down-sampled onto
a regular spatial grid per sub-detector to reduce hit multiplicity. Voxel edge lengths (`v_size`,
from `voxel_config`):

| Sub-detector (`detector` id) | Voxel size |
|---|---|
| ECAL endcap (9, 11) | 25 mm |
| ECAL barrel (10) | 60 mm |
| HCAL barrel/endcap (12, 13, 14) | 60 mm |

Within each voxel the cell energies are summed and the positions averaged.

**Step 2 — Energy calibration.** Cell energy is calibrated to MeV as
`energy = total_energy × calib_factor × 1000`, with per-sub-detector factors (`CALIBRATION` table,
`calibration.py`):

| Sub-detector | `calib_factor` |
|---|---|
| ECAL barrel (10) | 37.5 |
| ECAL endcap (9, 11) | 38.7 |
| HCAL barrel (13) | 45.0 |
| HCAL endcap (12, 14) | 46.9 |

**Step 3 — CLUE clustering.** The calibrated voxels are clustered **per event in full 3D
`(x, y, z)`, with energy as the point weight**, using the CLUE algorithm (CLUEstering library, run
through *`clue_clustering()`*). CLUE is a density-based algorithm: for each point it computes a
local energy density within radius `dc`, links each point to its nearest neighbour of higher density
within distance `dm`, promotes points whose density exceeds `rhoc` to cluster **seeds**, and assigns
the rest by following the nearest-higher chains; points that are neither seeds nor reachable are
labelled outliers (`cluster_id = −1`). Parameters used:

| Parameter | Value | Meaning |
|---|---|---|
| `dc` | 75.881 | local-density radius |
| `rhoc` | 104.343 | seed (critical) density threshold |
| `dm` | 87.097 | max distance to nearest-higher |
| `ppbin` | 16 | points-per-bin tiling parameter |

The voxel-level `cluster_id` and cluster centroid are then mapped back onto the original cells via
the voxel index (join in *`clue_clustering()`*). Outlier cells (`cluster_id < 0`) are dropped, and
clusters below `clusters_cutoff` in calibrated energy are removed (§2).

**Step 4 — Cluster features.** Per-cluster quantities are aggregated by *`create_calo_clusters()`*
(`create_trainning_dataset_pileup.py`); the resulting variables are listed in §8.

---

## 7. Particle Energy Deposits (Incidence)

The network's assignment target ("incidence") requires knowing **how much calibrated energy each
target particle deposited in each cluster**. This is built from the Geant4 truth contribution lists
attached to every calorimeter cell.

**Step 1 — Per-(cluster, particle) energy** (*`cluster_contrib_energy()`*, `preprocessing.py`).
Each cell carries parallel lists `contrib_particle_ids` and `contrib_energies`. These are
double-exploded (cell → contribution), calibrated (`× calib_factor`), and summed to give the
calibrated energy each *individual* particle deposited in each cluster.

**Step 2 — Attribute to targets** (*`cluster_purity()`* with *`backtrack_to_target()`*). Each
contributing particle is walked up its parentage chain until it reaches a **target particle** (its
"ultimate ancestor" in the target set). Energies are then summed per (event, cluster, target),
yielding internally `total_energy_deps_in_cluster` (energy that target deposited in that cluster),
`total_energy_deps` (the target's total across all clusters) and `purity = in_cluster / total`. Of
these, **only `total_energy_deps_in_cluster`** is kept in the written file, paired with the target's
`particle_idx` and the `cluster_idx` — i.e. a sparse list of `(particle, cluster, energy)` triplets.
This per-(cluster, target) deposit is exactly the information from which the model builds its
column-normalised particle↔node incidence matrix.

**Step 3 — Per-cluster hard-scatter vs pileup split**
(*`cluster_vertex_primary_deps()`*, `preprocessing.py`). The same calibrated contribution energies
are summed by `vertex_primary` instead of by particle, producing for every cluster two parallel
lists `vertex_primary_indices` / `vertex_primary_energies`. The entry for vertex 1 is the cluster's
**hard-scatter energy**; the remainder is pileup. Downstream this gives each cluster's HS energy and
HS energy fraction (used as the calorimeter HS/pileup classification target).

**Step 4 — Orphan removal** (*`filter_orphans_and_reindex()`*). Target particles that end up with
*neither* a track *nor* any cluster deposit carry no learnable signal and are removed; particle and
cluster indices are then reindexed to a contiguous range.

---

## 8. Network Input Variables

All input variables are derived quantities computed below. **Timing variables are intentionally
omitted** (a `cluster_time` is computed in the code but is not used by the network and is not
documented here).

### 8.1 Track variables

Computed by *`calculate_extrapolated_features_polars()`* (`preprocessing.py`) with magnetic field
`B = 3.0` T and calorimeter-face geometry `R_cal = 1080 mm`, `Z_cal = 3030 mm`.

| Variable | Definition | Notes |
|---|---|---|
| `d0` | transverse impact parameter | raw track fit |
| `z0` | longitudinal impact parameter | raw track fit |
| `phi` | azimuthal angle at perigee | raw track fit |
| `theta` | polar angle at perigee | raw track fit |
| `qop` | charge / momentum | raw track fit |
| `p` | `\|1/qop\|` | total momentum |
| `charge` | `sign(qop)` | |
| `pt` | `p · sin(theta)` | transverse momentum |
| `eta` | `−ln tan(theta/2)` | pseudorapidity |
| `track_tanlambda` | `cot(theta)` | R–z slope |
| `track_omega` | `charge / R_curv`, with `R_curv = pt / (0.0003 · B)` | signed curvature [1/mm] |
| `phi_int`, `eta_int` | helix extrapolation of the track to the calorimeter face (barrel at `R_cal`, else endcap at `Z_cal`) | impact-point direction; gives the track's calorimeter position |

(The `qop` used in the curvature/momentum formulas is clamped away from zero to avoid division
blow-ups.)

### 8.2 Cluster variables

Computed by *`create_calo_clusters()`* (`create_trainning_dataset_pileup.py`) from the calibrated
cells of each cluster. Let `cal_E = total_energy × calib_factor` per cell, and let the cluster
centroid be `(cx, cy, cz)` (from CLUE).

| Variable | Definition |
|---|---|
| `cluster_phi` | `atan2(cy, cx)` |
| `cluster_eta` | `arcsinh( cz / √(cx² + cy²) )` |
| `cluster_rho` | `√(cx² + cy²)` |
| `total_cluster_energy` | `Σ cal_E` over the cluster's cells |
| `hcal_energy` | `Σ cal_E` over HCAL cells only |
| `hcal_fraction` | `hcal_energy / total_cluster_energy` |
| `sigma_eta` | standard deviation of the cells' `hit_eta = arcsinh(z/ρ)` |
| `sigma_phi` | standard deviation of the cells' `hit_phi = atan2(y, x)` |
| `sigma_rho` | standard deviation of the cells' `hit_rho = √(x² + y²)` |
| `number_of_hits` | number of cells in the cluster |
| `energy_hits_std` | standard deviation of the cells' `cal_E` |
| `max_hit_energy` | maximum single-cell `cal_E` |

`sigma_eta/phi/rho` are the calorimeter **shower-shape widths**; together with `hcal_fraction` and
`max_hit_energy` they let the network distinguish electromagnetic from hadronic and broad from
narrow showers.

---

## 9. Output Tables — Exact Schema & Examples

Four Parquet files are written per shard (assembled in *`preprocess_for_model()`*, finalised by
*`filter_orphans_and_reindex()`*). **Every file uses one row per event**, with the per-object
quantities stored as equal-length **`List` columns** (element *i* across the lists describes object
*i*); nested `List(List(...))` columns hold per-object variable-length lists. Energies are in GeV
with calorimeter calibration applied (§6). The schemas and example values below are taken directly
from a real shard (`ggf_pu200/…-00000.parquet`); list columns are shown truncated to the first few
elements.

Index conventions used across the tables:
- `particle_idx` — contiguous 0…N−1 index of a **target particle** within the event (the canonical
  particle handle used by the model and the deposits table).
- `cluster_idx` — contiguous 0…M−1 index of a **cluster** within the event (matches the order of
  the `calo_clusters` lists).
- In `tracks`, `particle_idx = −1` flags a track whose majority particle is **not** a target
  (i.e. a pileup/secondary track); `vx/vy/vz/particle_pt` are then null.

### 9.1 `target_particles` — one entry per selected hard-scatter target particle

| Column | Dtype | Meaning |
|---|---|---|
| `event_id` | `UInt32` | event identifier (scalar) |
| `particle_id` | `List(UInt64)` | original simulation particle id |
| `particle_idx` | `List(UInt32)` | contiguous per-event target index (0…N−1) |
| `pdg_id` | `List(Int64)` | PDG code |
| `energy` | `List(Float32)` | truth energy [GeV] |
| `pt` | `List(Float32)` | transverse momentum [GeV] |
| `eta`, `phi` | `List(Float32)` | direction |
| `px, py, pz` | `List(Float32)` | momentum components [GeV] |
| `has_track` | `List(Boolean)` | whether a reconstructed track is matched |
| `vertex_primary` | `List(UInt16)` | vertex id (≡ 1 for these HS targets) |
| `vx, vy, vz` | `List(Float32)` | production vertex [mm] |

**Example (event 1, 187 target particles):**
```
event_id       = 1
particle_id    = [209, 228, 251, 382, …]
particle_idx   = [0,   1,   2,   3,   …]
pdg_id         = [-321, -211, 321, 22, …]      # K-, pi-, K+, photon
energy         = [4.160, 10.164, 3.910, 0.786, …]   GeV
pt             = [3.979, 1.435, 0.610, 0.748, …]    GeV
eta            = [0.275, -2.646, 2.537, 0.321, …]
phi            = [0.880, -0.810, -0.493, 0.343, …]
has_track      = [True, True, True, False, …]       # photon (382) has no track
vertex_primary = [1, 1, 1, 1, …]
vx,vy,vz       = [-0.0095, -0.0024, -78.697, …]     mm   (shared HS vertex)
```

### 9.2 `calo_clusters` — one entry per CLUE cluster

| Column | Dtype | Meaning |
|---|---|---|
| `event_id` | `UInt32` | event identifier (scalar) |
| `cluster_id` | `List(Int32)` | CLUE cluster id (per-event) |
| `total_cluster_energy` | `List(Float64)` | calibrated cluster energy [GeV] |
| `hcal_energy` | `List(Float64)` | calibrated energy in HCAL cells [GeV] |
| `hcal_fraction` | `List(Float64)` | `hcal_energy / total_cluster_energy` |
| `cluster_eta`, `cluster_phi` | `List(Float32)` | centroid direction |
| `cluster_rho` | `List(Float32)` | centroid transverse radius [mm] |
| `sigma_eta`, `sigma_phi`, `sigma_rho` | `List(Float32)` | shower-shape widths |
| `number_of_hits` | `List(UInt32)` | cell multiplicity |
| `energy_hits_std` | `List(Float64)` | std of per-cell energy [GeV] |
| `max_hit_energy` | `List(Float64)` | hottest cell energy [GeV] |
| `cluster_time` | `List(Float64)` | cluster time [ns] — **not used by the network** |
| `vertex_primary_indices` | `List(List(UInt16))` | per-cluster list of contributing vertex ids |
| `vertex_primary_energies` | `List(List(Float32))` | matching calibrated energy per vertex [GeV] |

The paired `vertex_primary_indices`/`vertex_primary_energies` give the HS-vs-pileup energy split per
cluster (the entry with vertex id 1 is the hard-scatter energy; see §7).

**Example (event 1, 4261 clusters):**
```
cluster_id              = [0, 1, 2, 3, …]
total_cluster_energy    = [3.053, 0.614, 1.760, 2.199, …]   GeV
hcal_energy             = [0.0, 0.0, 1.760, 2.199, …]       GeV
hcal_fraction           = [0.0, 0.0, 1.0, 1.0, …]           # 0 = pure ECAL, 1 = pure HCAL
cluster_eta             = [-1.321, 0.825, 2.101, -2.339, …]
cluster_phi             = [-1.296, 2.634, -2.198, -1.342, …]
cluster_rho             = [1284.8, 1416.8, 1017.5, 749.3, …]  mm
sigma_eta               = [0.0243, 0.0048, 0.0483, 0.0478, …]
number_of_hits          = [125, 41, 18, 22, …]
max_hit_energy          = [0.475, 0.136, 0.525, 0.886, …]    GeV
vertex_primary_indices  = [[1, 35, 50], [51, 83], [3, 91, 92], …]
vertex_primary_energies = [[1.072, 0.545, 0.108], [0.004, 0.610], [0.006, 0.017, 0.913], …]  GeV
#   → cluster 0 received 1.072 GeV from the hard scatter (vertex 1) plus pileup from vertices 35, 50
```

### 9.3 `tracks` — one entry per reconstructed track (HS **and** pileup)

| Column | Dtype | Meaning |
|---|---|---|
| `event_id` | `UInt32` | event identifier (scalar) |
| `track_id` | `List(UInt16)` | track id |
| `d0`, `z0` | `List(Float32)` | impact parameters [mm] |
| `phi`, `theta`, `qop` | `List(Float32)` | raw perigee fit parameters |
| `pt`, `eta` | `List(Float32)` | derived kinematics [GeV], — |
| `track_tanlambda` | `List(Float32)` | `cot θ` |
| `track_omega` | `List(Float32)` | signed curvature [1/mm] |
| `phi_int`, `eta_int` | `List(Float32)` | extrapolated calorimeter-face direction |
| `hit_ids` | `List(List(UInt32))` | tracker hit ids on the track |
| `vertex_primary` | `List(UInt16)` | vertex id of the track's majority particle |
| `particle_id` | `List(Int64)` | majority simulation particle id |
| `particle_idx` | `List(Int64)` | matched **target** index, or `−1` if not a target (pileup) |
| `particle_pt` | `List(Float32)` | matched target's pT [GeV]; null if `particle_idx = −1` |
| `vx, vy, vz` | `List(Float32)` | matched target's production vertex; null if `particle_idx = −1` |

**Example (event 1, 1103 tracks):**
```
track_id        = [1, 2, 6, 8, …]
d0              = [-0.143, 0.031, 0.052, 0.003, …]   mm
z0              = [-131.40, -8.98, 49.10, -131.57, …] mm
phi             = [-3.001, -3.049, -2.982, -3.019, …]
theta           = [0.265, 0.582, 3.022, 0.962, …]
qop             = [0.180, -0.418, 0.117, 0.773, …]
pt              = [1.458, 1.315, 1.021, 1.061, …]     GeV
eta             = [2.014, 1.206, -2.815, 0.650, …]
track_tanlambda = [3.679, 1.520, -8.316, 0.697, …]
track_omega     = [6.17e-4, -6.84e-4, 8.81e-4, 8.49e-4, …]  1/mm
vertex_primary  = [100, 87, 32, 100, …]               # all pileup in this head slice
particle_idx    = [-1, -1, -1, -1, …]                 # → not target particles
particle_id     = [52402, 47567, 13371, 53792, …]
```

### 9.4 `target_particles_deps` — sparse particle↔cluster energy deposits

The incidence ground truth, stored as three parallel `List` columns (one row per event); element
*k* says target `particle_idx[k]` deposited `total_energy_deps_in_cluster[k]` GeV into cluster
`cluster_idx[k]`.

| Column | Dtype | Meaning |
|---|---|---|
| `event_id` | `UInt32` | event identifier (scalar) |
| `particle_idx` | `List(UInt32)` | target particle index (matches `target_particles.particle_idx`) |
| `cluster_idx` | `List(UInt32)` | cluster index (matches `calo_clusters` order) |
| `total_energy_deps_in_cluster` | `List(Float32)` | calibrated energy this target put in this cluster [GeV] |

**Example (event 1, 1022 nonzero deposits):**
```
particle_idx                 = [9,    30,   95,    85,    68,   66,  …]
cluster_idx                  = [0,    0,    5,     5,     16,   16,  …]
total_energy_deps_in_cluster = [0.537, 0.535, 0.208, 0.033, 6.800, 0.157, …]   GeV
#   → cluster 0 shares energy between targets 9 and 30; cluster 16 is dominated by target 68 (6.8 GeV)
```

---

## 10. Parameter & Threshold Reference

| Parameter | Value | Role | Where |
|---|---|---|---|
| `truth_pt_cut` | 1.0 GeV | min truth-ancestor pT for a target | `set_target_particles_maskv4()` |
| `truth_eta_cut` | 3.0 | \|η\| acceptance | `set_target_particles_maskv4()` |
| `target_pt_cut` | 0.3 GeV | min target / track pT | `set_target_particles_maskv4()` |
| `clusters_cutoff` | 0.15 GeV (runner) / 0.1 (fn default) | min calibrated cluster energy | `preprocess_for_model()` |
| tracker radius² | 1080² mm² | `created_inside_calo` boundary | `add_created_inside_calo_mask()` |
| tracker half-length | 3030 mm | `created_inside_calo` boundary | `add_created_inside_calo_mask()` |
| `B_field` | 3.0 T | track curvature/extrapolation | `calculate_extrapolated_features_polars()` |
| `R_cal`, `Z_cal` | 1080 mm, 3030 mm | calorimeter face for extrapolation | `calculate_extrapolated_features_polars()` |
| CLUE `dc` / `rhoc` / `dm` / `ppbin` | 75.881 / 104.343 / 87.097 / 16 | clustering | `clue_clustering()` |
| voxel sizes | 25 mm (ECAL endcap), 60 mm (rest) | voxelization | `voxel_config`, `downsample.py` |
| calibration factors | 37.5 / 38.7 / 45.0 / 46.9 | ECAL barrel / ECAL endcap / HCAL barrel / HCAL endcap | `CALIBRATION`, `calibration.py` |
| HS vertex id | `vertex_primary == 1` | hard-scatter label | `preprocess_for_model()` |

---

## 11. Implementation Map

Consolidated index from pipeline step to the function(s) and file that implement it, for
verification.

| Step / quantity | Function | File |
|---|---|---|
| Top-level driver | `preprocess_for_model()` | `create_trainning_dataset_pileup.py` |
| η, φ, pT from momentum | `add_eta_and_phi_and_pt()` | `preprocessing.py` |
| Missing-parent flag | `add_orphan_mask()` | `preprocessing.py` |
| `has_track` mask | `add_particle_have_track_mask()` | `preprocessing.py` |
| `created_inside_calo` mask | `add_created_inside_calo_mask()` | `preprocessing.py` |
| `enter_calo` mask | `get_particles_id_parent_of_inside_calo_particles_maskv3()` → `map_calo_depositors_to_first_outside_ancestor()` | `preprocessing.py` |
| Nearest tracked ancestor | `map_to_nearest_ancestor_with_track()` | `preprocessing.py` |
| **Target selection** | `set_target_particles_maskv4()` | `preprocessing.py` |
| Back-track to target root | `backtrack_to_target_roots()` | `preprocessing.py` |
| Back-track to truth/target | `backtrack_to_target()` | `preprocessing.py` |
| Stable / unstable PDG lists | `STABLE_PDG_IDS`, `particles_decaying_immediately` (`unstable_pdg_ids_df`) | `pdg_mappings.py` |
| Track kinematics + extrapolation | `calculate_extrapolated_features_polars()` | `preprocessing.py` |
| Hit voxelization | `voxelize_hits()`, `voxel_config` | `downsample.py` |
| Energy calibration table | `CALIBRATION` | `calibration.py` |
| CLUE clustering | `clue_clustering()` (CLUEstering library) | `clue_clustering.py` |
| Cluster features | `create_calo_clusters()` | `create_trainning_dataset_pileup.py` |
| Per-(cluster, particle) energy | `cluster_contrib_energy()` | `preprocessing.py` |
| Deposits / purity (incidence) | `cluster_purity()` | `preprocessing.py` |
| Per-cluster HS vs pileup energy | `cluster_vertex_primary_deps()` | `preprocessing.py` |
| Orphan removal + reindex | `filter_orphans_and_reindex()` | `create_trainning_dataset_pileup.py` |

---

*All values and rules above were read directly from the current source
(`create_trainning_dataset_pileup.py`, `preprocessing.py`, `clue_clustering.py`, `downsample.py`,
`calibration.py`, `pdg_mappings.py`). Timing variables, which the network does not use, are omitted
by design.*
