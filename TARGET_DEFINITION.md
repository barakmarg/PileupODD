# Target Definition Summary

This document outlines how target particles are defined for the training dataset generation process. The logic is implemented in `primary/preprocessing.py` (specifically `set_target_particles_maskv2`) and `primary/create_trainning_dataset.py`.

## Overview
Target particles are the "truth" entities the model learns to predict. They are selected based on their detectability (Tracks/Calorimeter) and topological hierarchy to avoid double-counting (e.g., selecting both a parent and its children).

## Definition Logic

A particle is selected as a target if it satisfies the following steps:

### 1. Base Criteria (Measurability)
A particle is initially considered if it meets **at least one** of these conditions:
*   **Has Track (`has_track`)**: The particle is associated with a reconstructed track (via `MajorityParticleID` matching).
*   **Enters Calorimeter (`enter_calo`)**: The particle originates inside the tracker volume (defined as $R < 1080$ mm and $|z| < 3030$ mm) and is the ancestor of one or more calorimeter hits.
    *   *Implementation*: The system backtracks from calorimeter hits to find the first ancestor created within the tracker volume.

### 2. Exclusions
*   **Neutrinos**: Particles with PDG IDs $\pm 12, \pm 14, \pm 16$ are explicitly removed.
*   **Kinematics**:
    *   Transverse Momentum ($p_T$) $> 1.0$ GeV
    *   Pseudorapidity ($|\eta|$) $< 3.0$

### 3. Hierarchy & Veto (The "Tracked Ancestor" Rule)
To establish a clear hierarchy, we prioritize tracked particles:
*   **Veto**: If a particle has an ancestor that **has a track**, the particle is **discarded**.
*   **Exception**: If the particle *itself* has a track, it is kept (it becomes the "highest" representation).

*Effect*: If a tracked particle decays into untracked particles, only the parent is a target. If an untracked particle decays into a tracked particle, the child becomes the target (and the parent is ignored unless it is also the root of other things, see below).

### 4. Grouping of Untracked Candidates
For particles that are **untracked** but **enter the calorimeter** (e.g., photons, neutral hadrons):
*   These particles usually form chains (e.g., Primary Photon $\to$ Conversion $e^+e^-$ pairs).
*   If the $e^+e^-$ tracks are not reconstructed (untracked), we might have multiple valid candidates in a chain.
*   **Root Finding**: The system identifies connected chains of such untracked, calo-entering particles and collapses them to the **Root Ancestor**.
    *   The "Root" is the particle that satisfies the criteria whereas its parent does not.
    *   This ensures we target the primary neutral particle that initiated the shower/cluster.

## Summary of Final Targets
The final list comprises:
1.  **Tracked Particles**: All particles with tracks satisfying kinematic cuts (excluding those with tracked ancestors, though usually tracked particles form the "top" of the chain in this logic).
2.  **Neutral/Untracked Roots**: The roots of shower chains that enter the calorimeter but have no associated tracks.
