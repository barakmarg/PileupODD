import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from primary.clue import run_clue_hybrid
import torch
from torch.nn.utils.rnn import pad_sequence

from primary.calibration import CALIBRATION
# --------------------------------------------------------------------------------
# 1. SETUP & DEFINITIONS
# --------------------------------------------------------------------------------

# Define System Groups based on your CALIBRATION dataframe
ecal_ids = [9, 10, 11]  # Ecal Endcap Neg, Barrel, Endcap Pos
hcal_ids = [12, 13, 14] # Hcal Endcap Neg, Barrel, Endcap Pos

# Hyper-parameters per system
# NOTE: rhoc is the most sensitive parameter. 
#       If you get too many clusters, INCREASE rhoc.
#       If you lose signal, DECREASE rhoc.


def to_padded_torch(df: pl.DataFrame, pl_col_name:str):
    points_list = []
    features_list = []
    data_processed = df.select('event_id', pl_col_name)
    # Iterate and convert to Tensors
    for row in data_processed.iter_rows(named=True):
        # Shape: (N_hits, 3)
        p = np.column_stack([row[pl_col_name]])
        points_list.append(torch.tensor(p, dtype=torch.float32))


    # Pad Sequences to (Batch, Max_N, ...)
    # Use a far-away value for padding coords so they don't cluster with real data
    return pad_sequence(points_list, batch_first=True, padding_value=-1) 
# --------------------------------------------------------------------------------
# 2. HELPER FUNCTION
# --------------------------------------------------------------------------------
def run_clue_subset(full_points, full_energy, subset_mask, params, id_offset):
    """
    Runs CLUE on a subset of the data defined by the mask.
    Returns:
        - subset_ids: Global indices of the subset
        - cluster_labels: Labels for these points (offset by id_offset)
        - num_clusters: Count of new clusters found
    """
    # 1. Filter Data
    subset_idx = torch.where(subset_mask)
    
    if len(subset_idx) == 0:
        return subset_idx, np.array([]), 0

    points_sub = full_points[subset_idx]
    energy_sub = full_energy[subset_idx]

    # 2. Run CLUE (Your existing function)
    _, _, is_seed, local_labels = run_clue_hybrid(
        points_sub, 
        energy_sub, 
        dc=params['dc'], 
        rhoc=params['rhoc'], 
        dm=params['dm'], 
        max_num_neighbors=params['max_neighbors']
    )

    # 3. Apply Offset to valid clusters (keep noise as -1)
    # local_labels is numpy array from your function
    final_labels = np.where(local_labels != -1, local_labels + id_offset, -1)
    
    num_new_clusters = np.sum(is_seed.cpu().numpy())
    
    return subset_idx.cpu().numpy(), final_labels, num_new_clusters

def execute(calo, params_ecal , params_hcal, calo_points, calo_features):
    # --------------------------------------------------------------------------------
    # 3. EXECUTION
    # --------------------------------------------------------------------------------

    # Convert detector column to tensor for fast masking
    detector_ids =to_padded_torch(calo, 'detector')
    seed_coords = calo_points.cpu().numpy()

    # Initialize Output Array (-1 = Noise)
    global_cluster_ids = np.full(len(calo_points), -1, dtype=np.int32)
    current_cluster_count = 0

    # --- A. Run ECAL ---
    mask_ecal = torch.isin(detector_ids, torch.tensor(ecal_ids, device=calo_points.device))
    idx_ecal, lbl_ecal, n_ecal = run_clue_subset(
        calo_points, calo_features, mask_ecal, params_ecal, current_cluster_count
    )
    global_cluster_ids[idx_ecal] = lbl_ecal
    current_cluster_count += n_ecal

    # --- B. Run HCAL ---
    mask_hcal = torch.isin(detector_ids, torch.tensor(hcal_ids, device=calo_points.device))
    idx_hcal, lbl_hcal, n_hcal = run_clue_subset(
        calo_points, calo_features, mask_hcal, params_hcal, current_cluster_count
    )
    global_cluster_ids[idx_hcal] = lbl_hcal
    current_cluster_count += n_hcal

    # --------------------------------------------------------------------------------
    # 4. ASSIGNMENT & CENTERS (Standard Logic)
    # --------------------------------------------------------------------------------
    # Add IDs to DataFrame
    # Compute cluster center coordinates per hit

    # 2. Vectorized Energy-Weighted Center Calculation (Polars)
    # We explode the calo dataframe to hit-level to perform the math
    # Note: Using calo_features (GPU tensor) for weights to ensure calibration is included
    hit_level_df = calo.select(['event_id', 'x', 'y', 'z']).explode(['x', 'y', 'z'])
    hit_level_df = hit_level_df.with_columns([
        pl.Series(name='cluster_id', values=global_cluster_ids),
        pl.Series(name='energy', values=calo_features.cpu().numpy())
    ])

    # Calculate Weighted Means: Sum(Coord * Energy) / Sum(Energy)
    centroids = (
        hit_level_df.lazy()
        .filter(pl.col('cluster_id') != -1)
        .group_by('cluster_id')
        .agg([
            ((pl.col('x') * pl.col('energy')).sum() / pl.col('energy').sum()).alias('cx'),
            ((pl.col('y') * pl.col('energy')).sum() / pl.col('energy').sum()).alias('cy'),
            ((pl.col('z') * pl.col('energy')).sum() / pl.col('energy').sum()).alias('cz'),
        ])
        .collect()
    )

    # Join centroids back to hit level
    hit_level_df = hit_level_df.join(centroids, on='cluster_id', how='left')

    # Fill noise (-1) with 0 or original hit coordinates as per your requirement
    hit_level_df = hit_level_df.with_columns([
        pl.col('cx').fill_null(0.0),
        pl.col('cy').fill_null(0.0),
        pl.col('cz').fill_null(0.0),
    ])

    # 3. Aggregate back to Event-level (List columns)
    # This matches your original input structure
    calo_out = (
        hit_level_df.group_by('event_id')
        .agg([
            pl.col('cx').alias('cluster_cx'),
            pl.col('cy').alias('cluster_cy'),
            pl.col('cz').alias('cluster_cz'),
            pl.col('cluster_id')
        ])
    )
    
    # Merge the new columns back to the original calo (to keep other columns like detector/particle_ids)
    calo = calo.drop(['cluster_cx', 'cluster_cy', 'cluster_cz', 'cluster_id'], strict=False).join(calo_out, on='event_id')

    return calo, current_cluster_count

def metric_eval(calo, event_id, particles, calo_deps_mappings):
    result = (calo.lazy().select('contrib_particle_ids', 'contrib_energies', 'event_id', 'cluster_id', 'detector')
    .explode('contrib_particle_ids', 'contrib_energies', 'cluster_id', 'detector')
    .explode('contrib_particle_ids', 'contrib_energies')
    .rename({'contrib_particle_ids':'particle_id', 'contrib_energies':'energy_contribution'})
    # Calibrate energy contributions
    .join(CALIBRATION.lazy(), on='detector')
    .with_columns((pl.col('energy_contribution') * pl.col('calib_factor')).alias('energy_contribution'))
    .join(
        particles.lazy()
        .filter(pl.col('event_id')==event_id)
        .select('event_id', 'particle_id','vertex_primary')
        .explode('particle_id','vertex_primary'), on=['event_id','particle_id'], how='inner'

    )
    .join(calo_deps_mappings.lazy(),
            left_on=['event_id','particle_id'], right_on=['event_id','src_particle_id'], how='inner'
        )
    .filter(pl.col('target_particle_id').is_not_null())
    #.filter(pl.col('vertex_primary')==1)
    .with_columns(pl.when(pl.col('vertex_primary') == 1).then(1).otherwise(0).alias('hard_scatter'))
    # two classes: cluster_id == -1 (noise), cluster_id != -1 (signal)
    .with_columns(pl.when(pl.col('cluster_id') == -1).then(0).otherwise(1).alias('is_signal'))
    .group_by('is_signal', 'event_id', 'hard_scatter')
    .agg(pl.col('energy_contribution').sum().alias('primary_energy_contribution'))
    # do ratio

    ).collect()
    signal = result.filter(pl.col('is_signal')==1)['primary_energy_contribution'][0]
    noise = result.filter(pl.col('is_signal')!=1)['primary_energy_contribution'][0]
    return noise/signal # should be <0.01