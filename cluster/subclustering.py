from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import importlib

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from cluster.helpers.meanshift_mod import MeanShiftMod


def _slice_cluster_payload(
    points: np.ndarray,
    mask_calo: np.ndarray,
    high_mask: np.ndarray,
    energies: np.ndarray,
    member_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extract cluster-specific arrays for downstream inspection."""
    return {
        'points': points[member_indices].copy(),
        'mask_calo': mask_calo[member_indices].copy(),
        'high_mask': high_mask[member_indices].copy(),
        'energies': energies[member_indices].copy(),
    }


def _build_ms_model(points_entry: Dict[str, np.ndarray], bandwidth: float) -> "MeanShiftMod":
    """Create a MeanShiftMod instance seeded with the energetic points."""

    from cluster.helpers.meanshift_mod import MeanShiftMod

    seeds = points_entry['points'][points_entry['high_mask']]
    if seeds.size == 0:
        seeds = None
    return MeanShiftMod(bin_seeding=True, bandwidth=bandwidth, seeds=seeds)




def run_hierarchical_meanshift(
    all_datasets: Any,
    dataset_name: str = "OpenDataDetector/ColliderML_ttbar_pu0",
    *,
    until_index: int = 10,
    energy_threshold: float = 0.000005,
    high_energy_threshold_perc: float = 0.5,
    primary_bandwidth: float = 0.2,
    secondary_bandwidth: float = 0.05,
    large_cluster_threshold: int = 2000,
    do_weights: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    """Run two-level MeanShift clustering and return flattened clusters plus metadata."""

    import cluster.play as play

    play = importlib.reload(play)
    from cluster.play import get_points_for_clustering, parrallel_fit

    points_list = get_points_for_clustering(
        all_datasets,
        dataset_name=dataset_name,
        energy_threshold=energy_threshold,
        high_energy_threshold_perc=high_energy_threshold_perc,
        until_index=until_index,
    )
    print(f"Prepared points for {len(points_list)} events.")

    ms_models = [_build_ms_model(p, primary_bandwidth) for p in points_list]
    event_models = parrallel_fit(ms_models, points_list, do_weights=do_weights)

    clusters_payload: List[Dict[str, np.ndarray]] = []
    cluster_metadata: List[Dict[str, Any]] = []
    
    for event_index, (model, points_entry) in enumerate(zip(event_models, points_list)):
        # group by labels, see just labels for calo hits
        labels = model._labels
        for cluster_label in np.unique(labels):
            # member indices without tracks
            cluster_member_indices = np.where(labels[points_entry['mask_calo']] == cluster_label)[0]
    
            if len(cluster_member_indices) > large_cluster_threshold:
                # Split large clusters with secondary MeanShift
                # first calculate percentile threshold mask for high energy points
                from cluster.play import calc_percentile_threshold_mask
                energies_in_cluster = points_entry['energies'][cluster_member_indices]
                top_high_mask = calc_percentile_threshold_mask(energies_in_cluster, percentile=4.0)
                secondary_seeds = points_entry['points'][points_entry['mask_calo']][cluster_member_indices][top_high_mask]
                ms_cluster = MeanShiftMod(
                    bin_seeding=True,
                    bandwidth=secondary_bandwidth,
                    seeds=secondary_seeds if secondary_seeds.size > 0 else None,
                )
                member_indices = np.where(labels == cluster_label)[0]
                ms_cluster.fit(points_entry['points'][member_indices])

                payload ={
                    'event_index': event_index,
                    'cluster_indices': member_indices,
                    'calo_only_indices': cluster_member_indices,
                    'ms': ms_cluster,
                    'energies_in_cluster': energies_in_cluster
                }
                for 
                
            

        clusters_payload.extend(payloads)
        cluster_metadata.extend(info_records)

    return clusters_payload, cluster_metadata


def sub_clustering(datasets):
    m_n_list, cluster_info = run_hierarchical_meanshift(datasets)
    print(f"Total clusters materialized: {len(m_n_list)}")
    print(f"Metadata entries: {len(cluster_info)}")
    return m_n_list, cluster_info