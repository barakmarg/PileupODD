# 3D interactive calo cluster plot (per event) colored by MeanShift cluster

import numpy as np
import plotly.express as px
from cluster.helpers.meanshift_mod import MeanShiftMod
from cluster.play import to_eta, convert_to_cartesian_eta_phi, get_points_for_clustering



def plot_calo_clusters_3d_given(event_idx, points_clustring_info_list:list, ms: MeanShiftMod, all_datasets, dataset_name="OpenDataDetector/ColliderML_ttbar_pu0", show=True):
    calo = all_datasets[dataset_name]["calo_hits"]["train"].with_format("numpy")
    ev = calo[event_idx]
    labels = ms.labels_
    data = points_clustring_info_list[event_idx]['points']
    mask_calo = points_clustring_info_list[event_idx]['mask_calo']
    e_mask = points_clustring_info_list[event_idx]['e_mask']
    e = np.asarray(ev["total_energy"], dtype=float)

    x = ev["x"][e_mask]
    y = ev["y"][e_mask]
    z = ev["z"][e_mask]
    # Selected per-hit metadata aligned with x,y,z
    eta =  data[:, 0][mask_calo]
    phi =  data[:, 1][mask_calo]
    e_sel = e[e_mask]
    labels = labels[mask_calo]

    # Robust marker sizing based on energy
    if e_sel.size:
        lo = np.percentile(e_sel, 5)
        hi = np.percentile(e_sel, 95)
        sizes = np.clip(e_sel, lo, hi)
    else:
        sizes = np.ones_like(e_sel, dtype=float)

    fig = px.scatter_3d(
        x=x, y=y, z=z,
        color=labels.astype(str),
        size=sizes,
        size_max=14,
        opacity=1.0,
        hover_data={
            "cluster": labels,
            "energy": e_sel,
            "eta": eta,
            "phi": phi
        },
        title=f"Calo hits MeanShift clusters (event {event_idx}, n_clusters={len(np.unique(labels))})"
    )
    fig.update_layout(scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z"
    ))
    if show:
        fig.show()
    return labels, ms.cluster_centers_

def plot_calo_clusters_3d(all_datasets, dataset_name="OpenDataDetector/ColliderML_ttbar_pu0", event_idx=0, bandwidth=0.4, energy_threshold=0.0001, show=True):
    calo = all_datasets[dataset_name]["calo_hits"]["train"].with_format("numpy")
    points_list = get_points_for_clustering(all_datasets, dataset_name=dataset_name, energy_threshold=energy_threshold, until_index=event_idx+1)
    data = points_list[event_idx]['points']

    ms = MeanShiftMod(bandwidth=bandwidth, bin_seeding=True).fit(data)
    return plot_calo_clusters_3d_given(event_idx, points_list, ms, all_datasets, dataset_name=dataset_name, show=show)


def plot_2d_meanshift_results( points, m: MeanShiftMod, mask=None):
    # Inspect MeanShift results
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    labels_full = np.asarray(m.labels_)
    centers_full = np.asarray(m.cluster_centers_)

    if mask is not None:
        points_used = points[mask]
        labels_used = labels_full[mask]
    else:
        points_used = points
        labels_used = labels_full

    # Unique labels present after masking
    unique, counts = np.unique(labels_used, return_counts=True)

    # Guard: only keep labels that have a valid center index
    valid = (unique >= 0) & (unique < len(centers_full))
    unique_valid = unique[valid]
    counts_valid = counts[valid]
    centers_sel = centers_full[unique_valid]

    n_clusters = len(centers_full)
    print("n_clusters:", n_clusters)
    print("centers shape:", centers_full.shape)
    print("cluster counts (in masked view):", dict(zip(unique_valid, counts_valid)))

    # Summary table aligned with present labels
    summary = pd.DataFrame(centers_sel, columns=["eta_center", "phi_center"])
    summary["cluster_id"] = unique_valid
    summary["count"] = counts_valid
    print(summary.head())

    # Plot
    plt.figure(figsize=(6,5))
    sc = plt.scatter(points_used[:,0], points_used[:,1], c=labels_used, s=8, cmap="tab20", alpha=0.8)
    if n_clusters > 0:
        plt.scatter(centers_full[:,0], centers_full[:,1], c="k", s=80, marker="x")
    plt.xlabel("eta")
    plt.ylabel("phi")
    plt.title(f"MeanShift clusters (n={n_clusters})")
    plt.colorbar(sc, label="cluster id")
    plt.tight_layout()
    plt.show()

def plot_energy_ratio_vs_cutoff(all_datasets_loaded, dataset_name='OpenDataDetector/ColliderML_ttbar_pu0', N_events=100):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    
    dataset = all_datasets_loaded[dataset_name]
    calo_e = dataset['calo_hits']['train']
    total_energy_events = calo_e['total_energy'][:N_events]

    event_sum_E_N = np.array([np.sum(np.asarray(e, dtype=float)) for e in total_energy_events])

    cutoffs_mev = np.linspace(0, 5, 100)
    cutoffs_gev = cutoffs_mev * 0.001

    average_ratios = []
    avg_cells_left = []

    for cutoff in cutoffs_gev:
        ratios_for_current_cutoff = []
        cells_left_for_current_cutoff = []
        
        for i, event_energies_list in enumerate(total_energy_events):
            event_energies = np.asarray(event_energies_list, dtype=float)
            e_total = event_sum_E_N[i]
            
            if e_total > 0:
                energies_after_cutoff = event_energies[event_energies > cutoff]
                e_cutoff = np.sum(energies_after_cutoff)
                num_cells_left = len(energies_after_cutoff)
                ratios_for_current_cutoff.append(e_cutoff / e_total)
                cells_left_for_current_cutoff.append(num_cells_left)

        if ratios_for_current_cutoff:
            average_ratios.append(np.mean(ratios_for_current_cutoff))
            avg_cells_left.append(np.mean(cells_left_for_current_cutoff))
        else:
            average_ratios.append(0)
            avg_cells_left.append(0)

    # --- Create a Pandas DataFrame for Plotly ---
    # This is the standard way to pass data to Plotly Express
    df = pd.DataFrame({
        'Cutoff (MeV)': cutoffs_mev,
        'Avg Energy Ratio': average_ratios,
        'Avg Cells Remaining': avg_cells_left
    })

    # --- Create the Interactive Plot with a single Plotly Express command ---
    fig = px.scatter(
        df,
        x='Cutoff (MeV)',
        y='Avg Energy Ratio',
        color='Avg Cells Remaining',
        color_continuous_scale=px.colors.sequential.Viridis,
        # Here we define exactly what data appears on hover
        hover_data={
            'Cutoff (MeV)': ':.2f',          # Format to 2 decimal places
            'Avg Energy Ratio': ':.3f',      # Format to 3 decimal places
            'Avg Cells Remaining': ':.1f'   # Format to 1 decimal place
        },
        labels={
            "Cutoff (MeV)": "Cutoff Energy (MeV)",
            "Avg Energy Ratio": "Average E_cutoff / E_total",
            "Avg Cells Remaining": "Avg Cells Left"
        },
        title="Average Energy Ratio vs. Cutoff Energy (Interactive)"
    )

    # Optional: Add a line to connect the points and show the trend more clearly
    fig.add_scatter(x=df['Cutoff (MeV)'], y=df['Avg Energy Ratio'], mode='lines', line=dict(color='grey'), name='Trend')


    # Set the y-axis range and show the figure
    fig.update_layout(yaxis_range=[0, 1.1])
    fig.show()


def plot_cluster_cardinality_histogram(m: MeanShiftMod, mask=None, show=True, include_size_distribution=True,
                                       log_counts=False, title_prefix="MeanShift", ax=None):
    """Plot per-cluster cardinality (number of points in each cluster) and optionally
    the distribution of cluster sizes.

    Parameters
    ----------
    m : MeanShiftMod
        A fitted MeanShiftMod instance (must have labels_ and cluster_centers_).
    mask : array-like of bool, optional
        Boolean mask selecting a subset of points to consider when counting cluster
        membership. The mask length must match len(m.labels_). The cluster centers
        are NOT filtered – counts are derived only from the selected labels.
    show : bool
        If True, display the figure.
    include_size_distribution : bool
        If True, add a second subplot with the histogram of cluster sizes.
    log_counts : bool
        If True, use a log scale for the y-axis of the bar chart(s).
    title_prefix : str
        Text prefix for the plot title.
    ax : matplotlib Axes or sequence of Axes, optional
        If provided, draw into these axes instead of creating a new figure.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: cluster_id, count
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    labels_full = np.asarray(m.labels_)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != labels_full.shape[0]:
            raise ValueError(f"mask length {mask.shape[0]} != number of labels {labels_full.shape[0]}")
        labels_used = labels_full[mask]
    else:
        labels_used = labels_full

    # Compute counts per label (excluding potential noise label -1 separately if present)
    unique, counts = np.unique(labels_used, return_counts=True)
    order = np.argsort(unique)
    unique = unique[order]
    counts = counts[order]

    df = pd.DataFrame({
        'cluster_id': unique,
        'count': counts
    })

    n_clusters_total = len(np.unique(labels_full))
    n_clusters_present = len(unique)

    # Prepare axes
    if ax is None:
        if include_size_distribution:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            ax_bar, ax_hist = axes
        else:
            fig, ax_bar = plt.subplots(figsize=(7, 4))
            ax_hist = None
    else:
        if include_size_distribution:
            try:
                ax_bar, ax_hist = ax
            except Exception:
                raise ValueError("When include_size_distribution=True you must pass a sequence of two Axes.")
        else:
            ax_bar = ax
            ax_hist = None

    # Bar plot: cluster id vs count
    ax_bar.bar(df['cluster_id'].astype(str), df['count'], color="#1f77b4", alpha=0.85, edgecolor='black')
    ax_bar.set_xlabel("Cluster ID")
    ax_bar.set_ylabel("Points in cluster")
    bar_title = f"{title_prefix} cluster sizes (present={n_clusters_present}, total={n_clusters_total})"
    if mask is not None:
        bar_title += " (masked)"
    ax_bar.set_title(bar_title)
    if log_counts:
        ax_bar.set_yscale('log')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.4)

    # Optional histogram of the distribution of cluster sizes
    if include_size_distribution and ax_hist is not None:
        ax_hist.hist(df['count'], bins='auto', color="#ff7f0e", alpha=0.8, edgecolor='black')
        ax_hist.set_xlabel("Cluster size (cardinality)")
        ax_hist.set_ylabel("#Clusters")
        ax_hist.set_title("Distribution of cluster sizes")
        if log_counts:
            ax_hist.set_yscale('log')
        ax_hist.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    if show:
        plt.show()

    return df


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_cluster_energy_purity(ms_model: MeanShiftMod, event_data, show=True, mask_labels=None):
    """
    Calculates and plots a histogram of cluster energy purity for a single event.

    This function is designed to work with the specific 'calo_hits' data schema.

    The purity for each cluster is defined as the ratio:
    Ratio = E_total_cluster / E_total_truth_particles

    Parameters
    ----------
    ms_model : object
        A fitted clustering model (e.g., MeanShiftMod) that has a `.labels_` attribute.
        `labels_` is an array where the value at index `i` is the cluster ID for cell `i`.
        Noise points are expected to have a label of -1 and will be ignored.

    event_data : dict-like
        A single event record from the calo_hits dataset. It must contain the keys
        as described in the problem context:
        - 'total_energy': list of total energy per cell.
        - 'contrib_particle_ids': list of lists of particle IDs for each cell.
        - 'contrib_energies': list of lists of energies for each contribution.
        The length of these top-level lists must match len(ms_model.labels_).

    show : bool
        If True, display the matplotlib plot.

    Returns
    -------
    list
        A list containing the calculated purity ratio for each non-noise cluster.
    """
    cluster_labels = ms_model.labels_
    num_cells = len(event_data['total_energy'])
    if mask_labels is not None:
        cluster_labels = np.array(cluster_labels)
        cluster_labels = cluster_labels[mask_labels]
    if len(cluster_labels) != num_cells:
        raise ValueError(
            f"Mismatch in data size. Model has {len(cluster_labels)} labels, "
            f"but event data has {num_cells} cells."
        )

    # STEP 1: Build the ground truth map for the entire event.
    # Map each particle ID to its total energy deposited across all cells.
    # This corresponds to your 'x' dictionary.
    particle_truth_energies = defaultdict(float)
    for i in range(num_cells):
        for pid, energy in zip(event_data['contrib_particle_ids'][i], event_data['contrib_energies'][i]):
            particle_truth_energies[pid] += energy

    # STEP 2: For each cluster, sum its total cell energy and collect all contributing particle IDs.
    # These two dicts correspond to your 'y' dictionary concept.
    cluster_summed_energies = defaultdict(float)
    cluster_contributing_particles = defaultdict(set)

    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id == -1:  # Ignore noise points
            continue
        
        # Add the cell's energy to its cluster's total
        cluster_summed_energies[cluster_id] += event_data['total_energy'][i]
        
        # Add all contributing particle IDs from this cell to a set for the cluster
        # Using a set automatically handles uniqueness.
        cluster_contributing_particles[cluster_id].update(event_data['contrib_particle_ids'][i])

    # STEP 3: Calculate the purity ratio for each cluster.
    purity_ratios = []
    for cluster_id, e_cluster in cluster_summed_energies.items():
        # Get the unique particles that contributed to this cluster
        pids_in_cluster = cluster_contributing_particles[cluster_id]
        
        # Calculate the denominator: sum the total truth energy for each of those particles
        e_truth_denominator = sum(particle_truth_energies[pid] for pid in pids_in_cluster)
        
        if e_cluster == 0:
            purity_ratios.append(0.0)
        else:
            ratio = e_cluster / e_truth_denominator
            purity_ratios.append(ratio)

    if not purity_ratios:
        print("Warning: No valid clusters found to calculate ratios.")
        return []

    # STEP 4: Plot the results in a histogram.
    plt.figure(figsize=(10, 6))
    plt.hist(purity_ratios, bins=50, range=(0, 1.2), alpha=0.8, edgecolor='black', color='darkcyan')
    
    plt.xlabel("Cluster Energy Purity (E_cluster / E_truth_particles)")
    plt.ylabel("Number of Clusters")
    plt.title("Histogram of Cluster Energy Purity")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a vertical line at 1.0 for a perfect purity reference
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect Purity (Ratio = 1.0)')
    plt.legend()
    
    if show:
        plt.tight_layout()
        plt.show()
        
    return purity_ratios
