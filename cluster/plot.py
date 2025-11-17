# 3D interactive calo cluster plot (per event) colored by MeanShift cluster

import numpy as np
import plotly.express as px
from cluster.helpers.meanshift_mod import MeanShiftMod
from cluster.play import to_eta, convert_to_cartesian_eta_phi, get_points_for_clustering


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Any
import numpy as np
import pandas as pd
import plotly.express as px
from collections import defaultdict

# The MeanShiftMod class is not strictly needed for the function signature
# if we just access the .labels_ attribute, so we can use a generic type hint.
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from collections import defaultdict
from typing import Any

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

    # Use GeV directly (equivalent to 0–5 MeV = 0–0.005 GeV)
    cutoffs_gev = np.linspace(0, 0.005, 100)

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
        'Cutoff (GeV)': cutoffs_gev,
        'Avg Energy Ratio': average_ratios,
        'Avg Cells Remaining': avg_cells_left
    })

    # --- Create the Interactive Plot with a single Plotly Express command ---
    fig = px.scatter(
        df,
        x='Cutoff (GeV)',
        y='Avg Energy Ratio',
        color='Avg Cells Remaining',
        color_continuous_scale=px.colors.sequential.Viridis,
        # Here we define exactly what data appears on hover
        hover_data={
            'Cutoff (GeV)': ':.6f',          # GeV values in small range; show more precision
            'Avg Energy Ratio': ':.3f',      # Format to 3 decimal places
            'Avg Cells Remaining': ':.1f'   # Format to 1 decimal place
        },
        labels={
            "Cutoff (GeV)": "Cutoff Energy (GeV)",
            "Avg Energy Ratio": "Average E_cutoff / E_total",
            "Avg Cells Remaining": "Avg Cells Left"
        },
        title="Average Energy Ratio vs. Cutoff Energy (Interactive)"
    )

    # Optional: Add a line to connect the points and show the trend more clearly
    fig.add_scatter(x=df['Cutoff (GeV)'], y=df['Avg Energy Ratio'], mode='lines', line=dict(color='grey'), name='Trend')


    # Set the y-axis range and show the figure
    fig.update_layout(yaxis_range=[0, 1.1])
    fig.show()


# ...existing code...
def plot_cluster_cardinality_histogram(
    m,
    mask=None,
    show=True,
    include_size_distribution=True,
    log_counts=False,
    title_prefix="MeanShift",
    ax=None,
    keep_noise=False,
):
    """
    Plot per-cluster cardinality (number of points in each cluster). Supports a single
    MeanShiftMod instance OR a list of them (multiple events). When multiple models
    are provided, cluster IDs are local to each event; the plot aggregates them by
    assigning a global sequential index.

    Parameters
    ----------
    m : MeanShiftMod | Sequence[MeanShiftMod]
        Fitted instance or list of fitted instances (each must have labels_).
    mask : array-like[bool] | Sequence[array-like[bool]], optional
        Boolean mask for points in a single model OR list of masks (same length as m)
        for multiple models. If provided, lengths must match corresponding labels.
    show : bool
        If True, display the figure.
    include_size_distribution : bool
        If True, add a second subplot with histogram of cluster sizes.
    log_counts : bool
        If True, use log scale on y-axis.
    title_prefix : str
        Prefix for plot title.
    ax : matplotlib Axes or sequence of Axes, optional
        Axes to draw into. For multiple panels pass (ax_bar, ax_hist).
    keep_noise : bool
        If False, exclude label == -1 clusters (often noise) from counts.

    Returns
    -------
    pandas.DataFrame
        For single model:
            columns: cluster_id, count
        For multiple models:
            columns: global_cluster_idx, event_index, original_cluster_id, count
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections.abc import Sequence

    def _process_single(labels_full, mask_local):
        labels_full = np.asarray(labels_full)
        if mask_local is not None:
            mask_arr = np.asarray(mask_local, dtype=bool)
            if mask_arr.shape[0] != labels_full.shape[0]:
                raise ValueError(f"Mask length {mask_arr.shape[0]} != labels length {labels_full.shape[0]}")
            labels_used = labels_full[mask_arr]
        else:
            labels_used = labels_full

        # Optionally drop noise label
        if not keep_noise:
            labels_used = labels_used[labels_used != -1]

        unique, counts = np.unique(labels_used, return_counts=True)
        order = np.argsort(unique)
        unique = unique[order]
        counts = counts[order]

        return pd.DataFrame({'cluster_id': unique, 'count': counts})

    is_multi = isinstance(m, Sequence) and not isinstance(m, (str, bytes))

    if not is_multi:
        # Single model path
        df = _process_single(m.labels_, mask)
        n_clusters_total = len(np.unique(m.labels_ if keep_noise else np.asarray(m.labels_)[np.asarray(m.labels_) != -1]))
        n_clusters_present = len(df)
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
                ax_bar, ax_hist = ax
            else:
                ax_bar = ax
                ax_hist = None

        # Bar of cluster sizes
        ax_bar.bar(df['cluster_id'].astype(str), df['count'], color="#1f77b4", alpha=0.85, edgecolor='black')
        ax_bar.set_xlabel("Cluster ID")
        ax_bar.set_ylabel("Points in cluster")
        title = f"{title_prefix} cluster sizes (present={n_clusters_present}, total={n_clusters_total})"
        if mask is not None:
            title += " (masked)"
        ax_bar.set_title(title)
        if log_counts:
            ax_bar.set_yscale('log')
        ax_bar.grid(axis='y', linestyle='--', alpha=0.4)

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

    # Multi-model path
    models = list(m)
    n_events = len(models)

    # Normalize mask list
    if mask is not None:
        if not isinstance(mask, Sequence):
            raise ValueError("For multiple models, mask must be a sequence (list) of masks.")
        if len(mask) != n_events:
            raise ValueError("Number of masks must equal number of models.")
        mask_list = list(mask)
    else:
        mask_list = [None] * n_events

    per_event_frames = []
    for idx, (model, mask_local) in enumerate(zip(models, mask_list)):
        df_event = _process_single(model.labels_, mask_local)
        if df_event.empty:
            continue
        df_event['event_index'] = idx
        df_event['original_cluster_id'] = df_event['cluster_id']
        per_event_frames.append(df_event)

    if not per_event_frames:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['global_cluster_idx', 'event_index', 'original_cluster_id', 'count'])

    # Concatenate and assign global cluster indices
    concat_df = pd.concat(per_event_frames, ignore_index=True)
    concat_df['global_cluster_idx'] = np.arange(len(concat_df))

    # Prepare aggregated plot
    if ax is None:
        if include_size_distribution:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            ax_bar, ax_hist = axes
        else:
            fig, ax_bar = plt.subplots(figsize=(9, 4))
            ax_hist = None
    else:
        if include_size_distribution:
            ax_bar, ax_hist = ax
        else:
            ax_bar = ax
            ax_hist = None

    # Bar plot (global cluster index vs size)
    ax_bar.bar(concat_df['global_cluster_idx'].astype(str), concat_df['count'],
               color="#1f77b4", alpha=0.85, edgecolor='black')
    ax_bar.set_xlabel("Global Cluster Index")
    ax_bar.set_ylabel("Points in cluster")
    ax_bar.set_title(f"{title_prefix} aggregated cluster sizes (events={n_events}, clusters={len(concat_df)})")
    if log_counts:
        ax_bar.set_yscale('log')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.4)

    # Histogram of cluster size distribution across all events
    if include_size_distribution and ax_hist is not None:
        ax_hist.hist(concat_df['count'], bins='auto', color="#ff7f0e", alpha=0.8, edgecolor='black')
        ax_hist.set_xlabel("Cluster size (cardinality)")
        ax_hist.set_ylabel("#Clusters")
        ax_hist.set_title("Aggregated distribution of cluster sizes")
        if log_counts:
            ax_hist.set_yscale('log')
        ax_hist.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    if show:
        plt.show()

    return concat_df[['global_cluster_idx', 'event_index', 'original_cluster_id', 'count']]
# ...existing code...


def plot_aggregated_particle_completeness(
    ms_models: List[Any],
    full_dataset: Any,
    event_indices: List[int],
    show: bool = True,
    mask_list: List[np.ndarray] = None
):
    """
    Calculates and plots an aggregated histogram of particle energy completeness over multiple events.

    This function iterates through a list of specified event indices, calculates the
    completeness for each particle in each event, and combines them into a
    single summary plot.

    Parameters
    ----------
    ms_models : List[object]
        A list of fitted clustering models. Must be the same length as `event_indices`.
    full_dataset : dataset object or list of dicts
        The full dataset (e.g., calo_hits) that can be indexed to retrieve event data.
    event_indices : List[int]
        A list of integer indices for the events to be processed.
    show : bool
        If True, display the final aggregated matplotlib plot.
    mask_list : List[np.ndarray], optional
        A list of boolean masks. If provided, `mask_list[i]` is applied to the
        cells of event `event_indices[i]` before processing.

    Returns
    -------
    list
        A single flat list containing the calculated completeness ratios from all
        particles across all specified events.
    """
    if len(ms_models) != len(event_indices):
        raise ValueError("The number of models must match the number of event indices.")
    if mask_list is not None and len(mask_list) != len(event_indices):
        raise ValueError("The number of masks must match the number of event indices.")

    all_completeness_ratios = []

    print(f"Processing particle completeness for {len(event_indices)} events...")
    # Loop through each specified event index and its corresponding model
    for i, event_idx in enumerate(event_indices):
        ms_model = ms_models[i]
        event_data = full_dataset[event_idx]
        
        cluster_labels = ms_model.labels_
        
        # Apply mask if provided
        if mask_list is not None:
            mask = mask_list[i]
            cluster_labels = np.array(cluster_labels)
            cluster_labels = cluster_labels[mask]
        
        num_cells = len(event_data['total_energy'])
        if len(cluster_labels) != num_cells:
            print(f"Warning: Skipping event index {event_idx} due to data mismatch "
                  f"({len(cluster_labels)} labels vs {num_cells} cells).")
            continue

        # --- Logic for a single event, from the previous function ---
        particle_total_energy = defaultdict(float)
        particle_to_cluster_energies = defaultdict(lambda: defaultdict(float))

        for cell_idx in range(num_cells):
            cluster_id = cluster_labels[cell_idx]
            for pid, energy in zip(event_data['contrib_particle_ids'][cell_idx], event_data['contrib_energies'][cell_idx]):
                particle_total_energy[pid] += energy
                if cluster_id != -1:
                    particle_to_cluster_energies[pid][cluster_id] += energy

        completeness_ratios_for_event = []
        for pid, total_energy in particle_total_energy.items():
            if total_energy == 0: continue
            
            cluster_energies = particle_to_cluster_energies.get(pid)
            largest_dep_in_cluster = max(cluster_energies.values()) if cluster_energies else 0.0
            
            ratio = largest_dep_in_cluster / total_energy
            completeness_ratios_for_event.append(ratio)
        
        # Add the results from this event to the master list
        all_completeness_ratios.extend(completeness_ratios_for_event)

    if not all_completeness_ratios:
        print("Warning: No valid particles were found across any of the specified events.")
        return []
    
    print(f"Calculation complete. Found {len(all_completeness_ratios)} particles in total.")

    # --- Plot the aggregated results ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_completeness_ratios, bins=50, range=(0, 1.2), alpha=0.8, edgecolor='black', color='orangered')
    
    plt.xlabel("Particle Energy Completeness (Largest Cluster Dep / Total Particle Energy)")
    plt.ylabel("Total Number of Particles (from all events)")
    plt.title(f"Aggregated Histogram of Particle Completeness for {len(event_indices)} Events")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.yscale('log')
    
    plt.axvline(1.0, color='blue', linestyle='--', linewidth=2, label='Perfect Completeness (Ratio = 1.0)')
    plt.legend()
    
    if show:
        plt.tight_layout()
        plt.show()
        
    return all_completeness_ratios



def plot_cluster_energy_purity_interactive(ms_model: Any, event_data: dict, show: bool = True, mask_labels=None):
    """
    Calculates and plots an INTERACTIVE histogram of cluster energy purity using Plotly.

    This version corrects the hover template to properly display cluster details.

    Parameters
    ----------
    ms_model : object
        A fitted clustering model (e.g., MeanShiftMod) with a `.labels_` attribute.
    event_data : dict
        A single event record from the calo_hits dataset with keys like 'total_energy', etc.
    show : bool
        If True, display the interactive plot.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the detailed purity information for each cluster.
    """
    cluster_labels = ms_model.labels_
    num_cells = len(event_data['total_energy'])
    if mask_labels is not None:
        cluster_labels = np.array(cluster_labels)
        cluster_labels = cluster_labels[mask_labels]
    if len(cluster_labels) != num_cells:
        raise ValueError(f"Data size mismatch: Model has {len(cluster_labels)} labels, event has {num_cells} cells.")

    # STEP 1: Build the ground truth map for particle energies
    particle_truth_energies = defaultdict(float)
    for i in range(num_cells):
        for pid, energy in zip(event_data['contrib_particle_ids'][i], event_data['contrib_energies'][i]):
            particle_truth_energies[pid] += energy

    # STEP 2: Calculate cluster energies and contributing particles
    cluster_summed_energies = defaultdict(float)
    cluster_contributing_particles = defaultdict(set)
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id == -1: continue
        cluster_summed_energies[cluster_id] += event_data['total_energy'][i]
        cluster_contributing_particles[cluster_id].update(event_data['contrib_particle_ids'][i])

    # STEP 3: Create a detailed DataFrame with all cluster information
    cluster_data = []
    for cluster_id, e_cluster in cluster_summed_energies.items():
        pids_in_cluster = cluster_contributing_particles[cluster_id]
        e_truth = sum(particle_truth_energies[pid] for pid in pids_in_cluster)
        ratio = e_cluster / e_truth if e_truth > 0 else 0
        cluster_data.append({
            'cluster_id': cluster_id,
            'purity_ratio': ratio,
            'E_total_cluster': e_cluster,
            'E_total_truth_particles': e_truth
        })

    if not cluster_data:
        print("Warning: No valid clusters found to create a plot.")
        return pd.DataFrame()

    df = pd.DataFrame(cluster_data)

    # STEP 4: Manually bin the data and prepare for plotting
    bin_count = 50
    bin_range = (0, 1.5)
    bins = np.linspace(bin_range[0], bin_range[1], bin_count + 1)
    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(bin_count)]
    df['bin'] = pd.cut(df['purity_ratio'], bins=bins, labels=bin_labels, right=False, include_lowest=True)

    # Aggregate hover text for each bin
    def aggregate_hover_text(group):
        texts = []
        for _, row in group.head(15).iterrows(): # Limit to first 15 to keep tooltips manageable
            texts.append(
                f"ID: {row['cluster_id']} | E_cluster: {row['E_total_cluster']:.4f} | E_truth: {row['E_total_truth_particles']:.4f}"
            )
        if len(group) > 15:
            texts.append(f"<br>... and {len(group) - 15} more clusters")
        return "<br>".join(texts)

    plot_df = df.dropna(subset=['bin']).groupby('bin', as_index=False).agg(
        count=('cluster_id', 'count'),
        hover_text=('cluster_id', lambda x: aggregate_hover_text(df.loc[x.index]))
    )

    # STEP 5: Create the interactive plot with the CORRECTED hover template
    fig = px.bar(
        plot_df,
        x='bin',
        y='count',
        custom_data=['hover_text'], # Pass the aggregated text
        labels={'count': 'Number of Clusters', 'bin': 'Purity Ratio (E_cluster / E_truth)'}
    )

    # --- THIS IS THE FIX ---
    # The correct syntax is %{customdata[0]} to access the first element of custom_data.
    fig.update_traces(
        hovertemplate="<b>Bin Range:</b> %{x}<br>" +
                      "<b>Cluster Count:</b> %{y}<br><br>" +
                      "<b>--- Clusters in this Bin ---</b><br>" +
                      "%{customdata[0]}<extra></extra>" # <extra></extra> hides the secondary box
    )
    
    fig.update_layout(
        title_text="<b>Interactive Histogram of Cluster Energy Purity</b>",
        yaxis_type="log",
        xaxis={'tickangle': -45, 'type': 'category'}, # Explicitly set as category axis
        hoverlabel=dict(bgcolor="white", font_size=12)
    )

    # Find the bin index that contains 1.0 to draw the line correctly on a category axis
    bin_width = (bin_range[1] - bin_range[0]) / bin_count
    line_index = int(np.floor(1.0 / bin_width))
    
    fig.add_vline(x=line_index, line_width=2, line_dash="dash", line_color="red",
                  annotation_text="Perfect Purity", annotation_position="top right")

    if show:
        fig.show()
        
    return df

from typing import List, Any
def plot_aggregated_cluster_purity(
    ms_models: List[Any], 
    full_dataset: Any, 
    event_indices: List[int], 
    show: bool = True,
    mask_list: List[np.ndarray] = None
):
    """
    Calculates and plots an aggregated histogram of cluster energy purity over multiple events.

    This function iterates through a list of specified event indices, calculates the
    purity for each cluster in each of those events, and combines them into a
    single summary plot.

    Parameters
    ----------
    ms_models : List[object]
        A list of fitted clustering models (e.g., MeanShiftMod). The length of this list
        must match the length of `event_indices`, where `ms_models[i]` corresponds
        to the clustering result for the event at `event_indices[i]`.

    full_dataset : dataset object or list of dicts
        The full dataset (e.g., calo_hits) that can be indexed to retrieve a single
        event's data, like `full_dataset[event_idx]`.

    event_indices : List[int]
        A list of integer indices for the events to be processed from `full_dataset`.

    show : bool
        If True, display the final aggregated matplotlib plot.

    Returns
    -------
    list
        A single flat list containing the calculated purity ratios from all clusters
        across all specified events.
    """
    if len(ms_models) != len(event_indices):
        raise ValueError("The number of models must match the number of event indices.")

    all_purity_ratios = []

    print(f"Processing {len(event_indices)} events...")
    # Loop through each specified event index and its corresponding model
    for i, event_idx in enumerate(event_indices):
        ms_model = ms_models[i]
        event_data = full_dataset[event_idx]
        
        # --- The following logic is for a single event, adapted from your function ---
        cluster_labels = ms_model.labels_
        num_cells = len(event_data['total_energy'])
        if mask_list is not None:
            mask_labels = mask_list[i]
            cluster_labels = np.array(cluster_labels)
            cluster_labels = cluster_labels[mask_labels]
        if len(cluster_labels) != num_cells:
            print(f"Warning: Skipping event index {event_idx} due to data mismatch "
                  f"({len(cluster_labels)} labels vs {num_cells} cells).")
            continue

        particle_truth_energies = defaultdict(float)
        for cell_idx in range(num_cells):
            for pid, energy in zip(event_data['contrib_particle_ids'][cell_idx], event_data['contrib_energies'][cell_idx]):
                particle_truth_energies[pid] += energy

        cluster_summed_energies = defaultdict(float)
        cluster_contributing_particles = defaultdict(set)
        for cell_idx, cluster_id in enumerate(cluster_labels):
            if cluster_id == -1:
                continue
            cluster_summed_energies[cluster_id] += event_data['total_energy'][cell_idx]
            cluster_contributing_particles[cluster_id].update(event_data['contrib_particle_ids'][cell_idx])

        purity_ratios_for_event = []
        for cluster_id, e_cluster in cluster_summed_energies.items():
            pids_in_cluster = cluster_contributing_particles[cluster_id]
            e_truth_denominator = sum(particle_truth_energies[pid] for pid in pids_in_cluster)
            
            ratio = e_cluster / e_truth_denominator if e_truth_denominator > 0 else 0.0
            purity_ratios_for_event.append(ratio)
        
        # Add the results from this event to the master list
        all_purity_ratios.extend(purity_ratios_for_event)

    if not all_purity_ratios:
        print("Warning: No valid clusters were found across any of the specified events.")
        return []
    
    print(f"Calculation complete. Found {len(all_purity_ratios)} clusters in total.")

    # --- Plot the aggregated results ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_purity_ratios, bins=50, range=(0, 1.5), alpha=0.8, edgecolor='black', color='purple')
    
    plt.xlabel("Cluster Energy Purity (E_cluster / E_truth_particles)")
    plt.ylabel("Total Number of Clusters (from all events)")
    plt.title(f"Aggregated Histogram of Cluster Purity for {len(event_indices)} Events")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect Purity (Ratio = 1.0)')
    plt.legend()
    
    if show:
        plt.tight_layout()
        plt.show()
        
    return all_purity_ratios