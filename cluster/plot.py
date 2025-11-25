# 3D interactive calo cluster plot (per event) colored by MeanShift cluster

import numpy as np
import plotly.express as px
from cluster.helpers.meanshift_mod import MeanShiftMod
from cluster.play import to_eta, convert_to_cartesian_eta_phi, get_points_for_clustering

from cluster.pdg_mappings import PDG_ID_TO_NAME
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

def plot_calo_clusters_3d_given(event_idx, points_clustring_info:dict, ms: MeanShiftMod, all_datasets, dataset_name="OpenDataDetector/ColliderML_ttbar_pu0", show=True, mask_cluster_smaller_than=0):
    calo = all_datasets[dataset_name]["calo_hits"]["train"].with_format("numpy")
    ev = calo[event_idx]
    labels = ms.labels_
    data = points_clustring_info['points']
    mask_calo = points_clustring_info['mask_calo']
    e_mask = points_clustring_info['e_mask']
    e = np.asarray(ev["total_energy"], dtype=float)

    x = ev["x"][e_mask]
    y = ev["y"][e_mask]
    z = ev["z"][e_mask]
    # Selected per-hit metadata aligned with x,y,z
    eta =  data[:, 0][mask_calo]
    phi =  data[:, 1][mask_calo]
    e_sel = e[e_mask]
    labels = labels[mask_calo]

    unique, counts = np.unique(labels, return_counts=True)
    count_map = dict(zip(unique, counts))
    cluster_sizes = np.vectorize(count_map.get)(labels)

    mask_c_sizes = cluster_sizes >= mask_cluster_smaller_than
    x = x[mask_c_sizes]
    y = y[mask_c_sizes]
    z = z[mask_c_sizes]
    eta = eta[mask_c_sizes]
    phi = phi[mask_c_sizes]
    e_sel = e_sel[mask_c_sizes]
    labels = labels[mask_c_sizes]
    cluster_sizes = cluster_sizes[mask_c_sizes]
    
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
            "cluster_size": cluster_sizes,
            "energy": e_sel,
            "eta": eta,
            "phi": phi
        },
        title=f"Calo hits MeanShift clusters (event {event_idx}, n_clusters={len(np.unique(labels))})"
    )

    # --- Add Zoom Slider ---
    steps = []
    # Create steps from close (0.1) to far (2.5)
    for zoom in np.linspace(0.00001, 0.1, 250):
        step = dict(
            method="relayout",
            args=[{"scene.camera.eye": {"x": zoom, "y": zoom, "z": zoom}}],
            label=f"{zoom:.4f}"
        )
        steps.append(step)

    sliders = [dict(
        active=12, # Set default to roughly 1.3 (middle of range)
        currentvalue={"prefix": "Zoom Level: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z"
        ),
        sliders=sliders
    )
    if show:
        fig.show()
    return labels, ms.cluster_centers_

def plot_calo_clusters_3d(all_datasets, dataset_name="OpenDataDetector/ColliderML_ttbar_pu0", event_idx=0, bandwidth=0.4, energy_threshold=0.0001, show=True):
    calo = all_datasets[dataset_name]["calo_hits"]["train"].with_format("numpy")
    points_list = get_points_for_clustering(all_datasets, dataset_name=dataset_name, energy_threshold=energy_threshold, until_index=event_idx+1)
    data = points_list[event_idx]['points']

    ms = MeanShiftMod(bandwidth=bandwidth, bin_seeding=True).fit(data)
    return plot_calo_clusters_3d_given(event_idx, points_list[event_idx], ms, all_datasets, dataset_name=dataset_name, show=show)


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
            ax_hist.hist(df['count'], bins=100, color="#ff7f0e", alpha=0.8, edgecolor='black')
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
        ax_hist.hist(concat_df['count'], bins=100, color="#ff7f0e", alpha=0.8, edgecolor='black')
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
    
    total_particles = len(all_completeness_ratios)
    perfect_count = int(np.sum(np.isclose(all_completeness_ratios, 1.0, atol=1e-6)))
    perfect_pct = (perfect_count / total_particles * 100.0) if total_particles else 0.0

    print(f"Calculation complete. Found {total_particles} particles in total.")

    # --- Plot the aggregated results ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_completeness_ratios, bins=50, range=(0, 1.2), alpha=0.8, edgecolor='black', color='orangered')
    
    plt.xlabel("Particle Energy Completeness (Largest Cluster Dep / Total Particle Energy)")
    plt.ylabel("Total Number of Particles (from all events)")
    plt.title(
        f"Aggregated Histogram of Particle Completeness for {len(event_indices)} Events "
        f"(perfect={perfect_pct:.1f}% | {perfect_count}/{total_particles} particles)"
    )
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

import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from collections import defaultdict, deque

def plot_hierarchy_with_distances(all_datasets_loaded, event_idx=0):
    """
    Hierarchy Explorer with Generation Distance.
    - Calculates "Generations" relative to selection (e.g., +1 Child, -2 Grandparent).
    - Shows distance in Hover Text and Info Box.
    """
    
    # --- 1. Data Loading ---
    dataset = all_datasets_loaded["OpenDataDetector/ColliderML_ttbar_pu0"]
    p_data = dataset["particles"]["train"].with_format("numpy")[event_idx]
    c_data = dataset["calo_hits"]["train"].with_format("numpy")[event_idx]

    # Arrays & Cleaning
    all_pids = p_data["particle_id"].astype(np.int64)
    
    raw_parents = p_data["parent_id"]
    raw_parents = np.nan_to_num(raw_parents, nan=0.0)
    all_parent_ids = raw_parents.astype(np.int64)
    
    all_vx = p_data["vx"]
    all_vy = p_data["vy"]
    
    c_x = c_data["x"]
    c_y = c_data["y"]
    c_contrib_ids = c_data["contrib_particle_ids"]
    c_contrib_enes = c_data["contrib_energies"]

    # --- 2. Graph Structure ---
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}
    parent_map = {}
    children_map = defaultdict(list)
    out_degree = defaultdict(int)

    for i, pid in enumerate(all_pids):
        par_id = all_parent_ids[i]
        if par_id != 0 and par_id != pid and par_id in pid_to_idx:
            parent_map[pid] = par_id
            children_map[par_id].append(pid)
            out_degree[par_id] += 1
        else:
            parent_map[pid] = None

    # --- 3. Energy Calculation ---
    direct_energy = defaultdict(float)
    pid_to_cells = defaultdict(set)
    
    for cell_i, (contribs, energies) in enumerate(zip(c_contrib_ids, c_contrib_enes)):
        if contribs is None: continue
        for pid, en in zip(contribs, energies):
            pid = int(pid)
            direct_energy[pid] += float(en)
            pid_to_cells[pid].add(cell_i)

    inclusive_energy = direct_energy.copy()
    queue = deque([pid for pid in all_pids if out_degree[pid] == 0])
    
    while queue:
        child_id = queue.popleft()
        par_id = parent_map.get(child_id)
        if par_id is not None:
            inclusive_energy[par_id] += inclusive_energy[child_id]
            out_degree[par_id] -= 1
            if out_degree[par_id] == 0:
                queue.append(par_id)
                
    max_e = max(inclusive_energy.values()) if inclusive_energy else 10.0

    # --- 4. Visual Setup ---
    state = {'selected_pid': None, 'min_energy': 0.0}

    layout = go.Layout(
        title=f"Event {event_idx} Generation Mapper",
        xaxis_title="Vertex X",
        yaxis_title="Vertex Y",
        template="plotly_white",
        height=700,
        hovermode='closest',
        clickmode='event+select',
        uirevision='static_cam'
    )

    # Traces
    trace_calo = go.Scattergl(
        x=c_x, y=c_y, mode='markers',
        marker=dict(symbol='square', size=4, color='orange', opacity=0.3),
        visible=False, name='Calorimeter'
    )
    
    trace_lines = go.Scatter(
        x=[], y=[], mode='lines',
        line=dict(color='#666', width=1.5),
        hoverinfo='skip', name='Links'
    )

    trace_particles = go.Scattergl(
        x=all_vx, y=all_vy, mode='markers',
        marker=dict(size=6, color='#ccc'),
        text=[], customdata=all_pids,
        name='Particles', hoverinfo='text'
    )

    fig = go.FigureWidget(data=[trace_calo, trace_lines, trace_particles], layout=layout)
    
    # Widgets
    info_box = widgets.HTML("<b>Select a particle to measure lineage distances.</b>")
    btn_calo = widgets.ToggleButton(description="Show Calo", value=False, icon='eye')
    slider_energy = widgets.FloatSlider(value=0, min=0, max=max_e/2, step=0.1, description='Min E:')

    # --- 5. Logic: Distance Mapping ---

    def get_hierarchy_map(center_pid):
        """
        Returns a dictionary: {pid: distance_from_center}
        0 = Center
        +N = Descendant generation N
        -N = Ancestor generation N
        """
        dist_map = {center_pid: 0}
        
        # 1. Descendants (BFS Downwards) -> Positive Distance
        queue = deque([(center_pid, 0)])
        # We keep track of visited descendants to handle DAGs/Cycles safely
        visited_desc = {center_pid}
        
        while queue:
            curr, dist = queue.popleft()
            # Add children
            for child in children_map.get(curr, []):
                if child not in visited_desc:
                    visited_desc.add(child)
                    new_dist = dist + 1
                    dist_map[child] = new_dist
                    queue.append((child, new_dist))

        # 2. Ancestors (Iterative Upwards) -> Negative Distance
        curr = center_pid
        dist = 0
        while True:
            par = parent_map.get(curr)
            if par is None: break
            
            # Safety check: Stop if we hit a node we already mapped (cycle)
            if par in dist_map: break
            
            dist -= 1
            dist_map[par] = dist
            curr = par
            
        return dist_map

    def update_view():
        sel_pid = state['selected_pid']
        min_e = state['min_energy']

        # A. Build Visible Set
        if sel_pid is None:
            # --- VIEW ALL ---
            visible_ids = [pid for pid in all_pids if inclusive_energy[pid] >= min_e]
            dist_map = {} # No distances available
            
            cols = ['#cccccc'] * len(visible_ids)
            sizes = [5] * len(visible_ids)
            texts = [f"PID: {pid}" for pid in visible_ids]
            lx, ly = [], [] # No lines
            
            title_txt = f"All Particles (> {min_e:.2f} GeV)"
            info_txt = f"<b>All Particles</b>: {len(visible_ids)} visible"
            display_set = set(visible_ids)
            
        else:
            # --- VIEW HIERARCHY ---
            dist_map = get_hierarchy_map(sel_pid)
            
            # Filter visible by Energy
            visible_ids = [pid for pid in dist_map.keys() if inclusive_energy[pid] >= min_e]
            display_set = set(visible_ids)
            
            cols, sizes, texts = [], [], []
            
            # Stats for Info Box
            n_anc = sum(1 for d in dist_map.values() if d < 0)
            n_desc = sum(1 for d in dist_map.values() if d > 0)
            max_gen = max(dist_map.values()) if dist_map else 0
            min_gen = min(dist_map.values()) if dist_map else 0

            for pid in visible_ids:
                gen = dist_map[pid]
                e_val = inclusive_energy[pid]
                
                # Generate Text Label
                if gen == 0:
                    role = "SELECTED"
                    gen_str = "0"
                elif gen == -1:
                    role = "Parent"
                    gen_str = "-1"
                elif gen == 1:
                    role = "Child"
                    gen_str = "+1"
                elif gen < -1:
                    role = "Ancestor"
                    gen_str = f"{gen}"
                else:
                    role = "Descendant"
                    gen_str = f"+{gen}"

                hover_txt = (f"<b>PID: {pid}</b><br>"
                             f"Role: {role}<br>"
                             f"Generation: {gen_str}<br>"
                             f"Energy: {e_val:.3f} GeV")
                texts.append(hover_txt)

                # Color Logic
                if gen == 0:
                    cols.append('#D62728') # Red
                    sizes.append(14)
                elif gen < 0:
                    cols.append('#1F77B4') # Blue
                    sizes.append(8)
                elif gen > 0:
                    cols.append('#2CA02C') # Green
                    sizes.append(8)

            # Line Drawing (Visible Parent -> Visible Child)
            lx, ly = [], []
            for pid in visible_ids:
                par = parent_map.get(pid)
                if par is not None and par in display_set:
                    p_i = pid_to_idx[par]
                    c_i = pid_to_idx[pid]
                    lx.extend([all_vx[p_i], all_vx[c_i], None])
                    ly.extend([all_vy[p_i], all_vy[c_i], None])

            title_txt = f"Hierarchy of PID {sel_pid} (Generations {min_gen} to +{max_gen})"
            
            info_txt = f"""
            <div style="border:1px solid #ccc; padding:10px; border-radius:5px;">
                <h3 style="margin:0; color:#D62728">Selected: {sel_pid}</h3>
                <hr>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div>
                        <span style="color:#1F77B4; font-weight:bold">Ancestors</span><br>
                        Count: {n_anc}<br>
                        Max Depth: {abs(min_gen)}
                    </div>
                    <div>
                        <span style="color:#2CA02C; font-weight:bold">Descendants</span><br>
                        Count: {n_desc}<br>
                        Max Gen: +{max_gen}
                    </div>
                </div>
                <hr>
                <b>Total Energy:</b> {inclusive_energy[sel_pid]:.4f} GeV
            </div>
            """

        # B. Update Plot Data
        px, py = [], []
        for pid in visible_ids:
            idx = pid_to_idx[pid]
            px.append(all_vx[idx])
            py.append(all_vy[idx])

        # C. Update Calo Data
        active_cells = set()
        for pid in visible_ids:
            active_cells.update(pid_to_cells[pid])
        cx = [c_x[i] for i in active_cells]
        cy = [c_y[i] for i in active_cells]

        with fig.batch_update():
            # Particles
            fig.data[2].x = px
            fig.data[2].y = py
            fig.data[2].marker.color = cols
            fig.data[2].marker.size = sizes
            fig.data[2].text = texts
            fig.data[2].customdata = visible_ids
            
            # Lines
            fig.data[1].x = lx
            fig.data[1].y = ly
            
            # Calo
            fig.data[0].x = cx
            fig.data[0].y = cy
            
            fig.layout.title = title_txt
            
        info_box.value = info_txt

    # --- 6. Interactions ---
    def on_click(trace, points, selector):
        if not points.point_inds: return
        clicked_pid = trace.customdata[points.point_inds[0]]
        
        if state['selected_pid'] == clicked_pid:
            state['selected_pid'] = None
        else:
            state['selected_pid'] = clicked_pid
        update_view()

    def on_slider(change):
        state['min_energy'] = change['new']
        update_view()

    def on_toggle(change):
        fig.data[0].visible = change['new']

    fig.data[2].on_click(on_click)
    slider_energy.observe(on_slider, names='value')
    btn_calo.observe(on_toggle, names='value')
    
    update_view()
    
    return widgets.VBox([widgets.HBox([btn_calo, slider_energy]), fig, info_box])




# --- Usage ---
# Assuming 'all_datasets_loaded' exists in your variable scope:

import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from collections import defaultdict, deque

def plot_3d_particle_hierarchy(all_datasets_loaded, event_idx=0):
    """
    3D Particle Hierarchy Explorer (X, Y, Z).
    - Interactive 3D rotation and zooming.
    - Lineage tracing with generation coloring.
    - Recursive energy calculation.
    """
    
    # --- 1. Data Loading ---
    dataset = all_datasets_loaded["OpenDataDetector/ColliderML_ttbar_pu0"]
    p_data = dataset["particles"]["train"].with_format("numpy")[event_idx]
    c_data = dataset["calo_hits"]["train"].with_format("numpy")[event_idx]

    # Particle Data
    all_pids = p_data["particle_id"].astype(np.int64)
    all_pdg_ids = p_data["pdg_id"]
    pid_to_pdg = dict(zip(all_pids, all_pdg_ids))
    
    # Handle Parents (clean NaNs)
    raw_parents = p_data["parent_id"]
    raw_parents = np.nan_to_num(raw_parents, nan=0.0)
    all_parent_ids = raw_parents.astype(np.int64)
    
    # 3D Coordinates
    all_vx = p_data["vx"]
    all_vy = p_data["vy"]
    all_vz = p_data["vz"] # Added Z
    
    # Calo Data
    c_x = c_data["x"]
    c_y = c_data["y"]
    c_z = c_data["z"] # Added Z
    c_contrib_ids = c_data["contrib_particle_ids"]
    c_contrib_enes = c_data["contrib_energies"]

    # --- 2. Build Graph & Energy ---
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}
    parent_map = {}
    children_map = defaultdict(list)
    out_degree = defaultdict(int)

    for i, pid in enumerate(all_pids):
        par_id = all_parent_ids[i]
        if par_id != 0 and par_id != pid and par_id in pid_to_idx:
            parent_map[pid] = par_id
            children_map[par_id].append(pid)
            out_degree[par_id] += 1
        else:
            parent_map[pid] = None

    # Energy Calculation
    direct_energy = defaultdict(float)
    pid_to_cells = defaultdict(set)
    
    for cell_i, (contribs, energies) in enumerate(zip(c_contrib_ids, c_contrib_enes)):
        if contribs is None: continue
        for pid, en in zip(contribs, energies):
            pid = int(pid)
            direct_energy[pid] += float(en)
            pid_to_cells[pid].add(cell_i)

    inclusive_energy = direct_energy.copy()
    queue = deque([pid for pid in all_pids if out_degree[pid] == 0])
    
    while queue:
        child_id = queue.popleft()
        par_id = parent_map.get(child_id)
        if par_id is not None:
            inclusive_energy[par_id] += inclusive_energy[child_id]
            out_degree[par_id] -= 1
            if out_degree[par_id] == 0:
                queue.append(par_id)
                
    max_e = max(inclusive_energy.values()) if inclusive_energy else 10.0

    # --- 3. Visualization Setup (3D) ---
    state = {'selected_pid': None, 'min_energy': 0.0}

    layout = go.Layout(
        title=f"Event {event_idx} 3D Topology",
        width=900, height=700,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode='data' # Ensures physics proportions are correct
        ),
        hovermode='closest',
        clickmode='event+select',
        uirevision='static_cam', # Preserves rotation on update
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Trace 0: Calo (3D)
    trace_calo = go.Scatter3d(
        x=c_x, y=c_y, z=c_z,
        mode='markers',
        marker=dict(size=3, color='orange', opacity=0.3),
        visible=False, name='Calo Hits'
    )

    # Trace 1: Normal Lines (3D)
    trace_norm = go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines',
        line=dict(color='#888', width=3),
        hoverinfo='skip', name='Link'
    )

    # Trace 2: Jump Lines (3D - Red)
    trace_jump = go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines',
        line=dict(color='red', width=4, dash='dot'),
        hoverinfo='skip', name='Data Jump'
    )

    # Trace 3: Particles (3D)
    trace_particles = go.Scatter3d(
        x=all_vx, y=all_vy, z=all_vz,
        mode='markers',
        marker=dict(size=5, color='#ccc'),
        text=[], customdata=all_pids,
        name='Particles', hoverinfo='text'
    )

    fig = go.FigureWidget(data=[trace_calo, trace_norm, trace_jump, trace_particles], layout=layout)
    
    # Widgets
    info_box = widgets.HTML("<b>Click a particle to explore its 3D lineage.</b>")
    btn_calo = widgets.ToggleButton(description="Show Calo", value=False, icon='cube')
    slider_energy = widgets.FloatSlider(value=0, min=0, max=max_e/2, step=0.1, description='Min E:')

    # --- 4. Logic ---

    def get_gen_map(center_pid):
        dmap = {center_pid: 0}
        # Down
        q = deque([(center_pid, 0)])
        visited = {center_pid}
        while q:
            curr, d = q.popleft()
            for child in children_map.get(curr, []):
                if child not in visited:
                    visited.add(child)
                    dmap[child] = d + 1
                    q.append((child, d + 1))
        # Up
        curr = center_pid
        d = 0
        while True:
            par = parent_map.get(curr)
            if par is None or par in dmap: break
            d -= 1
            dmap[par] = d
            curr = par
        return dmap

    def update_view():
        sel_pid = state['selected_pid']
        min_e = state['min_energy']

        # A. Filtering
        if sel_pid is None:
            # All mode
            visible = [p for p in all_pids if inclusive_energy[p] >= min_e-1e-5]
            
            cols = ['#dddddd'] * len(visible)
            sizes = [4] * len(visible)
            texts = []
            for p in visible:
                pdg = pid_to_pdg.get(p)
                name = PDG_ID_TO_NAME.get(str(pdg), str(pdg))
                texts.append(f"PID: {p}<br>Name: {name}")
            
            # No lines in All mode (too heavy for 3D)
            xn, yn, zn = [], [], []
            xj, yj, zj = [], [], []
            
            title_txt = f"All Particles (> {min_e:.2f} GeV)"
            info_html = f"Showing {len(visible)} particles in 3D."
            
        else:
            # Tree mode
            gen_map = get_gen_map(sel_pid)
            visible = [p for p in gen_map.keys() if inclusive_energy[p] >= min_e-1e-5]
            display_set = set(visible)
            
            cols, sizes, texts = [], [], []
            n_anc = sum(1 for v in gen_map.values() if v < 0)
            n_desc = sum(1 for v in gen_map.values() if v > 0)

            for pid in visible:
                gen = gen_map[pid]
                par = parent_map.get(pid)
                par_str = str(par) if par else "Root"
                
                pdg = pid_to_pdg.get(pid)
                name = PDG_ID_TO_NAME.get(str(pdg), str(pdg))
                
                texts.append(f"PID: {pid}, Name: {name}<br>Parent: {par_str}<br>Gen: {gen:+d}<br>E: {inclusive_energy[pid]:.2f}")
                
                if gen == 0:
                    cols.append('#D62728') # Red
                    sizes.append(10)
                elif gen < 0:
                    cols.append('#1F77B4') # Blue
                    sizes.append(6)
                else:
                    cols.append('#2CA02C') # Green
                    sizes.append(6)

            # Line Construction
            xn, yn, zn = [], [], []
            xj, yj, zj = [], [], []
            
            for pid in visible:
                par = parent_map.get(pid)
                if par is not None and par in display_set:
                    p_i, c_i = pid_to_idx[par], pid_to_idx[pid]
                    
                    gap = abs(gen_map[pid] - gen_map[par])
                    
                    # Append coords separated by None
                    coords_x = [all_vx[p_i], all_vx[c_i], None]
                    coords_y = [all_vy[p_i], all_vy[c_i], None]
                    coords_z = [all_vz[p_i], all_vz[c_i], None]
                    
                    if gap == 1:
                        xn.extend(coords_x); yn.extend(coords_y); zn.extend(coords_z)
                    else:
                        xj.extend(coords_x); yj.extend(coords_y); zj.extend(coords_z)

            pdg_sel = pid_to_pdg.get(sel_pid)
            name_sel = PDG_ID_TO_NAME.get(str(pdg_sel), str(pdg_sel))

            title_txt = f"Hierarchy: PID {sel_pid} ({name_sel})"
            info_html = f"""
            <div style="border:1px solid #ccc; padding:8px;">
                <h3 style="color:#D62728; margin:0;">PID: {sel_pid}</h3>
                <b>Type:</b> {name_sel} (PDG: {pdg_sel})<br>
                Ancestors: {n_anc} | Descendants: {n_desc}<br>
                <b>Total E:</b> {inclusive_energy[sel_pid]:.2f} GeV
            </div>
            """

        # B. Batch Update
        with fig.batch_update():
            # Update Particles
            idx_list = [pid_to_idx[p] for p in visible]
            fig.data[3].x = [all_vx[i] for i in idx_list]
            fig.data[3].y = [all_vy[i] for i in idx_list]
            fig.data[3].z = [all_vz[i] for i in idx_list]
            fig.data[3].marker.color = cols
            fig.data[3].marker.size = sizes
            fig.data[3].text = texts
            fig.data[3].customdata = visible
            
            # Update Lines
            fig.data[1].x = xn; fig.data[1].y = yn; fig.data[1].z = zn
            fig.data[2].x = xj; fig.data[2].y = yj; fig.data[2].z = zj
            
            # Update Calo
            if visible:
                active_cells = set()
                for p in visible: active_cells.update(pid_to_cells[p])
                fig.data[0].x = [c_x[i] for i in active_cells]
                fig.data[0].y = [c_y[i] for i in active_cells]
                fig.data[0].z = [c_z[i] for i in active_cells]
            else:
                fig.data[0].x = []; fig.data[0].y = []; fig.data[0].z = []
            
            fig.layout.title = title_txt
        
        info_box.value = info_html

    # --- 5. Handlers ---
    def on_click(trace, points, selector):
        if not points.point_inds: return
        clicked = trace.customdata[points.point_inds[0]]
        state['selected_pid'] = None if state['selected_pid'] == clicked else clicked
        update_view()

    fig.data[3].on_click(on_click)
    slider_energy.observe(lambda c: (state.update({'min_energy': c['new']}), update_view()), names='value')
    btn_calo.observe(lambda c: fig.data[0].update(visible=c['new']), names='value')

    update_view()
    return widgets.VBox([widgets.HBox([btn_calo, slider_energy]), fig, info_box])


def plot_calo_cells_3d_first_n_events(
    all_datasets_loaded,
    dataset_name="OpenDataDetector/ColliderML_ttbar_pu0",
    num_events=5,
    show=True,
):
    """Plot unique calorimeter cells from the first *num_events* as a 3D scatter plot.

    Each detector component receives a distinct color and hovering a marker reveals the
    underlying cell identifier alongside the originating event index.
    """

    if num_events < 1:
        raise ValueError("num_events must be >= 1")

    dataset = all_datasets_loaded[dataset_name]["calo_hits"]["train"].with_format("numpy")
    max_events = min(num_events, len(dataset))

    seen_cells = set()
    records = []

    for event_idx in range(max_events):
        event = dataset[event_idx]

        xs = np.asarray(event["x"], dtype=float)
        ys = np.asarray(event["y"], dtype=float)
        zs = np.asarray(event["z"], dtype=float)
        detectors = event["detector"]

        for i in range(len(detectors)):
            det = detectors[i]
            det_str = det.decode("utf-8") if isinstance(det, bytes) else str(det)
            key = (
                det_str,
                round(float(xs[i]), 5),
                round(float(ys[i]), 5),
                round(float(zs[i]), 5),
            )

            if key in seen_cells:
                continue

            seen_cells.add(key)
            records.append(
                {
                    "x": float(xs[i]),
                    "y": float(ys[i]),
                    "z": float(zs[i]),
                    "detector": det_str,
                    "event_index": event_idx,
                }
            )

    if not records:
        print("No calorimeter cells found for the requested events.")
        return None

    df = pd.DataFrame(records)

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="detector",
        hover_data={
            "event_index": True,
            "detector": False,
        },
        title=f"Unique Calorimeter Cells for First {max_events} Events",
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        legend_title_text="Calorimeter Component",
    )

    if show:
        fig.show()

    return fig



import polars as pl
import matplotlib.pyplot as plt

def plot_production_time_histogram(df: pl.DataFrame, n_events: int = 100, bins: int = 50, log_scale: bool = True):
    """
    Creates a histogram of particle production times for the first n_events.
    
    Args:
        df: The Polars DataFrame containing the dataset.
        n_events: Number of events to process.
        bins: Number of bins for the histogram.
    """
    print(f"Processing first {n_events} events for histogram...")

    # --- PERFORMANCE SECTION (C/Rust Backend) ---
    # 1. Slice: Take only the first n rows (Cheap operation, minimal memory)
    # 2. Select: Keep only the 'time' column
    # 3. Explode: Flatten the List<float> into a single contiguous Float Array.
    #    This runs in compiled Rust code.
    time_series = (
        df.head(n_events)
        .select(pl.col("time"))
        .explode("time")
        .drop_nulls() # Safety check for empty lists
    )

    # Convert the Polars Series (Rust) to a Numpy Array (C) for plotting
    # This is extremely fast as it's a contiguous memory dump.
    flat_times = time_series["time"].to_numpy()

    if len(flat_times) == 0:
        print("Warning: No particles found in the selected events.")
        return

    # --- PLOTTING SECTION ---
    plt.figure(figsize=(10, 6))
    
    # Matplotlib's hist is also C-optimized
    plt.hist(flat_times, bins=bins, color='royalblue', edgecolor='black', alpha=0.7)
    
    plt.title(f"Particle Production Time Distribution (First {n_events} Events)")
    plt.xlabel("Production Time (ns)")
    plt.ylabel("Count")

    plt.grid(axis='y', alpha=0.5)
    
    # Log scale is often useful for time if there are delayed decays
    if log_scale:   
        plt.yscale('log')
        plt.ylabel("Count (log scale)")
    
    plt.show()

    # Optional: Print stats using Polars fast aggregations
    print(f"Total particles plotted: {len(flat_times)}")
    print(f"Mean time: {flat_times.mean():.4f} ns")
    print(f"Max time:  {flat_times.max():.4f} ns")

# --- HOW TO USE ---
# Assuming 'df' is the dataframe you loaded in the previous step:
# plot_production_time_histogram(df, n_events=1000, bins=100)

def foo():
    pass