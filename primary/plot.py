import polars as pl
import matplotlib.pyplot as plt
from primary.preprocessing import cluster_purity, particle_energy_calo_deposits_ratio
import plotly.express as px
from primary.preprocessing import particle_purity_by_class

import numpy as np
from primary.pdg_mappings import PDG_ID_TO_NAME

import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from collections import defaultdict, deque

def plot_production_time_histogram(df: pl.DataFrame, n_events: int = 100, bins: int = 50, log_scale: bool = True, filter_expression:pl.Expr=None):
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
    if filter_expression is not None:
        time_series = (
            df.head(n_events)
            .explode(pl.col(pl.List)) # Flatten all lists to align particles
            .filter(filter_expression)
            .select(pl.col("time"))
            #.explode("time")
            .drop_nulls() # Safety check for empty lists
        )
    else:
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
    
    plt.title(f"Particle Production Time Distribution (First {n_events} Events) with filter={filter_expression} ")
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


def plot_cluster_cardinallity(calo_hits_with_clusters:pl.DataFrame)->None:
    """
    Plots the distribution of cluster cardinalities (number of hits per cluster).
    """
    # Explode to align hits with clusters
    exploded = calo_hits_with_clusters.select(['event_id', 'cluster_id']).explode([ 'cluster_id'])
    
    # Count hits per cluster
    cluster_counts = (
        exploded
        .group_by(['event_id', 'cluster_id'])
        .agg(pl.count().alias('hit_count'))
    )
    
    plt.figure(figsize=(10,6))
    plt.hist(cluster_counts['hit_count'].to_numpy(), bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.title("Cluster Cardinality Distribution")
    plt.xlabel("Number of Hits in Cluster")
    plt.ylabel("Number of Clusters")
    plt.grid(axis='y', alpha=0.5)
    plt.show()

def plot_clusters_purity(calo_hits_with_clusters: pl.DataFrame, ancestors: pl.DataFrame) -> pl.DataFrame:
    purity_df = cluster_purity(calo_hits_with_clusters, ancestors)
    purity_df = (
        purity_df
        .sort(["purity", "cluster_id"], descending=[True, False])
        .unique(subset=["event_id", "ultimate_ancestor_id"], keep="first")
        .select(["event_id", "cluster_id", "purity", "ultimate_ancestor_id"])
    )

    purity_np = purity_df["purity"].to_numpy()
    counts, bin_edges = np.histogram(purity_np, bins=50)
    total = counts.sum()
    tail_eff = (counts[-1] + counts[-2]) / total if total else 0.0
    tail_lower_edge = bin_edges[-2] if len(bin_edges) >= 2 else 0.0

    plt.figure(figsize=(10, 6))
    plt.hist(purity_np, bins=50, color="seagreen", edgecolor="black", alpha=0.7)
    plt.title(f"Event Partitionning , Particle Purity- energy wise. E_largest_partition/E_total_deposited (tail ≥ {tail_lower_edge:.2f}: {tail_eff:.2%})")
    plt.xlabel("Purity")
    plt.ylabel("Number of particles")
    plt.yscale('log')
    plt.grid(axis="y", alpha=0.5)
    plt.show()

    return purity_df

def plot_particle_purity(   calo_hits: pl.DataFrame, 
    ancestors: pl.DataFrame, 
    particles: pl.DataFrame)->None:
    purity_df = particle_energy_calo_deposits_ratio(calo_hits, ancestors, particles)
    # just group by 
    purity_df =(
    purity_df.select([ "purity"]    )
)
    plt.figure(figsize=(10,6))
    plt.hist(purity_df['purity'].to_numpy(), bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    plt.title("Particle calo deps ratio Distribution")
    plt.xlabel("E_particle decendants deposited in calo / E_particle")
    plt.ylabel("Number of particles")
    # log scale
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.5)
    plt.show()

def plot_particle_purity_by_class(
    calo_hits: pl.DataFrame, 
    ancestors: pl.DataFrame, 
    particles: pl.DataFrame,
    pdg_classes: list
) -> None:
    
    # 1. Calculate Purity
    purity_df = particle_purity_by_class(calo_hits, ancestors, particles, pdg_classes)

    # 2. Select columns (ensure total_particle_energy is kept)
    purity_df = purity_df.select([
        "class_id", 
        "purity", 
        "total_particle_energy",
        "pdg_id"
    ])

    # 3. Define the intervals: (Low, High)
    # The last tuple (50, None) represents 50 -> Infinity
    energy_intervals = [(-0.1, 0.001), (0.001, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, None)]
    
    # 4. Partition by class_id (Efficient separation)
    class_partitions = purity_df.partition_by("class_id", as_dict=True)

    

    # 5. Iterate over classes
    for class_id, df_class in class_partitions.items():
        
        plt.figure(figsize=(10, 6))
        
        # --- DYNAMIC RANGE CALCULATION ---
        # Since purity variance is huge, we calculate the range for THIS class
        # to ensure the histogram covers the data properly.
        min_p = df_class["purity"].min()
        max_p = df_class["purity"].max()
        
        # Create 50 shared bins for this class so all energy lines align on x-axis
        # If max_p is huge (outliers), consider using percentile (e.g., 98th) to clip
        bins = np.linspace(min_p, max_p, 50)

        # Colors for the 4 intervals
        colors = plt.cm.turbo(np.linspace(0, 1, len(energy_intervals)))
        
        has_data = False

        for i, (low_e, high_e) in enumerate(energy_intervals):
            
            # Construct the filter
            if high_e is not None:
                # Normal range: low <= E < high
                condition = (pl.col("total_particle_energy") >= low_e) & \
                            (pl.col("total_particle_energy") < high_e)
                label = f"{low_e} < E < {high_e}"
            else:
                # Overflow range: E >= 50
                condition = (pl.col("total_particle_energy") >= low_e)
                label = f"E >{low_e}"
                
            subset = df_class.filter(condition)
            if not subset.is_empty():
                x=3
                particles_with_counts = subset.select(['pdg_id']).group_by('pdg_id').count().sort('count', descending=True).head(10)
                particles = particles_with_counts['pdg_id'].to_list()
                counts = particles_with_counts['count'].to_list()
                particle_names = [PDG_ID_TO_NAME.get(str(pdg), str(pdg)) for pdg in particles]
                particles_count_str = [f"{name} ({count})" for name, count in zip(particle_names, counts)]
                label +=  " Particles found: " + ", ".join(particles_count_str)


            if not subset.is_empty():
                has_data = True
                plt.hist(
                    subset["purity"], 
                    bins=bins,           # Use shared bins!
                    histtype='step',     # Step ensures we see overlapping lines
                    linewidth=2,
                    label=label,
                    color=colors[i]
                )

        if not has_data:
            plt.close()
            continue

        plt.title(f"Distribution for Class {class_id}")
        plt.xlabel(" (Calo deps by descendants Energy / Particle Energy)")
        plt.ylabel("Amount of particles")
        
        # Log scale helps if the tail is very long
        plt.grid(axis='y', alpha=0.3, which='both')
        plt.legend(
            title="Particle Energy [GeV]",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )
        # log scale for y-axis
        plt.yscale('log')
        # Optional: Limit x-axis if variance is TOO huge (e.g., max purity is 1000 but 99% data is < 2)
        # plt.xlim(min_p, np.percentile(df_class["purity"], 99)) 
        
        plt.show()

import polars as pl
import matplotlib.pyplot as plt

def plot_ancestor_distribution(
    df: pl.DataFrame, 
    bins: int = 50, 
    log_scale: bool = False,
    figsize: tuple = (10, 6)
) -> pl.DataFrame:
    """
    Aggregates the number of unique ultimate ancestors per event and plots a histogram.
    
    Returns:
        pl.DataFrame: The aggregated data [event_idx, ancestor_count]
    """
    print("Aggregating unique ancestors per event...")

    # 1. Aggregate: Count unique ancestors per event
    stats = (
        df.lazy()
        .group_by("event_idx")
        .agg(
            pl.col("ultimate_ancestor_id").n_unique().alias("ancestor_count")
        )
        .collect()
    )

    # 2. Print Basic Statistics
    print("-" * 30)
    print(f"Total Events: {stats.height}")
    print(f"Mean Ancestors/Event: {stats['ancestor_count'].mean():.2f}")
    print(f"Median Ancestors/Event: {stats['ancestor_count'].median():.1f}")
    print(f"Max Ancestors/Event:  {stats['ancestor_count'].max()}")
    print("-" * 30)

    # 3. Plotting
    plt.figure(figsize=figsize)
    
    # Extract data as numpy array for matplotlib
    data = stats["ancestor_count"]
    
    plt.hist(data, bins=bins, color='#1f77b4', edgecolor='black', alpha=0.8)
    
    plt.title("Distribution of Unique Ultimate Ancestors per Event", fontsize=14)
    plt.xlabel("Count of Unique Ancestors", fontsize=12)
    plt.ylabel("Number of Events", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Number of Events (Log Scale)", fontsize=12)

    plt.tight_layout()
    plt.show()

    return stats


def plot_target_vs_truth_energy_sum(particles: pl.DataFrame, eta_cut: float = 3.5, pt_cut: float =0.5):
    """
    Plots the sum of energies of target particles vs. truth particles per event.
    """
    # Filter target particles
    target_energy_sum = (particles.lazy()
                        .select(['event_id', 'energy', 'is_target_particle', 'eta', 'pt','charge'])
                        .explode(['energy', 'is_target_particle', 'eta', 'pt', 'charge'])
                        .filter(
                            (pl.col('is_target_particle')) &
                            (pl.col('eta').abs() < eta_cut) &
                            (pl.col('pt') > pt_cut) #&
                        #(pl.col('charge').abs() > 0)
                        )
                        .group_by('event_id').agg(pl.col('energy').sum().alias('target_energy_sum'))
                        )


    # Sum energies per event for truth particles
    truth_energy_sum = (
        particles.lazy()
       .select(['event_id', 'energy', 'is_parent_missing', 'eta', 'pt', 'charge', 'pdg_id'])
          .explode(['energy', 'is_parent_missing', 'eta', 'pt', 'charge', 'pdg_id'])
        .filter(
            (pl.col('is_parent_missing')) &
            (pl.col('eta').abs() < eta_cut) &
            (pl.col('pt') > pt_cut) &
            ((pl.col('pdg_id').abs() != 12) & (pl.col('pdg_id').abs() != 14) & (pl.col('pdg_id').abs() != 16) )
              #&
            #(pl.col('charge').abs() > 0)
        )
        .group_by('event_id')
        .agg(pl.col('energy').sum().alias('truth_energy_sum'))
    )

    # Join the two sums on event_id
    energy_comparison = target_energy_sum.join(truth_energy_sum, on='event_id', how='inner').collect()

    # Convert to numpy for plotting
    x = energy_comparison['truth_energy_sum'].to_numpy()
    y = energy_comparison['target_energy_sum'].to_numpy()

    # Calculate ratio
    ratio = y / x

    plt.figure(figsize=(10, 6))
    plt.hist(ratio, bins=50, color='purple', edgecolor='black', alpha=0.7)
    plt.title(f"Ratio of Target Energy / Truth Energy (eta_cut={eta_cut}, pt_cut={pt_cut})")
    plt.xlabel("Energy Ratio (Target / Truth)")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    plt.show()


def plot_num_contributing_clusters(
    calo: pl.DataFrame,
    result: pl.DataFrame,
    particles: pl.DataFrame,
    cut_off_percent: float = 0.05,
    pt_cut: float = 1.0,
    eta_cut: float = 3.0,
    log_scale: bool = True,
    figsize: tuple = (12, 5),
) -> pl.DataFrame:
    """
    Histogram of number of contributing clusters per particle with customizable cuts.
    
    Args:
        calo: DataFrame with calorimeter hits and cluster information.
        result: DataFrame with backtracked particle-cluster associations.
        particles: DataFrame with particle properties (pt, eta).
        cut_off_percent: Cutoff percentage for cluster contribution filtering (default: 0.05).
        pt_cut: Transverse momentum cut in GeV (default: 1.0).
        eta_cut: Pseudorapidity cut (default: 3.0).
        log_scale: Whether to use log scale for y-axis (default: True).
        figsize: Figure size as (width, height) tuple (default: (12, 5)).
    """
    from primary.preprocessing import number_of_clusters_per_particle
    
    # Compute cluster statistics per particle with the specified cuts
    ancestor_stats = number_of_clusters_per_particle(
        calo,
        result,
        particles=particles.filter(pl.col('event_id').is_in(calo['event_id'].unique().implode())),
        cut_off_percent=cut_off_percent,
        pt_cut=pt_cut,
        eta_cut=eta_cut
    )
    
    # 1. Get the data
    data = ancestor_stats["num_contributing_clusters"].to_numpy()
    
    # 2. Calculate min/max to define discrete integer bins
    min_val = int(data.min())
    max_val = int(data.max())
    
    # Create bins centered on integers: [min-0.5, min+0.5, min+1.5, ..., max+0.5]
    # This ensures each integer gets its own bar
    discrete_bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    
    # 3. Plot
    fig, ax = plt.subplots(figsize=figsize)
    counts, bins, patches = ax.hist(data, bins=discrete_bins, color="steelblue", edgecolor="black", alpha=0.75)
    
    # Calculate percentage of first bin
    first_bin_count = counts[0]
    total_count = sum(counts)
    first_bin_percentage = (first_bin_count / total_count) * 100
    
    second_bin_count = counts[1]
    second_bin_percentage = (second_bin_count / total_count) * 100
    # Set labels and title with cuts information
    ax.set_xlabel("Number of contributing clusters")
    ax.set_ylabel("Count")
    title = f"#cluster/particle distribution (First bin(0): {first_bin_percentage:.1f}%, Second bin(1): {second_bin_percentage:.1f}%)"
    title += f"\nCuts: pt>{pt_cut} GeV, |η|<{eta_cut}, particle_contrib_cut_off={cut_off_percent*100:.1f}%"
    ax.set_title(title)
    
    # Grid settings
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.15)  # Add faint vertical grid lines for clarity
    
    # Log scale
    if log_scale:
        ax.set_yscale("log")
    
    plt.tight_layout()
    plt.show()
    return ancestor_stats


def plot_num_contributing_ancestors(
    calo: pl.DataFrame,
    result: pl.DataFrame,
    particles: pl.DataFrame,
    cut_off_percent: float = 0.05,
    pt_cut: float = 1.0,
    eta_cut: float = 3.0,
    log_scale: bool = True,
    figsize: tuple = (12, 5),
) -> pl.DataFrame:
    """
    Histogram of number of particles per cluster with customizable cuts.
    
    Args:
        calo: DataFrame with calorimeter hits and cluster information.
        result: DataFrame with backtracked particle-cluster associations.
        particles: DataFrame with particle properties (pt, eta).
        cut_off_percent: Cutoff percentage for cluster contribution filtering (default: 0.05).
        pt_cut: Transverse momentum cut in GeV (default: 1.0).
        eta_cut: Pseudorapidity cut (default: 3.0).
        log_scale: Whether to use log scale for y-axis (default: True).
        figsize: Figure size as (width, height) tuple (default: (12, 5)).
    """
    from primary.preprocessing import number_of_particles_per_cluster
    
    # Compute particle statistics per cluster with the specified cuts
    cluster_stats = number_of_particles_per_cluster(
        calo,
        result,
        particles=particles.filter(pl.col('event_id').is_in(calo['event_id'].unique().implode())),
        cut_off_percent=cut_off_percent,
        pt_cut=pt_cut,
        eta_cut=eta_cut
    )
    
    # 1. Get the data
    data = cluster_stats["num_contributing_ancestors"].to_numpy()
    
    # 2. Calculate min/max to define discrete integer bins
    min_val = int(data.min())
    max_val = int(data.max())
    
    # Create bins centered on integers: [min-0.5, min+0.5, min+1.5, ..., max+0.5]
    # This ensures each integer gets its own bar
    discrete_bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    
    # 3. Plot
    fig, ax = plt.subplots(figsize=figsize)
    counts, bins, patches = ax.hist(data, bins=discrete_bins, color="steelblue", edgecolor="black", alpha=0.75)
    
    # Calculate percentage of first bin
    first_bin_count = counts[0]
    total_count = sum(counts)
    first_bin_percentage = (first_bin_count / total_count) * 100
    
    # Set labels and title with cuts information
    ax.set_xlabel("Number of particles per cluster")
    ax.set_ylabel("Count")
    title = f"#particles/cluster distribution (First bin: {first_bin_percentage:.1f}%)"
    title += f"\nCuts: pt>{pt_cut} GeV, |η|<{eta_cut}, cluster_contrib_cut_off={cut_off_percent*100:.1f}%"
    ax.set_title(title)
    
    # Grid settings
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.15)  # Add faint vertical grid lines for clarity
    
    # Log scale
    if log_scale:
        ax.set_yscale("log")
    
    plt.tight_layout()
    plt.show()
    return cluster_stats

def histogram_ht_ratio(particles: pl.DataFrame)-> pl.DataFrame:
    # sum pt of target particles within eta cut, pt cut
    ht_target = (
    particles.lazy()
    .select(['event_id', 'particle_id','pt', 'eta', 'is_target_particle', 'pdg_id'])
    .explode(['particle_id','pt', 'eta', 'is_target_particle', 'pdg_id'])
    .filter((pl.col('is_target_particle') ) &
            # filter out neutrinos
             ((pl.col('pdg_id').abs() != 12) & (pl.col('pdg_id').abs() != 14) & (pl.col('pdg_id').abs() != 16) )
            )
    .group_by('event_id')
    .agg(pl.col('pt').sum().alias('ht_target'))
    )

    ht_truth =(
    particles.lazy()
    .select(['event_id', 'particle_id','pt', 'eta', 'is_truth_particle', 'pdg_id'])
    .explode(['particle_id','pt', 'eta', 'is_truth_particle', 'pdg_id'])
    .filter((pl.col('is_truth_particle')) &
               ((pl.col('pdg_id').abs() != 12) & (pl.col('pdg_id').abs() != 14) & (pl.col('pdg_id').abs() != 16) )
               )
    .group_by('event_id')
    .agg(pl.col('pt').sum().alias('ht_truth'))
    )
    ht_combined = ht_target.join(ht_truth, on='event_id', how='inner').collect(streaming=True)

    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy for plotting
    x = ht_combined['ht_truth'].to_numpy()
    y = ht_combined['ht_target'].to_numpy()

    # Calculate ratio
    ratio = np.divide(y, x, out=np.zeros_like(y), where=x!=0)

    # Calculate statistics
    mean_ratio = np.mean(ratio)
    std_ratio = np.std(ratio)
    n_events = len(ratio)

    plt.figure(figsize=(10, 6))
    plt.hist(ratio, bins=50, range=(0, 2), color='blue', edgecolor='black', alpha=0.7)
    plt.title(f"Ratio of Target HT / Truth HT - Mean: {mean_ratio:.2f}, Std: {std_ratio:.2f}, N: {n_events}")
    plt.xlabel("HT Target / HT Truth")
    plt.ylabel("Number of Events") 
    return ht_combined

def histogram_energy_ratio(particles: pl.DataFrame)-> pl.DataFrame:
    # sum pt of target particles within eta cut, pt cut
    ht_target = (
    particles.lazy()
    .select(['event_id', 'particle_id','energy', 'eta', 'is_target_particle', 'pdg_id'])
    .explode(['particle_id','energy', 'eta', 'is_target_particle', 'pdg_id'])
    .filter((pl.col('is_target_particle') ) &
            # filter out neutrinos
             ((pl.col('pdg_id').abs() != 12) & (pl.col('pdg_id').abs() != 14) & (pl.col('pdg_id').abs() != 16) )
            )
    .group_by('event_id')
    .agg(pl.col('energy').sum().alias('energy_target'))
    )

    ht_truth =(
    particles.lazy()
    .select(['event_id', 'particle_id','energy', 'eta', 'is_truth_particle', 'pdg_id'])
    .explode(['particle_id','energy', 'eta', 'is_truth_particle', 'pdg_id'])
    .filter((pl.col('is_truth_particle')) &
               ((pl.col('pdg_id').abs() != 12) & (pl.col('pdg_id').abs() != 14) & (pl.col('pdg_id').abs() != 16) )
               )
    .group_by('event_id')
    .agg(pl.col('energy').sum().alias('energy_truth'))
    )
    ht_combined = ht_target.join(ht_truth, on='event_id', how='inner').collect(streaming=True)

    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy for plotting
    x = ht_combined['energy_truth'].to_numpy()
    y = ht_combined['energy_target'].to_numpy()
    # Calculate ratio
    ratio = np.divide(y, x, out=np.zeros_like(y), where=x!=0)

    # Calculate statistics
    mean_ratio = np.mean(ratio)
    std_ratio = np.std(ratio)
    n_events = len(ratio)

    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 2.05, 0.05)
    plt.hist(ratio, bins=bins, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f"Ratio of Target Sum Energy / Truth Sum Energy - Mean: {mean_ratio:.2f}, Std: {std_ratio:.2f}, N: {n_events}, bins=np.arange(0, 2.05, 0.05)")
    plt.xlabel("Energy Target / Energy Truth")
    plt.ylabel("Number of Events") 
    return ht_combined


import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def hist2d_ht_truth_vs_target(particles: pl.DataFrame, eta_cut: float, pt_cut: float, bins=100) -> pl.DataFrame:
    """
    Plots a 2D histogram of Truth HT vs Target HT.
    Returns the combined DataFrame containing the calculated HT values.
    """
    
    # --- 1. Calculate Target HT ---
    ht_target = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'pt', 'eta', 'is_target_particle', 'pdg_id'])
        .explode(['particle_id', 'pt', 'eta', 'is_target_particle', 'pdg_id'])
        .filter(
            (pl.col('is_target_particle')) &
            (pl.col('eta').abs() < eta_cut) &
            (pl.col('pt') > pt_cut) &
            # filter out neutrinos (12, 14, 16)
            (~pl.col('pdg_id').abs().is_in([12, 14, 16]))
        )
        .group_by('event_id')
        .agg(pl.col('pt').sum().alias('ht_target'))
    )

    # --- 2. Calculate Truth HT ---
    ht_truth = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'pt', 'eta', 'is_parent_missing', 'pdg_id'])
        .explode(['particle_id', 'pt', 'eta', 'is_parent_missing', 'pdg_id'])
        .filter(
            (pl.col('is_parent_missing')) &
            (pl.col('eta').abs() < eta_cut) &
            (pl.col('pt') > pt_cut) &
            # filter out neutrinos (12, 14, 16)
            (~pl.col('pdg_id').abs().is_in([12, 14, 16]))
        )
        .group_by('event_id')
        .agg(pl.col('pt').sum().alias('ht_truth'))
    )

    # --- 3. Join Data ---
    ht_combined = ht_target.join(ht_truth, on='event_id', how='inner').collect(streaming=True)

    # --- 4. Plotting ---
    x = ht_combined['ht_truth'].to_numpy()
    y = ht_combined['ht_target'].to_numpy()

    # Determine plot limits based on data max
    max_val = max(np.max(x), np.max(y)) * 1.05

    plt.figure(figsize=(10, 8))
    
    # 2D Histogram
    # cmin=1 ensures bins with 0 count are white/transparent rather than the lowest color
    h = plt.hist2d(x, y, bins=bins, range=[[0, max_val], [0, max_val]], 
                   cmap='viridis', norm=LogNorm(), cmin=1)
    
    # Add colorbar
    cbar = plt.colorbar(h[3])
    cbar.set_label('Number of Events')

    # Add diagonal reference line (Ideal y=x)
    plt.plot([0, max_val], [0, max_val], 'g--', linewidth=1.5, label='Ideal (y=x)')
    
    # Add orange line for y=0.9x
    plt.plot([0, max_val], [0, 0.9 * max_val], 'o--', linewidth=1.5, label='y=0.9x')
    plt.plot([0, max_val], [0, 0.7 * max_val], 'r--', linewidth=1.5, label='y=0.7x')

    
    # Labels and Title
    plt.xlabel("Truth HT")
    plt.ylabel("Target HT")
    plt.title(f"Truth HT vs Target HT\n(eta_cut={eta_cut}, pt_cut={pt_cut})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return ht_combined

def plot_calo_cluster_cutoff_plots(calo: pl.DataFrame):

    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import numpy as np
    from primary.calibration import CALIBRATION

    cluster_stats = (
        calo.lazy()
        .select(["event_id", "cluster_id"])
        .explode("cluster_id")
        .group_by(["event_id", "cluster_id"])
        .agg(pl.count().alias("count"))
        .rename({"count": "cluster_size"})
        .collect()
    )
    size_cdf = (
        cluster_stats.lazy()
        .group_by("cluster_size")
        .agg(pl.count().alias("num_clusters"))
        .sort("cluster_size")
        .collect()
        .with_columns([
            pl.col("num_clusters").cum_sum().shift(1).fill_null(0).alias("clusters_lt"),
        ])
        .with_columns([
            (pl.col("clusters_lt") / pl.col("num_clusters").sum() * 100).alias("percentage_lt"),
        ])
    )

    fig = go.Figure(
        go.Scatter(
            x=size_cdf["cluster_size"],
            y=size_cdf["percentage_lt"],
            mode="markers+lines",
            marker=dict(size=8, color="mediumslateblue"),
            hovertemplate="Partition size < %{x}<br>Share: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cumulative share of partitions by size",
        xaxis_title="Partition size (cells)",
        yaxis_title="Percentage of partitions",
        yaxis=dict(range=[0, 100]),
    )
    fig.show()



    cluster_energy_stats = (
        calo.lazy()
        .select(["event_id", "cluster_id", "detector", "total_energy"])
        .explode(["cluster_id", "detector", "total_energy"])
        .join(
            CALIBRATION.lazy().select(["detector", "calib_factor"]),
            on="detector",
        )
        .group_by(["event_id", "cluster_id"])
        .agg([
            pl.len().alias("cluster_size"),
            (pl.col("total_energy") * pl.col("calib_factor")).sum().alias("cluster_energy"),
        ])
        .collect(streaming=True)
    )

    # 2. Calculate Global Energy Distribution
    energy_cdf = (
        cluster_energy_stats.lazy()
        # BUG FIX: Group ONLY by cluster_size. 
        # We want to sum the energy of ALL clusters of size X in the entire dataset.
        .group_by("cluster_size")
        .agg([
            pl.col("cluster_energy").sum().alias("total_energy_at_size"),
        ])
        .sort("cluster_size")
        .with_columns([
            # Cumulative sum of the GLOBAL energy
            pl.col("total_energy_at_size")
            .cum_sum()
            .shift(1)
            .fill_null(0)
            .alias("energy_lt_global")
        ])
        .with_columns([
            # Normalize by the grand total energy of the entire dataset
            (pl.col("energy_lt_global") / pl.col("total_energy_at_size").sum() * 100)
            .alias("avg_energy_pct")
        ])
        .collect()
    )

    # 3. Plot
    fig = go.Figure(
        go.Scatter(
            x=energy_cdf["cluster_size"],
            y=energy_cdf["avg_energy_pct"],
            mode="markers+lines",
            marker=dict(size=6, color="darkorchid"),
            line=dict(width=2),
            hovertemplate="Partition size < %{x}<br>Total Energy share: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cumulative share of energy by partition size (Global)",
        xaxis_title="Partition size (cells)",
        yaxis_title="Energy share [%]",
        yaxis=dict(range=[0, 100]),
        template="plotly_white"
    )
    fig.show()


