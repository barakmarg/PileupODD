import polars as pl
import matplotlib.pyplot as plt
from primary.preprocessing import cluster_purity
import plotly.express as px
import numpy as np

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

def plot_purity(calo_hits_with_clusters:pl.DataFrame, ancestors:pl.DataFrame)->None:
    purity_df = cluster_purity(calo_hits_with_clusters, ancestors)
    # just group by 
    purity_df =(
    purity_df
    # 1. Sort by purity descending. 
    # (Optional: Add 'cluster_id' ascending to break ties deterministically)
    .sort(["purity", "cluster_id"], descending=[True, False])
    
    # 2. Keep only the first row (highest purity) for every event/ancestor combo
    .unique(subset=["event_id", "ultimate_ancestor_id"], keep="first")
    
    # 3. Select only the requested columns
    .select(["event_id", "cluster_id", "purity", "ultimate_ancestor_id"])
)
    plt.figure(figsize=(10,6))
    plt.hist(purity_df['purity'].to_numpy(), bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    plt.title("Cluster Purity Distribution")
    plt.xlabel("Purity")
    plt.ylabel("Number of particles")
    plt.grid(axis='y', alpha=0.5)
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

# --- Usage ---
# stats_df = plot_ancestor_distribution(df)