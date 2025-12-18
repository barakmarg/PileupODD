import polars as pl
import numpy as np
import plotly.graph_objects as go
from primary.calibration import CALIBRATION
# -----------------------------------------------------------------------------
# Helper: Data Prep (Same as before)
# -----------------------------------------------------------------------------
def _prepare_event_data(calo_hits: pl.DataFrame, event_idx: int) -> pl.DataFrame:
    if "cluster_id" not in calo_hits.columns:
        raise ValueError("Column 'cluster_id' not found. Run clustering first.")

    return (
        calo_hits.lazy()
        .filter(pl.col("event_id") == event_idx)
        .select(['x', 'y', 'z', 'total_energy', 'detector', 'cluster_id'])
        .explode(['x', 'y', 'z', 'total_energy', 'detector', 'cluster_id'])
        .join(CALIBRATION.lazy(), on='detector', how='left')
        .with_columns([
            (pl.col('total_energy') * pl.col('calib_factor').fill_null(1.0)).alias('E')
        ])
        .collect()
    )

# -----------------------------------------------------------------------------
# Main Visualization Function
# -----------------------------------------------------------------------------
def plot_calo_clusters_interactive(calo_hits: pl.DataFrame, 
                                   event_idx: int = 0, 
                                   min_cluster_size: int = 0,
                                   show: bool = True):
    """
    Plots interactive 3D clusters with toggles for Cells vs Centroids.
    Calculates Energy-Weighted Center of Mass for centroids.
    """
    
    # --- 1. Get Cell Data ---
    cells_df = _prepare_event_data(calo_hits, event_idx)
    
    # Calculate size for filtering
    cells_df = cells_df.with_columns(pl.len().over("cluster_id").alias("cluster_size"))
    
    if min_cluster_size > 0:
        cells_df = cells_df.filter(pl.col("cluster_size") >= min_cluster_size)
    
    if len(cells_df) == 0:
        print("No clusters found.")
        return

    # --- 2. Calculate Cluster Centroids (Energy Weighted) ---
    # We group by cluster_id to get one row per cluster
    centroids_df = (
        cells_df.lazy()
        .group_by("cluster_id")
        .agg([
            # Weighted Mean for Position
            ((pl.col("x") * pl.col("E")).sum() / pl.col("E").sum()).alias("cx"),
            ((pl.col("y") * pl.col("E")).sum() / pl.col("E").sum()).alias("cy"),
            ((pl.col("z") * pl.col("E")).sum() / pl.col("E").sum()).alias("cz"),
            
            # Totals
            pl.col("E").sum().alias("total_energy"),
            pl.len().alias("num_cells")
        ])
        .collect()
    )

    # --- 3. Prepare Arrays for Plotting ---
    
    # A. Cell Data
    c_x = cells_df["x"].to_numpy()
    c_y = cells_df["y"].to_numpy()
    c_z = cells_df["z"].to_numpy()
    c_E = cells_df["E"].to_numpy()
    c_id = cells_df["cluster_id"].to_numpy()
    
    # B. Centroid Data
    k_x = centroids_df["cx"].to_numpy()
    k_y = centroids_df["cy"].to_numpy()
    k_z = centroids_df["cz"].to_numpy()
    k_E = centroids_df["total_energy"].to_numpy()
    k_id = centroids_df["cluster_id"].to_numpy()
    k_count = centroids_df["num_cells"].to_numpy()

    # --- 4. Sizing Logic (Normalization) ---
    
    # Cells: Size based on individual hit energy
    # Range: 2px to 8px
    c_sizes = np.clip(c_E, np.percentile(c_E, 5), np.percentile(c_E, 95))
    c_sizes = 2 + 6 * (c_sizes - c_sizes.min()) / (c_sizes.max() - c_sizes.min() + 1e-9)

    # Centroids: Size based on Total Cluster Energy
    # Range: 10px to 40px (Big blobs)
    k_sizes = np.clip(k_E, np.percentile(k_E, 5), np.percentile(k_E, 95))
    k_sizes = 10 + 30 * (k_sizes - k_sizes.min()) / (k_sizes.max() - k_sizes.min() + 1e-9)

    # --- 5. Create Traces (Graph Objects) ---
    
    # Trace 1: The Cells
    trace_cells = go.Scatter3d(
        x=c_x, y=c_y, z=c_z,
        mode='markers',
        name='Cells',
        marker=dict(
            size=c_sizes,
            color=c_id,          # Color by ID
            colorscale='Turbo',  # Vibrant colors
            opacity=1.0,         # Solid
            line=dict(width=0)
        ),
        # Custom Hover for Cells
        customdata=np.stack((c_id, c_E), axis=-1),
        hovertemplate=(
            "<b>Cell</b><br>" +
            "Cluster ID: %{customdata[0]}<br>" +
            "Energy: %{customdata[1]:.4f} GeV<br>" +
            "<extra></extra>"
        )
    )

    # Trace 2: The Centroids
    trace_centroids = go.Scatter3d(
        x=k_x, y=k_y, z=k_z,
        mode='markers',
        name='Clusters',
        marker=dict(
            size=k_sizes,
            color=k_id,          # Same ID = Same Color as cells
            colorscale='Turbo',
            opacity=0.4,         # Transparent "Ghost" blobs
            symbol='circle',
            line=dict(width=2, color='white') # White outline to pop
        ),
        # Custom Hover for Centroids
        customdata=np.stack((k_id, k_E, k_count), axis=-1),
        hovertemplate=(
            "<b>CLUSTER CENTER</b><br>" +
            "ID: %{customdata[0]}<br>" +
            "Total Energy: %{customdata[1]:.4f} GeV<br>" +
            "Cell Count: %{customdata[2]}<br>" +
            "<extra></extra>"
        )
    )

    fig = go.Figure(data=[trace_cells, trace_centroids])

    # --- 6. Interactivity: Buttons & Sliders ---
    
    # A. Visibility Buttons (The "Switches")
    # args[0] targets the 'visible' attribute of traces.
    # We have 2 traces: [Cells, Centroids]
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    label="Show All",
                    method="update",
                    args=[{"visible": [True, True]}]
                ),
                dict(
                    label="Cells Only",
                    method="update",
                    args=[{"visible": [True, False]}]
                ),
                dict(
                    label="Clusters Only",
                    method="update",
                    args=[{"visible": [False, True]}]
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.05,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )
    ]

    # B. Zoom Slider (Camera control)
    steps = []
    for zoom in np.linspace(0.1, 2.5, 50):
        steps.append(dict(
            method="relayout",
            args=[{"scene.camera.eye": {"x": zoom, "y": zoom, "z": zoom}}],
            label=f"{zoom:.1f}"
        ))
    
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Zoom: "},
        pad={"t": 50},
        steps=steps
    )]

    # --- 7. Layout Finalization ---
    fig.update_layout(
        title=f"Event {event_idx} | {len(k_id)} Clusters",
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)"
        ),
        updatemenus=updatemenus,
        sliders=sliders,
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=80)
    )

    if show:
        fig.show()
