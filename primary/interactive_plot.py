import polars as pl
import numpy as np
import plotly.graph_objects as go
from primary.calibration import CALIBRATION
from primary.pdg_mappings import PDG_ID_TO_NAME
# -----------------------------------------------------------------------------
# Helper: Data Prep (Same as before)
# -----------------------------------------------------------------------------
def _get_cluster_contributors(cells_df: pl.DataFrame, 
                              particles_df: pl.DataFrame, 
                              event_idx: int) -> pl.DataFrame:
    """
    Calculates the top 5 particles contributing energy to each cluster.
    """
    
    # 1. Explode to the Particle Contribution Level
    flat_contribs = (
        cells_df.lazy()
        .select(["cluster_id", "contrib_particle_ids", "contrib_energies", "calib_factor"])
        .explode(["contrib_particle_ids", "contrib_energies"])
        .rename({
            "contrib_particle_ids": "particle_id", 
            "contrib_energies": "raw_energy"
        })
        # Apply Calibration
        .with_columns(
            (pl.col("raw_energy") * pl.col("calib_factor").fill_null(1.0)).alias("calibrated_energy")
        )
    )

    # 2. Aggregate: Sum Calibrated Energy per (Cluster, Particle)
    cluster_particle_stats = (
        flat_contribs
        .group_by(["cluster_id", "particle_id"])
        .agg(pl.col("calibrated_energy").sum().alias("total_particle_E"))
        .filter(pl.col("total_particle_E") > 0)
    )

    # 3. Get Truth Names
    truth_map = (
        particles_df.lazy()
        .filter(pl.col("event_id") == event_idx)
        .select(["particle_id", "pdg_id"])
        .explode(["particle_id", "pdg_id"])
    )

    truth_ids = truth_map.collect()
    pdg_map_dict = {
        pid: PDG_ID_TO_NAME.get(str(pid), str(pid)) 
        for pid in truth_ids["pdg_id"].unique().to_list()
    }

    # 4. Format Output String
    formatted = (
        cluster_particle_stats
        .join(truth_map, on="particle_id", how="left")
        .collect()
        .with_columns(
            pl.col("pdg_id").replace(pdg_map_dict, default="Unknown").alias("particle_name")
        )
        # Sort so that the top energy contributors are first
        .sort("total_particle_E", descending=True)
        # Use maintain_order=True to keep the sorting when grouping
        .group_by("cluster_id", maintain_order=True)
        .agg([
            pl.format(
                # FIXED: Removed {:.3f}, used {} instead
                "<b>{}</b> (PDG:{}) ID:{}: <b>{} GeV</b>", 
                pl.col("particle_name"),
                pl.col("pdg_id"),
                pl.col("particle_id"),
                pl.col("total_particle_E").round(4) # Rounding happens here
            ).slice(0, 5).alias("top_strings")
        ])
        .with_columns(
            pl.col("top_strings").list.join("<br>").alias("contributors_text")
        )
        .select(["cluster_id", "contributors_text"])
    )
    
    return formatted

# -----------------------------------------------------------------------------
# Main Plotting Function
# -----------------------------------------------------------------------------
def plot_calo_clusters_3d(calo_hits: pl.DataFrame, 
                          particles: pl.DataFrame, 
                          event_idx: int = 0, 
                          min_cluster_size: int = 0,
                          show: bool = True):
    """
    Plots interactive 3D clusters using PRE-CALCULATED MeanShift centers.
    """
    
    # --- 1. Validation ---
    required = {'cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz', 
                'contrib_particle_ids', 'contrib_energies'}
    
    if not required.issubset(calo_hits.columns):
        missing = required - set(calo_hits.columns)
        raise ValueError(f"Missing columns: {missing}. Ensure calo_hits has cluster info and contributors.")

    # --- 2. Extract & Calibrate Event Data ---
    cells_df = (
        calo_hits.lazy()
        .filter(pl.col("event_id") == event_idx)
        .select([
            'x', 'y', 'z', 'total_energy', 'detector', 
            'cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz',
            'contrib_particle_ids', 'contrib_energies'
        ])
        .explode([
            'x', 'y', 'z', 'total_energy', 'detector', 
            'cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz',
            'contrib_particle_ids', 'contrib_energies'
        ])
        .join(CALIBRATION.lazy(), on='detector', how='left')
        .with_columns([
            (pl.col('total_energy') * pl.col('calib_factor').fill_null(1.0)).alias('E')
        ])
        .with_columns([
            pl.col("E").sum().over("cluster_id").alias("cluster_total_energy")
        ])
        .collect()
    )
    
    cells_df = cells_df.with_columns(pl.len().over("cluster_id").alias("cluster_size"))
    if min_cluster_size > 0:
        cells_df = cells_df.filter(pl.col("cluster_size") >= min_cluster_size)
    
    if len(cells_df) == 0:
        print(f"No clusters found for event {event_idx}.")
        return

    # --- 3. Prepare Cluster Centers ---
    unique_clusters = (
        cells_df.lazy()
        .group_by("cluster_id")
        .agg([
            pl.col("cluster_cx").first().alias("cx"),
            pl.col("cluster_cy").first().alias("cy"),
            pl.col("cluster_cz").first().alias("cz"),
            pl.col("E").sum().alias("total_energy"),
            pl.len().alias("num_cells")
        ])
        .collect()
    )
    
    # Calculate Truth Contributors
    top_contributors = _get_cluster_contributors(cells_df, particles, event_idx)
    
    # Join Truth info
    unique_clusters = unique_clusters.join(top_contributors, on="cluster_id", how="left")
    unique_clusters = unique_clusters.with_columns(
        pl.col("contributors_text").fill_null("No truth link")
    )

    # --- 4. Prepare Arrays ---
    c_x = cells_df["x"].to_numpy()
    c_y = cells_df["y"].to_numpy()
    c_z = cells_df["z"].to_numpy()
    c_E = cells_df["E"].to_numpy()
    c_id = cells_df["cluster_id"].to_numpy()
    c_cluster_E = cells_df["cluster_total_energy"].to_numpy()
    
    k_x = unique_clusters["cx"].to_numpy()
    k_y = unique_clusters["cy"].to_numpy()
    k_z = unique_clusters["cz"].to_numpy()
    k_E = unique_clusters["total_energy"].to_numpy()
    k_id = unique_clusters["cluster_id"].to_numpy()
    k_count = unique_clusters["num_cells"].to_numpy()
    k_text = unique_clusters["contributors_text"].to_numpy()

    # --- 5. Scaling ---
    if len(c_E) > 0:
        c_norm = np.clip(c_E, np.percentile(c_E, 5), np.percentile(c_E, 95))
        c_sizes = 2 + 5 * (c_norm - c_norm.min()) / (c_norm.max() - c_norm.min() + 1e-9)
    else:
        c_sizes = 3

    if len(k_E) > 0:
        k_norm = np.clip(k_E, np.percentile(k_E, 5), np.percentile(k_E, 95))
        k_sizes = 15 + 35 * (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min() + 1e-9)
    else:
        k_sizes = 20

    # --- 6. Plotting ---
    trace_cells = go.Scatter3d(
        x=c_x, y=c_y, z=c_z,
        mode='markers',
        name='Cells',
        marker=dict(size=c_sizes, color=c_id, colorscale='Turbo', opacity=1.0, line=dict(width=0)),
        customdata=np.stack((c_id, c_E, c_cluster_E), axis=-1),
        hovertemplate=(
            "<b>Cell</b><br>" +
            "Cluster ID: %{customdata[0]}<br>" +
            "Cell Energy: %{customdata[1]:.4f} GeV<br>" +
            "Cluster Total E: %{customdata[2]:.4f} GeV<br>" +
            "<extra></extra>"
        )
    )

    trace_centers = go.Scatter3d(
        x=k_x, y=k_y, z=k_z,
        mode='markers',
        name='Cluster Centers',
        marker=dict(
            size=k_sizes, color=k_id, colorscale='Turbo', opacity=0.5, symbol='circle',
            line=dict(width=2, color='white')
        ),
        customdata=np.stack((k_id, k_E, k_count, k_text), axis=-1),
        hovertemplate=(
            "<b>CLUSTER CENTER</b><br>" +
            "ID: %{customdata[0]}<br>" +
            "Total Energy: %{customdata[1]:.4f} GeV<br>" +
            "Cells: %{customdata[2]}<br>" +
            "<br><b>Top Contributors:</b><br>" +
            "%{customdata[3]}" +
            "<extra></extra>"
        )
    )

    fig = go.Figure(data=[trace_cells, trace_centers])

    updatemenus = [dict(
        type="buttons", direction="left",
        buttons=[
            dict(label="Show All", method="update", args=[{"visible": [True, True]}]),
            dict(label="Cells Only", method="update", args=[{"visible": [True, False]}]),
            dict(label="Centers Only", method="update", args=[{"visible": [False, True]}]),
        ],
        pad={"r": 10, "t": 10}, showactive=True, x=0.0, xanchor="left", y=1.1, yanchor="top"
    )]
    
    steps = [dict(method="relayout", args=[{"scene.camera.eye": {"x": z, "y": z, "z": z}}], label=f"{z:.1f}") for z in np.linspace(0.1, 2.5, 50)]
    sliders = [dict(active=10, currentvalue={"prefix": "Zoom: "}, pad={"t": 50}, steps=steps)]

    fig.update_layout(
        title=f"Event {event_idx} | {len(k_id)} Clusters",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        updatemenus=updatemenus, sliders=sliders, template="plotly_white", margin=dict(l=0, r=0, b=0, t=80)
    )

    if show:
        fig.show()