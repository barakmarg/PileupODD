from matplotlib import widgets
import polars as pl
import numpy as np
import plotly.graph_objects as go
from primary.calibration import CALIBRATION
from primary.pdg_mappings import PDG_ID_TO_NAME
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
from IPython.display import display
import plotly.graph_objects as go
import polars as pl
import numpy as np

import plotly.graph_objects as go
import polars as pl
import numpy as np

# -----------------------------------------------------------------------------
# 1. Feature Engineering: Origin & Contributors
# -----------------------------------------------------------------------------
def _get_cluster_info(cells_df: pl.DataFrame, 
                      particles_df: pl.DataFrame, 
                      event_idx: int) -> pl.DataFrame:
    """
    Calculates top contributors AND determines if the cluster is 
    Hard Scatter (>50% energy from vertex_primary==0) or Pileup.
    """
    
    # 1. Get Truth Map
    truth_map = (
        particles_df.lazy()
        .filter(pl.col("event_id") == event_idx)
        .select(["particle_id", "pdg_id", "vertex_primary"])
        .explode(["particle_id", "pdg_id", "vertex_primary"])
    )

    # 2. Explode Cells
    flat_contribs = (
        cells_df.lazy()
        .select(["cluster_id", "contrib_particle_ids", "contrib_energies", "calib_factor"])
        .explode(["contrib_particle_ids", "contrib_energies"])
        .rename({"contrib_particle_ids": "particle_id", "contrib_energies": "raw_energy"})
        .with_columns((pl.col("raw_energy") * pl.col("calib_factor").fill_null(1.0)).alias("calibrated_energy"))
    )

    # JOIN Contributions with Truth
    contribs_with_truth = flat_contribs.join(truth_map, on="particle_id", how="left")

    # 3. Determine Cluster Origin
    cluster_origin = (
        contribs_with_truth
        .group_by("cluster_id")
        .agg([
            pl.col("calibrated_energy").sum().alias("total_E_contribs"),
            pl.col("calibrated_energy").filter(pl.col("vertex_primary") == 1).sum().alias("hs_E")
        ])
        .with_columns(
            (pl.col("hs_E") / pl.col("total_E_contribs")).fill_null(0.0).alias("hs_fraction")
        )
        .with_columns(
            pl.when(pl.col("hs_fraction") > 0.5)
            .then(pl.lit("Hard Scatter"))
            .otherwise(pl.lit("Pileup"))
            .alias("cluster_origin")
        )
    )

    # 4. Get Top Contributors String
    truth_ids = truth_map.collect()
    pdg_map_dict = {
        pid: PDG_ID_TO_NAME.get(str(pid), str(pid)) 
        for pid in truth_ids["pdg_id"].unique().to_list()
    }

    cluster_particle_stats = (
        contribs_with_truth
        .group_by(["cluster_id", "particle_id"])
        .agg([
            pl.col("calibrated_energy").sum().alias("total_particle_E"),
            pl.col("pdg_id").first().alias("pdg_id")
        ])
        .filter(pl.col("total_particle_E") > 0)
    )

    formatted_text = (
        cluster_particle_stats.collect()
        .with_columns(pl.col("pdg_id").replace(pdg_map_dict, default="Unknown").alias("particle_name"))
        .sort("total_particle_E", descending=True)
        .group_by("cluster_id", maintain_order=True)
        .agg([
            pl.format(
                "<b>{}</b> (PDG:{}) ID:{}: <b>{} GeV</b>", 
                pl.col("particle_name"), pl.col("pdg_id"), pl.col("particle_id"), pl.col("total_particle_E").round(4)
            ).slice(0, 5).alias("top_strings")
        ])
        .with_columns(pl.col("top_strings").list.join("<br>").alias("contributors_text"))
        .select(["cluster_id", "contributors_text"])
    )
    
    return formatted_text.join(cluster_origin.collect(), on="cluster_id", how="left")


# -----------------------------------------------------------------------------
# 2. Main Interactive Plotter
# -----------------------------------------------------------------------------
def plot_calo_clusters_interactive(calo_hits: pl.DataFrame, 
                                   particles: pl.DataFrame, 
                                   event_idx: int = 0, 
                                   min_cluster_size: int = 0):
    
    # --- 1. Extract & Calibrate ---
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
        .with_columns([(pl.col('total_energy') * pl.col('calib_factor').fill_null(1.0)).alias('E')])
        .with_columns([pl.col("E").sum().over("cluster_id").alias("cluster_total_energy")])
        .collect()
    )
    
    cells_df = cells_df.with_columns(pl.len().over("cluster_id").alias("cluster_size"))
    if min_cluster_size > 0:
        cells_df = cells_df.filter(pl.col("cluster_size") >= min_cluster_size)
    
    if len(cells_df) == 0:
        print(f"No clusters found for event {event_idx}.")
        return None

    # --- 2. Prepare Unique Clusters & Origin Info ---
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
    
    cluster_info = _get_cluster_info(cells_df, particles, event_idx)
    unique_clusters = unique_clusters.join(cluster_info, on="cluster_id", how="left")

    # --- 3. Split Centers into Hard Scatter vs Pileup ---
    hs_mask = unique_clusters["cluster_origin"] == "Hard Scatter"
    hs_clusters = unique_clusters.filter(hs_mask)
    pu_clusters = unique_clusters.filter(~hs_mask)

    # --- 4. Arrays for Plotly & Callbacks ---
    c_x, c_y, c_z, c_E, c_id = cells_df["x"].to_numpy(), cells_df["y"].to_numpy(), cells_df["z"].to_numpy(), cells_df["E"].to_numpy(), cells_df["cluster_id"].to_numpy()
    
    c_norm = np.clip(c_E, np.percentile(c_E, 5), np.percentile(c_E, 95))
    c_sizes_all = 2 + 5 * (c_norm - c_norm.min()) / (c_norm.max() - c_norm.min() + 1e-9)

    def get_k_sizes(energy_array):
        if len(energy_array) == 0: return np.array([])
        k_norm = np.clip(energy_array, np.percentile(energy_array, 5), np.percentile(energy_array, 95))
        return 15 + 35 * (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min() + 1e-9)

    # --- 5. Build Figure Widget ---
    fig = go.FigureWidget()

    # TRACE 0: Hard Scatter Centers (Red)
    E_hs = hs_clusters["total_energy"].to_numpy()
    trace_hs = go.Scatter3d(
        x=hs_clusters["cx"].to_numpy(), y=hs_clusters["cy"].to_numpy(), z=hs_clusters["cz"].to_numpy(),
        mode='markers', name='Hard Scatter Clusters',
        marker=dict(size=get_k_sizes(E_hs), color='#ff4b4b', opacity=0.8, line=dict(width=2, color='white')),
        customdata=np.stack((hs_clusters["cluster_id"].to_numpy(), E_hs, hs_clusters["hs_fraction"].to_numpy(), hs_clusters["contributors_text"].to_numpy()), axis=-1),
        hovertemplate="<b>HARD SCATTER CLUSTER</b><br>ID: %{customdata[0]}<br>E: %{customdata[1]:.4f} GeV<br>HS Fraction: %{customdata[2]:.1%}<br><br>%{customdata[3]}<extra></extra>"
    )
    fig.add_trace(trace_hs)

    # TRACE 1: Pileup Centers (Grey/Blue)
    E_pu = pu_clusters["total_energy"].to_numpy()
    trace_pu = go.Scatter3d(
        x=pu_clusters["cx"].to_numpy(), y=pu_clusters["cy"].to_numpy(), z=pu_clusters["cz"].to_numpy(),
        mode='markers', name='Pileup Clusters',
        marker=dict(size=get_k_sizes(E_pu), color='#8b9dc3', opacity=0.5, line=dict(width=1, color='white')),
        customdata=np.stack((pu_clusters["cluster_id"].to_numpy(), E_pu, pu_clusters["hs_fraction"].to_numpy(), pu_clusters["contributors_text"].to_numpy()), axis=-1),
        hovertemplate="<b>PILEUP CLUSTER</b><br>ID: %{customdata[0]}<br>E: %{customdata[1]:.4f} GeV<br>HS Fraction: %{customdata[2]:.1%}<br><br>%{customdata[3]}<extra></extra>"
    )
    fig.add_trace(trace_pu)

    # TRACE 2: Selected Cells
    trace_cells = go.Scatter3d(
        x=[], y=[], z=[], mode='markers', name='Selected Cells',
        marker=dict(size=[], color=[], colorscale='Turbo', opacity=1.0, line=dict(width=0)),
        hoverinfo='text'
    )
    fig.add_trace(trace_cells)

    # --- 6. Click Callback Logic ---
    hs_ids = hs_clusters["cluster_id"].to_numpy()
    pu_ids = pu_clusters["cluster_id"].to_numpy()

    def update_point(trace, points, selector):
        if not points.point_inds: return
        idx = points.point_inds[0]
        
        # Identify Cluster ID
        if trace.name == 'Hard Scatter Clusters':
            selected_cluster_id = hs_ids[idx]
        else:
            selected_cluster_id = pu_ids[idx]
        
        # Mask and Count
        mask = (c_id == selected_cluster_id)
        if not np.any(mask): return
        
        # CALCULATE NUMBER OF HITS HERE
        num_hits = np.count_nonzero(mask)

        with fig.batch_update():
            # Update cells trace
            fig.data[2].x = c_x[mask]
            fig.data[2].y = c_y[mask]
            fig.data[2].z = c_z[mask]
            fig.data[2].marker.color = c_id[mask]
            fig.data[2].marker.size = c_sizes_all[mask]
            fig.data[2].hovertext = [f"ID: {cid}<br>E: {e:.4f} GeV" for cid, e in zip(c_id[mask], c_E[mask])]
            
            # Update Title with Hit Count
            cluster_origin = "Hard Scatter" if trace.name == 'Hard Scatter Clusters' else "Pileup"
            fig.layout.title.text = (
                f"Event {event_idx} | Selected {cluster_origin} Cluster ID: {selected_cluster_id} "
                f"| <b>Hits: {num_hits}</b>"
            )

    fig.data[0].on_click(update_point)
    fig.data[1].on_click(update_point)

    # --- 7. Distance Calculator ---
    cluster_centers = unique_clusters.to_dict(as_series=False)
    cluster_ids_set = set(unique_clusters["cluster_id"].to_numpy())

    # Create widgets for distance calculation
    txt_cluster1 = widgets.IntText(value=0, placeholder='Cluster ID 1', description='Cluster 1:', layout=widgets.Layout(width='200px'))
    txt_cluster2 = widgets.IntText(value=0, placeholder='Cluster ID 2', description='Cluster 2:', layout=widgets.Layout(width='200px'))
    btn_calc_distance = widgets.Button(description='Calculate Distance', button_style='info', layout=widgets.Layout(width='150px'))
    result_box = widgets.HTML("<b>Distance Result:</b> Enter cluster IDs and click Calculate")

    def calculate_distance(_):
        cid1 = txt_cluster1.value
        cid2 = txt_cluster2.value

        # Check if cluster IDs are valid
        if cid1 not in cluster_ids_set:
            result_box.value = f"<b style='color:red'>Error: Cluster ID {cid1} not found</b>"
            return
        if cid2 not in cluster_ids_set:
            result_box.value = f"<b style='color:red'>Error: Cluster ID {cid2} not found</b>"
            return

        # Get cluster centers
        idx1 = unique_clusters.filter(pl.col("cluster_id") == cid1).select(["cx", "cy", "cz"])
        idx2 = unique_clusters.filter(pl.col("cluster_id") == cid2).select(["cx", "cy", "cz"])

        if len(idx1) == 0 or len(idx2) == 0:
            result_box.value = "<b style='color:red'>Error: Could not retrieve cluster coordinates</b>"
            return

        x1, y1, z1 = idx1["cx"][0], idx1["cy"][0], idx1["cz"][0]
        x2, y2, z2 = idx2["cx"][0], idx2["cy"][0], idx2["cz"][0]

        # Calculate Euclidean distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        # Display result
        result_box.value = (
            f"<div style='border:2px solid #2ca02c; padding:10px; background-color:#f0f8f0; border-radius:5px;'>"
            f"<b>Distance between Cluster {cid1} and Cluster {cid2}:</b><br>"
            f"<span style='font-size:18px; color:#2ca02c;'><b>{distance:.4f}</b></span> (units)<br>"
            f"<small>Center 1: ({x1:.2f}, {y1:.2f}, {z1:.2f})<br>"
            f"Center 2: ({x2:.2f}, {y2:.2f}, {z2:.2f})</small>"
            f"</div>"
        )

    btn_calc_distance.on_click(calculate_distance)

    distance_row = widgets.HBox([txt_cluster1, txt_cluster2, btn_calc_distance])

    # --- 8. Layout ---
    fig.update_layout(
        title=f"Event {event_idx} | {len(hs_clusters)} Hard Scatter / {len(pu_clusters)} Pileup Clusters",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        template="plotly_white", margin=dict(l=0, r=0, b=0, t=50), height=700,
        legend=dict(x=0, y=1, orientation="v")
    )

    return widgets.VBox([distance_row, result_box, fig])
# -----------------------------------------------------------------------------

import plotly.graph_objs as go
import ipywidgets as widgets
import numpy as np
import polars as pl
from collections import defaultdict, deque

def plot_3d_particle_hierarchy(particles: pl.DataFrame, calo_hits: pl.DataFrame, event_id=0):
    """
    3D Particle Hierarchy Explorer (Corrected).
    - Fixed: Calorimeter hits now visible in global view.
    - Fixed: IndexError crash on rapid clicking (race condition).
    - Updated: Initial Zoom = 20x, Max Zoom = 200x.
    """
    
    # --- 1. Data Loading ---
    p_data = particles.filter(pl.col('event_id') == event_id)
    c_data = calo_hits.filter(pl.col('event_id') == event_id)
    
    # Calibration Handling
    try:
        from primary.calibration import CALIBRATION
        has_calib = True
    except ImportError:
        has_calib = False
        
    if has_calib:
        c_data = (
            c_data.lazy()
            .select(['event_id', 'x', 'y', 'z', 'contrib_energies', 'contrib_particle_ids', 'detector'])
            .explode(['x', 'y', 'z', 'contrib_energies', 'contrib_particle_ids', 'detector'])
            .with_row_index('global_idx')
            .join(CALIBRATION.lazy(), on="detector", how="left")
            .explode(['contrib_energies', 'contrib_particle_ids'])
            .with_columns([
                (pl.col('contrib_energies') * pl.col('calib_factor').fill_null(1.0)).alias('energy'),
                pl.col('contrib_particle_ids').cast(pl.Int64).alias('particle_id')
            ])
            .sort('global_idx')
            .group_by(['event_id','x', 'y', 'z', 'detector'], maintain_order=True)
            .agg([
                pl.col('particle_id').alias('contrib_particle_ids'),
                pl.col('energy').alias('contrib_energies')
            ])
            .group_by(['event_id'], maintain_order=True)  
            .agg('*')
            .collect()
        )
    
    # Particle Data Extraction
    all_pids = p_data["particle_id"].explode().to_numpy()
    all_pdg_ids = p_data["pdg_id"].explode().to_numpy()
    particle_energies = p_data["energy"].explode().to_numpy()

    # Mappings
    pid_to_pdg = dict(zip(all_pids, all_pdg_ids))
    pid_set = set(all_pids) 
    
    # Parents 
    raw_parents = p_data["parent_id"].explode().to_numpy()
    raw_parents = np.nan_to_num(raw_parents, nan=0.0)
    all_parent_ids = raw_parents.astype(np.int64)
    
    # 3D Coordinates
    all_vx = p_data["vx"].explode().to_numpy()
    all_vy = p_data["vy"].explode().to_numpy()
    all_vz = p_data["vz"].explode().to_numpy()
    
    # Momentum Extraction
    def safe_extract(col):
        try:
            return p_data[col].explode().to_numpy()
        except Exception:
            return np.zeros_like(all_vx)
    
    all_px = safe_extract("px")
    all_py = safe_extract("py")
    all_pz = safe_extract("pz")
    all_p_mag = np.sqrt(all_px**2 + all_py**2 + all_pz**2)
    
    # Safe mag for normalization (avoid div by zero)
    safe_mag = np.where(all_p_mag == 0, 1.0, all_p_mag)
    max_p_mag = float(np.max(all_p_mag)) if len(all_p_mag) > 0 else 10.0

    # Calo Data Extraction (preserve per-hit contributor lists)
    if c_data.is_empty():
        c_hits_df = c_data.select(["x", "y", "z", "contrib_particle_ids", "contrib_energies"]) if {
            "x", "y", "z", "contrib_particle_ids", "contrib_energies"
        }.issubset(c_data.columns) else c_data
    else:
        c_hits_df = (
            c_data.lazy()
            .select(["x", "y", "z", "contrib_particle_ids", "contrib_energies"])
            .explode(["x", "y", "z", "contrib_particle_ids", "contrib_energies"])
            .collect()
        )

    c_x = c_hits_df["x"].to_numpy() if "x" in c_hits_df.columns else np.array([])
    c_y = c_hits_df["y"].to_numpy() if "y" in c_hits_df.columns else np.array([])
    c_z = c_hits_df["z"].to_numpy() if "z" in c_hits_df.columns else np.array([])
    c_contrib_ids = c_hits_df["contrib_particle_ids"].to_list() if "contrib_particle_ids" in c_hits_df.columns else []
    c_contrib_enes = c_hits_df["contrib_energies"].to_list() if "contrib_energies" in c_hits_df.columns else []

    # Scene Bounds Calculation
    max_coord = float(max(np.max(np.abs(all_vz)), 1000.0))
    if len(c_z) > 0:
        max_coord = max(max_coord, float(np.max(np.abs(c_z))))
        
    # FIXED ARROW LENGTH (5% of scene)
    ARROW_LEN = max(0.50, max_coord * 0.05)

    # Target Mapping
    target_mask = p_data["is_target_particle"].explode().to_numpy()
    pid_to_is_target = dict(zip(all_pids, target_mask))

    # --- 2. Build Graph ---
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
        if contribs is None or energies is None:
            continue
        if not isinstance(contribs, (list, tuple, np.ndarray)):
            contribs = [contribs]
        if not isinstance(energies, (list, tuple, np.ndarray)):
            energies = [energies]
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

    # --- 3. Visualization Setup ---
    state = {
        'selected_pid': None, 
        'min_energy': 0.0,
        'gen_filter_active': False,
        'gen_low': -2,
        'gen_high': 2,
        'target_filter_active': False,
        'eta_viz_active': False,
        'eta_val': 2.5,
        'mom_viz_active': False,
        'min_mom': 0.0,
        'zoom_level': 20.0  # Initial Zoom
    }

    layout = go.Layout(
        title=f"Event {event_id} 3D Topology",
        width=900, height=750,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode='manual', 
            aspectratio=dict(x=1, y=1, z=1),
            uirevision='constant_view_id'
        ),
        hovermode='closest',
        clickmode='event+select',
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # 0. Calo Hits
    if len(c_x) > 0:
        c_r = np.sqrt(c_x**2 + c_y**2 + c_z**2)
        c_r_safe = np.where(c_r == 0, 1.0, c_r)
        c_phi = np.arctan2(c_y, c_x)
        c_theta = np.arccos(np.clip(c_z / c_r_safe, -1.0, 1.0))
        c_eta = -np.log(np.tan(c_theta / 2.0))
        c_custom = np.stack((c_eta, c_phi), axis=-1)
    else:
        c_custom = np.empty((0, 2))

    trace_calo = go.Scatter3d(
        x=c_x, y=c_y, z=c_z, mode='markers',
        marker=dict(size=3, color='orange', opacity=0.3),
        visible=False, name='Calo Hits',
        customdata=c_custom,
        hovertemplate=(
            "<b>Calo Hit</b><br>" +
            "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>" +
            "eta: %{customdata[0]:.3f}<br>phi: %{customdata[1]:.3f}<br>"
            "<extra></extra>"
        )
    )

    # 1. Link (Hierarchy)
    trace_norm = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='#888', width=3), hoverinfo='skip', name='Link')
    # 2. Data Jump
    trace_jump = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=4, dash='dot'), hoverinfo='skip', name='Data Jump')

    # 3. Particles
    trace_particles = go.Scatter3d(
        x=all_vx, y=all_vy, z=all_vz, mode='markers',
        marker=dict(size=5, color='#ccc'),
        text=[], customdata=all_pids,
        name='Particles', hoverinfo='text'
    )
    
    # 4. Momentum Vectors
    trace_momentum = go.Scatter3d(
        x=[], y=[], z=[],
        mode='lines',
        line=dict(
            width=5, 
            color=[], 
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title='Mom (GeV)', len=0.5, x=0.9)
        ),
        name='Momentum',
        visible=False,
        hoverinfo='skip'
    )
    
    # 5. Eta Cone
    trace_eta = go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], color='cyan', opacity=0.15, name='Eta Cut', visible=False, hoverinfo='skip')

    fig = go.FigureWidget(data=[trace_calo, trace_norm, trace_jump, trace_particles, trace_momentum, trace_eta], layout=layout)
    
    # --- Widgets ---
    info_box = widgets.HTML("<b>Click a particle or search a PID.</b>")
    
    txt_search = widgets.Text(value='', placeholder='PID', description='PID:', layout=widgets.Layout(width='180px'))
    btn_search = widgets.Button(description='Go', button_style='primary', icon='search', layout=widgets.Layout(width='60px'))
    btn_calo = widgets.ToggleButton(description="Show Calo", value=False, icon='cube', layout=widgets.Layout(width='120px'))
    slider_energy = widgets.FloatSlider(value=0, min=0, max=max_e/2, step=0.1, description='Min E:', layout=widgets.Layout(width='250px'))

    btn_target_filter = widgets.ToggleButton(description="Target Only", value=False, icon='bullseye', button_style='info', layout=widgets.Layout(width='120px'))
    btn_gen_filter = widgets.ToggleButton(description="Filter Gen", value=False, icon='filter', layout=widgets.Layout(width='120px'))
    txt_gen_low = widgets.IntText(value=-2, description='Low <', layout=widgets.Layout(width='140px'), disabled=True)
    txt_gen_high = widgets.IntText(value=2, description='< High', layout=widgets.Layout(width='140px'), disabled=True)

    btn_show_eta = widgets.ToggleButton(description="Show Eta", value=False, icon='eye', button_style='warning', layout=widgets.Layout(width='120px'))
    slider_eta = widgets.FloatSlider(value=2.5, min=0.1, max=5.0, step=0.1, description='|Eta|:', layout=widgets.Layout(width='250px'), disabled=True)
    
    btn_show_mom = widgets.ToggleButton(description="Show Mom.", value=False, icon='location-arrow', button_style='success', layout=widgets.Layout(width='120px'))
    slider_mom_filter = widgets.FloatSlider(value=0, min=0, max=max_p_mag, step=0.1, description='Min Mom:', layout=widgets.Layout(width='250px'), disabled=True)

    slider_zoom = widgets.FloatSlider(
        value=20.0,     # Initial Value
        min=1.0, 
        max=200.0,      # Max Value
        step=0.1, 
        description='Zoom:', 
        icon='search-plus', 
        layout=widgets.Layout(width='250px')
    )

    btn_view_xy = widgets.Button(description="XY", icon='arrows-alt', layout=widgets.Layout(width='60px'))
    btn_view_xz = widgets.Button(description="XZ", icon='arrows-alt', layout=widgets.Layout(width='60px'))
    btn_view_yz = widgets.Button(description="YZ", icon='arrows-alt', layout=widgets.Layout(width='60px'))

    # --- Logic ---

    def get_gen_map(center_pid):
        dmap = {center_pid: 0}
        q = deque([(center_pid, 0)])
        visited = {center_pid}
        while q:
            curr, d = q.popleft()
            for child in children_map.get(curr, []):
                if child not in visited:
                    visited.add(child)
                    dmap[child] = d + 1
                    q.append((child, d + 1))
        curr = center_pid
        d = 0
        while True:
            par = parent_map.get(curr)
            if par is None or par in dmap:
                break
            d -= 1
            dmap[par] = d
            curr = par
        return dmap
    
    def generate_cone_mesh(eta, z_limit):
        theta = 2.0 * np.arctan(np.exp(-eta))
        r_max = z_limit * np.tan(theta)
        N = 32
        phi = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = [0.0]
        y = [0.0]
        z = [0.0]
        x.extend(r_max * np.cos(phi))
        y.extend(r_max * np.sin(phi))
        z.extend([z_limit] * N)
        x.extend(r_max * np.cos(phi))
        y.extend(r_max * np.sin(phi))
        z.extend([-z_limit] * N)
        i_l, j_l, k_l = [], [], []
        for m in range(1, N+1):
            next_m = m + 1 if m < N else 1
            i_l.append(0)
            j_l.append(m)
            k_l.append(next_m)
        offset = N
        for m in range(1, N+1):
            next_m = m + 1 if m < N else 1
            i_l.append(0)
            j_l.append(m + offset)
            k_l.append(next_m + offset)
        return x, y, z, i_l, j_l, k_l

    PDG_NAMES_FALLBACK = {'11': 'e-', '-11': 'e+', '22': 'gamma', '13': 'mu-', '-13': 'mu+', '211': 'pi+', '-211': 'pi-'}

    def update_view(msg_override=None):
        sel_pid = state['selected_pid']
        min_e = state['min_energy']
        gen_active = state['gen_filter_active']
        g_low, g_high = state['gen_low'], state['gen_high']
        tgt_active = state['target_filter_active']
        eta_active = state['eta_viz_active']
        eta_val = state['eta_val']
        mom_active = state['mom_viz_active']
        min_mom = state['min_mom']
        zoom = state['zoom_level']

        # A. Filtering
        if sel_pid is None:
            # --- Global View ---
            visible = []
            for p in all_pids:
                if inclusive_energy[p] < min_e - 1e-5:
                    continue
                if tgt_active and not pid_to_is_target.get(p, False):
                    continue
                visible.append(p)
            
            cols, sizes, texts = [], [], []
            for p in visible:
                pdg = str(pid_to_pdg.get(p))
                name = globals().get('PDG_ID_TO_NAME', PDG_NAMES_FALLBACK).get(pdg, pdg)
                if pid_to_is_target.get(p, False):
                    cols.append('#800080')
                    sizes.append(6)
                else:
                    cols.append('#dddddd')
                    sizes.append(4)
                texts.append(f"PID: {p}<br>Name: {name}<br>E: {inclusive_energy[p]:.4f}")
            
            xn, yn, zn = [], [], []
            xj, yj, zj = [], [], []
            title_txt = f"All Particles (> {min_e:.2f} GeV)"
            info_html = f"Showing {len(visible)} particles."
            
        else:
            # --- Hierarchy View ---
            gen_map = get_gen_map(sel_pid)
            visible = []
            for p, gen in gen_map.items():
                if inclusive_energy[p] < min_e - 1e-5:
                    continue
                if gen_active and not (g_low < gen < g_high):
                    continue
                if tgt_active and not pid_to_is_target.get(p, False):
                    continue
                visible.append(p)
            
            display_set = set(visible)
            cols, sizes, texts = [], [], []
            n_anc = sum(1 for v in gen_map.values() if v < 0)
            n_desc = sum(1 for v in gen_map.values() if v > 0)

            for pid in visible:
                gen = gen_map[pid]
                pdg = str(pid_to_pdg.get(pid))
                name = globals().get('PDG_ID_TO_NAME', PDG_NAMES_FALLBACK).get(pdg, pdg)
                is_tgt = pid_to_is_target.get(pid, False)
                tgt_str = " (TARGET)" if is_tgt else ""
                texts.append(f"PID: {pid}{tgt_str}<br>Gen: {gen:+d}<br>E: {inclusive_energy[pid]:.4f}")
                
                if gen == 0:
                    cols.append('#D62728')
                    sizes.append(10)
                elif is_tgt:
                    cols.append('#800080')
                    sizes.append(7)
                elif gen < 0:
                    cols.append('#1F77B4')
                    sizes.append(6)
                else:
                    cols.append('#2CA02C')
                    sizes.append(6)

            xn, yn, zn = [], [], []
            xj, yj, zj = [], [], []
            
            for pid in visible:
                par = parent_map.get(pid)
                if par is not None and par in display_set:
                    p_i, c_i = pid_to_idx[par], pid_to_idx[pid]
                    gap = abs(gen_map[pid] - gen_map[par])
                    coords = ([all_vx[p_i], all_vx[c_i], None], 
                              [all_vy[p_i], all_vy[c_i], None], 
                              [all_vz[p_i], all_vz[c_i], None])
                    if gap == 1:
                        xn.extend(coords[0])
                        yn.extend(coords[1])
                        zn.extend(coords[2])
                    else:
                        xj.extend(coords[0])
                        yj.extend(coords[1])
                        zj.extend(coords[2])

            pdg_sel = str(pid_to_pdg.get(sel_pid))
            name_sel = globals().get('PDG_ID_TO_NAME', PDG_NAMES_FALLBACK).get(pdg_sel, pdg_sel)
            title_txt = f"Hierarchy: PID {sel_pid} ({name_sel})"
            
            sel_i = pid_to_idx.get(sel_pid)
            vx_sel = all_vx[sel_i] if sel_i is not None else float("nan")
            vy_sel = all_vy[sel_i] if sel_i is not None else float("nan")
            vz_sel = all_vz[sel_i] if sel_i is not None else float("nan")

            info_html = f"""
            <div style="border:1px solid #ccc; padding:8px;">
                <h3 style="color:#D62728; margin:0;">PID: {sel_pid}</h3>
                <b>Type:</b> {name_sel} ({pdg_sel})<br>
                Ancestors: {n_anc} | Descendants: {n_desc}<br>
                <b>Total E:</b> {inclusive_energy[sel_pid]:.4f} GeV<br>
                <b>Vertex:</b> ({vx_sel:.2f}, {vy_sel:.2f}, {vz_sel:.2f})
            </div>
            """

        # B. Batch Update
        with fig.batch_update():
            # Apply Zoom via Axis Range
            axis_limit = max_coord / zoom
            fig.layout.scene.xaxis.range = [-axis_limit, axis_limit]
            fig.layout.scene.yaxis.range = [-axis_limit, axis_limit]
            fig.layout.scene.zaxis.range = [-axis_limit, axis_limit]

            idx_list = [pid_to_idx[p] for p in visible]
            
            # 1. Particles
            if idx_list:
                fig.data[3].x = [all_vx[i] for i in idx_list]
                fig.data[3].y = [all_vy[i] for i in idx_list]
                fig.data[3].z = [all_vz[i] for i in idx_list]
                fig.data[3].marker.color = cols
                fig.data[3].marker.size = sizes
                fig.data[3].text = texts
                fig.data[3].customdata = visible
            else:
                fig.data[3].x = []
                fig.data[3].y = []
                fig.data[3].z = []
                fig.data[3].customdata = [] # Safety clear
            
            # 2. Links
            fig.data[1].x = xn
            fig.data[1].y = yn
            fig.data[1].z = zn
            fig.data[2].x = xj
            fig.data[2].y = yj
            fig.data[2].z = zj
            
            # 3. Calo (FIXED LOGIC)
            if sel_pid is None:
                # Global view: Show ALL hits
                fig.data[0].x = c_x
                fig.data[0].y = c_y
                fig.data[0].z = c_z
            elif visible:
                # Hierarchy view: Show LINKED hits
                active_cells = set()
                for p in visible:
                    active_cells.update(pid_to_cells[p])
                fig.data[0].x = [c_x[i] for i in active_cells]
                fig.data[0].y = [c_y[i] for i in active_cells]
                fig.data[0].z = [c_z[i] for i in active_cells]
            else:
                fig.data[0].x = []
                fig.data[0].y = []
                fig.data[0].z = []

            # 4. Momentum Lines (With Filter)
            if mom_active and idx_list:
                # Filter visible particles by Momentum Magnitude
                idx_filtered = [i for i in idx_list if all_p_mag[i] >= min_mom]
                
                if idx_filtered:
                    sub_px = all_px[idx_filtered]
                    sub_py = all_py[idx_filtered]
                    sub_pz = all_pz[idx_filtered]
                    sub_mag = all_p_mag[idx_filtered]
                    sub_safe = safe_mag[idx_filtered]
                    
                    vx = all_vx[idx_filtered]
                    vy = all_vy[idx_filtered]
                    vz = all_vz[idx_filtered]
                    
                    tx = vx + (sub_px / sub_safe) * ARROW_LEN
                    ty = vy + (sub_py / sub_safe) * ARROW_LEN
                    tz = vz + (sub_pz / sub_safe) * ARROW_LEN
                    
                    n_points = len(idx_filtered)
                    
                    combined_x = np.empty(n_points * 3, dtype=object)
                    combined_x[0::3] = vx
                    combined_x[1::3] = tx
                    combined_x[2::3] = None
                    
                    combined_y = np.empty(n_points * 3, dtype=object)
                    combined_y[0::3] = vy
                    combined_y[1::3] = ty
                    combined_y[2::3] = None
                    
                    combined_z = np.empty(n_points * 3, dtype=object)
                    combined_z[0::3] = vz
                    combined_z[1::3] = tz
                    combined_z[2::3] = None
                    
                    combined_c = np.empty(n_points * 3, dtype=np.float64)
                    combined_c[0::3] = sub_mag
                    combined_c[1::3] = sub_mag
                    combined_c[2::3] = sub_mag
                    
                    fig.data[4].x = combined_x
                    fig.data[4].y = combined_y
                    fig.data[4].z = combined_z
                    fig.data[4].line.color = combined_c
                    fig.data[4].visible = True
                else:
                    fig.data[4].visible = False
            else:
                fig.data[4].visible = False

            # 5. Eta Cones
            if eta_active:
                ex, ey, ez, ei, ej, ek = generate_cone_mesh(eta_val, max_coord * 1.1)
                fig.data[5].x = ex
                fig.data[5].y = ey
                fig.data[5].z = ez
                fig.data[5].i = ei
                fig.data[5].j = ej
                fig.data[5].k = ek
                fig.data[5].visible = True
            else:
                fig.data[5].visible = False
            
            fig.layout.title = title_txt
        
        info_box.value = msg_override if msg_override else info_html

    # --- Handlers ---
    def on_click(trace, points, selector):
        if not points.point_inds:
            return
        
        # 1. FIXED: Boundary check to prevent IndexError during rapid updates
        click_idx = points.point_inds[0]
        if trace.customdata is None or click_idx >= len(trace.customdata):
            return

        clicked = trace.customdata[click_idx]
        state['selected_pid'] = None if state['selected_pid'] == clicked else clicked
        update_view()

    def run_search(_):
        val = txt_search.value.strip()
        if not val:
            return
        try:
            target_pid = int(val)
            if target_pid in pid_set:
                state['selected_pid'] = target_pid
                update_view()
            else:
                info_box.value = f"<b style='color:red'>PID {target_pid} not found.</b>"
        except ValueError:
            info_box.value = "<b style='color:red'>Invalid PID.</b>"

    fig.data[3].on_click(on_click)
    slider_energy.observe(lambda c: (state.update({'min_energy': c['new']}), update_view()), names='value')
    btn_calo.observe(lambda c: (fig.data[0].update(visible=c['new']), update_view()), names='value')
    btn_target_filter.observe(lambda c: (state.update({'target_filter_active': c['new']}), update_view()), names='value')
    
    def toggle_gen(change):
        state['gen_filter_active'] = change['new']
        txt_gen_low.disabled = not change['new']
        txt_gen_high.disabled = not change['new']
        update_view()

    btn_gen_filter.observe(toggle_gen, names='value')
    txt_gen_low.observe(lambda c: (state.update({'gen_low': c['new']}), update_view()), names='value')
    txt_gen_high.observe(lambda c: (state.update({'gen_high': c['new']}), update_view()), names='value')

    def toggle_eta(change):
        state['eta_viz_active'] = change['new']
        slider_eta.disabled = not change['new']
        update_view()

    btn_show_eta.observe(toggle_eta, names='value')
    slider_eta.observe(lambda c: (state.update({'eta_val': c['new']}), update_view()), names='value')
    
    def toggle_mom(change):
        state['mom_viz_active'] = change['new']
        slider_mom_filter.disabled = not change['new']
        update_view()

    btn_show_mom.observe(toggle_mom, names='value')
    
    slider_mom_filter.observe(lambda c: (state.update({'min_mom': c['new']}), update_view()), names='value')
    slider_zoom.observe(lambda c: (state.update({'zoom_level': c['new']}), update_view()), names='value')

    btn_search.on_click(run_search)
    txt_search.on_submit(run_search)

    def set_view_xy(_):
        fig.layout.scene.camera = dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0))

    def set_view_xz(_):
        fig.layout.scene.camera = dict(eye=dict(x=0, y=2.5, z=0), up=dict(x=0, y=0, z=1))

    def set_view_yz(_):
        fig.layout.scene.camera = dict(eye=dict(x=2.5, y=0, z=0), up=dict(x=0, y=0, z=1))

    btn_view_xy.on_click(set_view_xy)
    btn_view_xz.on_click(set_view_xz)
    btn_view_yz.on_click(set_view_yz)

    update_view()
    
    row1 = widgets.HBox([txt_search, btn_search, btn_calo, slider_energy])
    row2 = widgets.HBox([btn_target_filter, btn_gen_filter, txt_gen_low, txt_gen_high])
    row3 = widgets.HBox([btn_show_eta, slider_eta, btn_show_mom, slider_mom_filter])
    row4 = widgets.HBox([slider_zoom, btn_view_xy, btn_view_xz, btn_view_yz])
    
    return widgets.VBox([row1, row2, row3, row4, fig, info_box])


def target_vs_truth_particles(particles):
    import polars as pl
    import plotly.graph_objects as go
    from ipywidgets import VBox, HBox, Output
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from primary.pdg_mappings import PDG_ID_TO_NAME
    from primary.preprocessing import backtrack_to_target
    
    # 1. PREPARE DATA
    # ------------------------------------------------
    target_p = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_target_particle'])
        .explode( 'particle_id','is_target_particle')
        .filter(pl.col('is_target_particle') )
        .select(['event_id', 'particle_id'])
    )
    truth_p = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'is_truth_particle'])
        .explode('particle_id','is_truth_particle')
        .filter(pl.col('is_truth_particle') )
        .select(['event_id', 'particle_id'])
    )
    mappings = backtrack_to_target(particles=particles, src_df=target_p, target_df=truth_p).rename({'src_particle_id':'target_particle_id','target_particle_id':'truth_particle_id'})
        
    combined = (
        particles.lazy()
        .select(['event_id', 'particle_id', 'pt', 'eta', 'energy'])
        .explode(['particle_id', 'pt', 'eta', 'energy']) 
        .join(mappings.lazy(),
            left_on=['event_id', 'particle_id'],
                right_on=['event_id', 'target_particle_id']
            , how='inner')
        .group_by(['event_id', 'truth_particle_id'])
        .agg([pl.col('pt').sum().alias('total_target_pt'),
            pl.col('energy').sum().alias('total_target_energy')])
        .join(
            particles.lazy()
            .select(['event_id', 'particle_id', 'pt', 'eta', 'pdg_id', 'energy'])
            .explode(['particle_id', 'pt', 'eta', 'pdg_id', 'energy']) 
            .rename({'pt':'truth_pt', 'particle_id':'truth_particle_id'}),
            on=['event_id', 'truth_particle_id'],
            how='inner'
        )
        .with_columns([
            (pl.col('truth_pt') - pl.col('total_target_pt')).alias('pt_diff'),
            (pl.col('total_target_pt') / pl.col('truth_pt')).alias('pt_ratio'),
            (pl.col('total_target_energy') / pl.col('energy')).alias('energy_ratio')
        ])).collect()
    
    pdg_stats = (
        combined.lazy()
        .group_by("pdg_id")
        .agg([
            pl.col("pt_ratio").mean().alias("avg_pt_ratio"),
            pl.col("energy_ratio").mean().alias("avg_energy_ratio"),
            pl.col("energy").sum().alias("total_energy"),
            pl.len().alias("count")
        ])
        .filter(pl.col("count") >= 50)
        .collect()
    )

    # Prepare Bar Chart Data
    df_bar = pdg_stats.sort("avg_pt_ratio", descending=False).head(40).to_pandas()
    df_bar['name'] = df_bar['pdg_id'].map(lambda x: PDG_ID_TO_NAME.get(str(x), PDG_ID_TO_NAME.get(int(x), "Unknown")))
    df_bar['label'] = df_bar['name'] + " (" + df_bar['pdg_id'].astype(str) + ")"

    df_bar_en = pdg_stats.sort("avg_energy_ratio", descending=False).head(40).to_pandas()
    df_bar_en['name'] = df_bar_en['pdg_id'].map(lambda x: PDG_ID_TO_NAME.get(str(x), PDG_ID_TO_NAME.get(int(x), "Unknown")))
    df_bar_en['label'] = df_bar_en['name'] + " (" + df_bar_en['pdg_id'].astype(str) + ")"

    # 2. CREATE FIGURES
    # ------------------------------------------------

    # -- Main Bar Chart Sum Pt --
    f_bar = go.FigureWidget(
        data=[go.Bar(
            x=df_bar['label'], y=df_bar['avg_pt_ratio'],
            marker=dict(color=df_bar['avg_pt_ratio'], colorscale='Viridis'),
            customdata=np.stack((df_bar['pdg_id'], df_bar['total_energy']), axis=-1),
            hovertemplate="<b>%{x}</b><br>Avg Pt Ratio: %{y:.3f}<br>Count: %{text}<br>Total Energy: %{customdata[1]:.2e}<extra></extra>",
            text=df_bar['count'],
        )],
        layout=go.Layout(title="Top 20 Worst Pt Ratios", xaxis_title="Particle", yaxis_title="Avg Pt Ratio", height=400, margin=dict(l=40, r=40, t=40, b=80))
    )

    # -- Main Bar Chart Energy --
    f_bar_en_widget = go.FigureWidget(
        data=[go.Bar(
            x=df_bar_en['label'], y=df_bar_en['avg_energy_ratio'],
            marker=dict(color=df_bar_en['avg_energy_ratio'], colorscale='Viridis'),
            customdata=np.stack((df_bar_en['pdg_id'], df_bar_en['total_energy']), axis=-1),
            hovertemplate="<b>%{x}</b><br>Avg Energy Ratio: %{y:.3f}<br>Count: %{text}<br>Total Energy: %{customdata[1]:.2e}<extra></extra>",
            text=df_bar_en['count'],
        )],
        layout=go.Layout(title="Top 20 Worst Energy Ratios", xaxis_title="Particle", yaxis_title="Avg Energy Ratio", height=400, margin=dict(l=40, r=40, t=40, b=80))
    )

    # -- Detail Histograms (Distributions) --
    f_pt = go.FigureWidget(
        data=[go.Histogram(x=[], name="Pt", marker_color='#636EFA')],
        layout=go.Layout(title="Pt Distribution", height=300, margin=dict(l=40, r=40, t=40, b=40), yaxis_type='log')
    )
    f_en = go.FigureWidget(
        data=[go.Histogram(x=[], name="Energy", marker_color='#EF553B')],
        layout=go.Layout(title="Energy Distribution", height=300, margin=dict(l=40, r=40, t=40, b=40), yaxis_type='log')
    )

    # -- Detail Histograms (Ratios) -- 
    # CHANGED: These are now FigureWidgets instead of Output widgets
    f_ratio = go.FigureWidget(
        data=[go.Histogram(
            x=[], name="Pt Ratio", marker_color='blue', opacity=0.7,
            xbins=dict(start=0, end=2.05, size=0.05) # Pre-define bins here
        )],
        layout=go.Layout(
            title="Pt Ratio", xaxis_title="Target PT / Truth PT", yaxis_title="Events",
            height=300, margin=dict(l=40, r=40, t=40, b=40)
        )
    )
    f_en_ratio = go.FigureWidget(
        data=[go.Histogram(
            x=[], name="Energy Ratio", marker_color='red', opacity=0.7,
            xbins=dict(start=0, end=2.05, size=0.05)
        )],
        layout=go.Layout(
            title="Energy Ratio", xaxis_title="Target Energy / Truth Energy", yaxis_title="Events",
            height=300, margin=dict(l=40, r=40, t=40, b=40)
        )
    )

    # -- 2D Matplotlib Plots (Still need Output widgets for Matplotlib) --
    f_pt2d = Output()
    f_en2d = Output()

    # 3. DEFINE CLICK CALLBACK
    # ------------------------------------------------
    def update_graphs_generic(selected_pdg, particle_name):
        # Filter Data
        subset = (
            combined.lazy()
            .filter(pl.col("pdg_id") == selected_pdg)
            .select(["truth_pt", "energy", "total_target_pt", "total_target_energy", "pt_ratio", "energy_ratio"])
            .collect()
        )

        sum_truth_energy = subset["energy"].sum()
        sum_target_energy = subset["total_target_energy"].sum()

        # Update Distribution Widgets (Batch animate for smoothness)
        f_pt.layout.title.text = f"Pt Dist: {particle_name}"
        f_en.layout.title.text = f"Energy Dist: {particle_name} | Truth: {sum_truth_energy:.1e}"

        with f_pt.batch_animate():
            f_pt.data[0].x = subset["truth_pt"]
            
        with f_en.batch_animate():
            f_en.data[0].x = subset["energy"]

        # Update Ratio Widgets (Batch animate for smoothness - NO display() calls)
        pt_ratios = subset["pt_ratio"].to_numpy()
        en_ratios = subset["energy_ratio"].to_numpy()
        
        f_ratio.layout.title.text = f"Pt Ratio: {particle_name} (Mean: {np.mean(pt_ratios):.2f})"
        f_en_ratio.layout.title.text = f"Energy Ratio: {particle_name} (Mean: {np.mean(en_ratios):.2f})"

        with f_ratio.batch_animate():
            f_ratio.data[0].x = pt_ratios
            
        with f_en_ratio.batch_animate():
            f_en_ratio.data[0].x = en_ratios
        
        # Update 2D Histograms (Matplotlib requires recreating the plot)
        with f_pt2d:
            f_pt2d.clear_output(wait=True)
            x = subset['truth_pt'].to_numpy()
            y = subset['total_target_pt'].to_numpy()
            max_val = max(np.max(x), np.max(y)) * 1.05
            plt.figure(figsize=(6, 5))
            h = plt.hist2d(x, y, bins=100, range=[[0, max_val], [0, max_val]], 
                            cmap='viridis', norm=LogNorm(), cmin=1)
            plt.colorbar(h[3], label='Count')
            plt.plot([0, max_val], [0, max_val], 'g--', label='y=x')
            plt.xlabel("Truth PT"); plt.ylabel("Target PT")
            plt.title(f"Truth vs Target PT: {particle_name}")
            plt.grid(True, alpha=0.3); plt.show()
        
        with f_en2d:
            f_en2d.clear_output(wait=True)
            x = subset['energy'].to_numpy()
            y = subset['total_target_energy'].to_numpy()
            max_val = max(np.max(x), np.max(y)) * 1.05
            plt.figure(figsize=(6, 5))
            h = plt.hist2d(x, y, bins=100, range=[[0, max_val], [0, max_val]], 
                            cmap='viridis', norm=LogNorm(), cmin=1)
            plt.colorbar(h[3], label='Count')
            plt.plot([0, max_val], [0, max_val], 'g--', label='y=x')
            plt.xlabel("Truth Energy"); plt.ylabel("Target Energy")
            plt.title(f"Truth vs Target Energy: {particle_name}")
            plt.grid(True, alpha=0.3); plt.show()

    def update_graphs_pt(trace, points, selector):
        if not points.point_inds: return
        idx = points.point_inds[0]
        update_graphs_generic(trace.customdata[idx][0], df_bar.iloc[idx]['name'])

    def update_graphs_en(trace, points, selector):
        if not points.point_inds: return
        idx = points.point_inds[0]
        update_graphs_generic(trace.customdata[idx][0], df_bar_en.iloc[idx]['name'])

    f_bar.data[0].on_click(update_graphs_pt)
    f_bar_en_widget.data[0].on_click(update_graphs_en)

    # 4. DISPLAY
    # ------------------------------------------------
    return VBox([
        f_bar,
        f_bar_en_widget,
        HBox([f_ratio, f_en_ratio]), # These are now FigureWidgets
        HBox([f_pt, f_en]),
        HBox([f_pt2d, f_en2d])
    ])