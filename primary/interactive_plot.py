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
# -----------------------------------------------------------------------------


def plot_3d_particle_hierarchy(particles: pl.DataFrame, calo_hits: pl.DataFrame, event_id=0):
    """
    3D Particle Hierarchy Explorer.
    - Updated: Initial Zoom = 20x, Max Zoom = 200x.
    - Fixed: Eta cones do not stretch view.
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
    
    # Particle Data
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
    # Safe mag for normalization
    safe_mag = np.where(all_p_mag == 0, 1.0, all_p_mag)
    max_p_mag = float(np.max(all_p_mag)) if len(all_p_mag) > 0 else 10.0

    # Calo Data Extraction
    c_x = c_data["x"].explode().to_numpy()
    c_y = c_data["y"].explode().to_numpy()
    c_z = c_data["z"].explode().to_numpy()
    c_contrib_ids = c_data["contrib_particle_ids"].explode().to_numpy()
    c_contrib_enes = c_data["contrib_energies"].explode().to_numpy()

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
        if contribs is None:
            continue
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
    # CHANGED: Initial zoom level set to 20.0
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
        'zoom_level': 20.0  # Initial State
    }

    layout = go.Layout(
        title=f"Event {event_id} 3D Topology",
        width=900, height=750,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            # FIXED: 'data' -> 'manual' to prevent reshaping when cones appear
            aspectmode='manual', 
            # FIXED: Forces 1:1:1 geometric scaling regardless of data limits
            aspectratio=dict(x=1, y=1, z=1),
            uirevision='constant_view_id'  # Prevents camera reset
        ),
        hovermode='closest',
        clickmode='event+select',
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=50)
    )

    trace_calo = go.Scatter3d(
        x=c_x, y=c_y, z=c_z, mode='markers',
        marker=dict(size=3, color='orange', opacity=0.3),
        visible=False, name='Calo Hits'
    )

    trace_norm = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='#888', width=3), hoverinfo='skip', name='Link')
    trace_jump = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='red', width=4, dash='dot'), hoverinfo='skip', name='Data Jump')

    trace_particles = go.Scatter3d(
        x=all_vx, y=all_vy, z=all_vz, mode='markers',
        marker=dict(size=5, color='#ccc'),
        text=[], customdata=all_pids,
        name='Particles', hoverinfo='text'
    )
    
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

    # =========================================================
    # CHANGED: Initial=20.0, Max=200.0
    # =========================================================
    slider_zoom = widgets.FloatSlider(
        value=20.0,     # Initial Value
        min=1.0, 
        max=200.0,      # Max Value
        step=0.1, 
        description='Zoom:', 
        icon='search-plus', 
        layout=widgets.Layout(width='250px')
    )

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
            
            info_html = f"""
            <div style="border:1px solid #ccc; padding:8px;">
                <h3 style="color:#D62728; margin:0;">PID: {sel_pid}</h3>
                <b>Type:</b> {name_sel} ({pdg_sel})<br>
                Ancestors: {n_anc} | Descendants: {n_desc}<br>
                <b>Total E:</b> {inclusive_energy[sel_pid]:.4f} GeV
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
            
            # 2. Links
            fig.data[1].x = xn
            fig.data[1].y = yn
            fig.data[1].z = zn
            fig.data[2].x = xj
            fig.data[2].y = yj
            fig.data[2].z = zj
            
            # 3. Calo
            if visible:
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
        clicked = trace.customdata[points.point_inds[0]]
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
    btn_calo.observe(lambda c: fig.data[0].update(visible=c['new']), names='value')
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
    
    # New Sliders Listeners
    slider_mom_filter.observe(lambda c: (state.update({'min_mom': c['new']}), update_view()), names='value')
    slider_zoom.observe(lambda c: (state.update({'zoom_level': c['new']}), update_view()), names='value')

    btn_search.on_click(run_search)
    txt_search.on_submit(run_search)

    update_view()
    
    row1 = widgets.HBox([txt_search, btn_search, btn_calo, slider_energy])
    row2 = widgets.HBox([btn_target_filter, btn_gen_filter, txt_gen_low, txt_gen_high])
    row3 = widgets.HBox([btn_show_eta, slider_eta, btn_show_mom, slider_mom_filter])
    row4 = widgets.HBox([slider_zoom])
    
    return widgets.VBox([row1, row2, row3, row4, fig, info_box])