import polars as pl
import matplotlib.pyplot as plt
from primary.preprocessing import cluster_purity, particle_purity
import plotly.express as px
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

def plot_clusters_purity(calo_hits_with_clusters:pl.DataFrame, ancestors:pl.DataFrame)->None:
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

def plot_particle_purity(   calo_hits: pl.DataFrame, 
    ancestors: pl.DataFrame, 
    particles: pl.DataFrame)->None:
    purity_df = particle_purity(calo_hits, ancestors, particles)
    # just group by 
    purity_df =(
    purity_df.select([ "purity"]    )
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

def plot_3d_particle_hierarchy(particles: pl.DataFrame, calo_hits: pl.DataFrame, event_idx=0):
    """
    3D Particle Hierarchy Explorer (X, Y, Z).
    - Interactive 3D rotation and zooming.
    - Lineage tracing with generation coloring.
    - Recursive energy calculation.
    """
    
    # --- 1. Data Loading ---
    p_data = particles[event_idx]
    c_data = calo_hits[event_idx]

    # Particle Data
    all_pids = p_data["particle_id"].explode().to_numpy()
    all_pdg_ids = p_data["pdg_id"].explode().to_numpy()
    pid_to_pdg = dict(zip(all_pids, all_pdg_ids))
    
    # Handle Parents (clean NaNs)
    raw_parents = p_data["parent_id"].explode().to_numpy()
    raw_parents = np.nan_to_num(raw_parents, nan=0.0)
    all_parent_ids = raw_parents.astype(np.int64)
    
    # 3D Coordinates
    all_vx = p_data["vx"].explode().to_numpy()
    all_vy = p_data["vy"].explode().to_numpy()
    all_vz = p_data["vz"].explode().to_numpy() # Added Z
    
    # Calo Data
    c_x = c_data["x"].explode().to_numpy()
    c_y = c_data["y"].explode().to_numpy()
    c_z = c_data["z"].explode().to_numpy() # Added Z
    c_contrib_ids = c_data["contrib_particle_ids"].explode().to_numpy()
    c_contrib_enes = c_data["contrib_energies"].explode().to_numpy()
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
