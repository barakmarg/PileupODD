import torch
import numpy as np
import time
from torch_scatter import scatter_add, scatter_min
from torch_cluster import radius_graph
import torch
import numpy as np
import time
from torch_scatter import scatter_add, scatter_min
from torch_cluster import radius_graph

def run_clue_hybrid(points, energy, batch=None, dc=2.0, rhoc=4.0, dm=4.0, max_num_neighbors=400):
    """
    Robust Hybrid CLUE Algorithm.
    - GPU: Graph building, Rho, Delta.
    - CPU: Cluster ID assignment (Pointer Jumping).
    - Includes fix for IndexKernel assertion errors.
    """
    
    def get_time():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    t_start = get_time()

    # -----------------------------------------------------------
    # PREPARATION
    # -----------------------------------------------------------
    # Handle Batch Dimension (B, N, C) -> (B*N, C)
    is_batched_input = False
    if points.dim() == 3:
        is_batched_input = True
        B, N_per_event, D = points.size()
        
        # Flatten input
        points = points.reshape(-1, D) # (B*N, D)
        if energy.dim() == 2:
            energy = energy.reshape(-1) # (B*N)
            
        # Auto-generate batch vector if not provided
        if batch is None:
            # Create batch index [0, 0, ..., 1, 1, ...] on the same device
            batch = torch.arange(B, device=points.device).repeat_interleave(N_per_event)
            
    if not points.is_cuda and torch.cuda.is_available():
        points = points.cuda()
        energy = energy.cuda()
        if batch is not None:
            batch = batch.cuda()
    
    # Ensure energy is 1D (N,)
    if energy.dim() > 1:
        energy = energy.squeeze()
        
    device = points.device
    N = points.size(0)
    
    # -----------------------------------------------------------
    # STEP 0: MASKING PADDING
    # -----------------------------------------------------------
    # Assume points with negative energy are padding/invalid
    valid_mask = energy >= 0
    if not valid_mask.all():
        # Set energy of invalid points to 0 to be safe for scatter_add
        energy = energy * valid_mask.float()
    
    dc2 = dc * dc
    dm2 = dm * dm
    max_r = max(dc, dm)

    # -----------------------------------------------------------
    # STEP 1: GRAPH BUILDING
    # -----------------------------------------------------------
    # radius_graph returns [2, E]
    edge_index = radius_graph(points, r=max_r, batch=batch, max_num_neighbors=max_num_neighbors, loop=True)
    row, col = edge_index
    
    # Filter edges regarding valid mask
    # We only want edges where both source (row) AND target (col) are valid
    if not valid_mask.all():
        is_valid_edge = valid_mask[row] & valid_mask[col]
        row = row[is_valid_edge]
        col = col[is_valid_edge]
    
    # Compute squared distances
    
    # Compute squared distances
    diff = points[row] - points[col]
    dist_sq = (diff * diff).sum(dim=1)
    
    t_graph = get_time()
    #print(f"[GPU] Graph Build:    {t_graph - t_start:.4f}s | Edges: {row.size(0)}")

    # -----------------------------------------------------------
    # STEP 2: LOCAL DENSITY (RHO)
    # -----------------------------------------------------------
    rho_mask = dist_sq <= dc2
    
    # scatter_add: sum energy of neighbors (col) onto target (row)
    rho = scatter_add(energy[col[rho_mask]], row[rho_mask], dim=0, dim_size=N)

    t_rho = get_time()
    #print(f"[GPU] Rho Calc:       {t_rho - t_graph:.4f}s")

    # -----------------------------------------------------------
    # STEP 3: NEAREST HIGHER DENSITY (DELTA)
    # -----------------------------------------------------------
    rho_row = rho[row]
    rho_col = rho[col]
    
    # Filter: Neighbor must have higher density 
    # OR same density with higher index (tie-breaker)
    is_higher = (rho_col > rho_row) | ((rho_col == rho_row) & (col > row))
    
    # Filter by distance if dm is stricter than the graph radius
    if dm < max_r:
        is_higher &= (dist_sq <= dm2)

    # Create subsets for valid edges
    # We use .contiguous() to ensure safe memory access in kernels
    valid_dists_sq = dist_sq[is_higher].contiguous()
    valid_rows = row[is_higher].contiguous()
    valid_cols = col[is_higher].contiguous()
    
    # Initialize containers
    delta_sq = torch.full((N,), float('inf'), device=device)
    nearest_higher_idx = torch.full((N,), -1, dtype=torch.long, device=device)

    # CRITICAL FIX: handle empty edge case safely
    if valid_dists_sq.size(0) > 0:
        # Find MIN distance to higher density neighbor
        # scatter_min returns (min_values, argmin_indices_into_src)
        # If a point 'i' has no neighbors in valid_rows, argmin[i] = valid_dists_sq.size(0) (Sentinel)
        delta_sq_vals, argmin = scatter_min(valid_dists_sq, valid_rows, dim=0, dim_size=N)
        
        # Determine which points actually found a neighbor
        # STRICT CHECK: argmin must be a valid index in valid_cols
        src_len = valid_dists_sq.size(0)
        has_neighbor = argmin < src_len
        
        # 1. Fill Delta Squared
        delta_sq[has_neighbor] = delta_sq_vals[has_neighbor]
        
        # 2. Retrieve Neighbor Index safely
        # We only index into valid_cols using indices that we confirmed are < src_len
        valid_indices = argmin[has_neighbor]
        nearest_higher_idx[has_neighbor] = valid_cols[valid_indices]
    
    delta = delta_sq.sqrt()
    
    t_delta = get_time()
    #print(f"[GPU] Delta Calc:     {t_delta - t_rho:.4f}s")

    # -----------------------------------------------------------
    # STEP 4: IDENTIFY SEEDS
    # -----------------------------------------------------------
    is_seed = (rho > rhoc) & (delta > dm)
    
    t_seeds = get_time()

    # -----------------------------------------------------------
    # STEP 5: CLUSTER ASSIGNMENT (CPU POINTER JUMPING)
    # -----------------------------------------------------------
    # Move minimal data to CPU
    # .cpu().numpy() implies synchronization
    cpu_is_seed = is_seed.cpu().numpy()
    cpu_nh_idx = nearest_higher_idx.cpu().numpy() 
    
    t_transfer = get_time()
    #print(f"[Trans] GPU -> CPU:   {t_transfer - t_seeds:.4f}s")

    # Initialize Cluster IDs (-1 for noise)
    cluster_ids = np.full(N, -1, dtype=np.int32)
    
    # Assign unique IDs to seeds [0, 1, 2, ...]
    num_seeds = np.sum(cpu_is_seed)
    cluster_ids[cpu_is_seed] = np.arange(num_seeds)
    
    # Build Predecessor Array
    # Points point to themselves by default
    predecessor = np.arange(N)
    # Points with higher density neighbors point to them
    # We strictly only follow valid neighbors (!= -1)
    has_neighbor = cpu_nh_idx != -1
    predecessor[has_neighbor] = cpu_nh_idx[has_neighbor]
    
    # Pointer Jumping (Path Compression)
    # Flattens the dependency tree so everyone points to their Root
    steps = 0
    max_steps = 100 # Safety limit
    
    while steps < max_steps:
        steps += 1
        new_predecessor = predecessor[predecessor]
        
        # Check convergence using numpy array comparison
        if np.array_equal(new_predecessor, predecessor):
            break
        predecessor = new_predecessor
        
    # Final Assignment: Inherit ID from the Root
    # If Root is a Seed, we get a Cluster ID.
    # If Root is a Noise Peak (rho < rhoc), it has ID -1, so we get -1.
    final_cluster_ids = cluster_ids[predecessor]
    

    # Return to original shape if input was (B, N, C)
    if is_batched_input:
        rho = rho.view(B, N_per_event)
        delta = delta.view(B, N_per_event)
        is_seed = is_seed.view(B, N_per_event)
        final_cluster_ids = final_cluster_ids.reshape(B, N_per_event)

    t_end = get_time()
    #print(f"[CPU] Pointer Jump:   {t_end - t_transfer:.4f}s (Converged in {steps} steps)")
    #print(f"-------------------------------------------")
    #print(f"Total Time:           {t_end - t_start:.4f}s")
    #print(f"Found {num_seeds} clusters.")

    return rho, delta, is_seed, final_cluster_ids