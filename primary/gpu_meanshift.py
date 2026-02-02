import torch
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class MeanShiftGPU(BaseEstimator, ClusterMixin):
    """
    Implementation of 'GPU-accelerated Faster Mean Shift with euclidean distance metrics'
    (Le You et al., 2021) with OOM protection via batching.
    """
    def __init__(self, bandwidth=None, 
                 n_initial=128, 
                 l_factor=8, 
                 h_factor=32, 
                 gamma=0.9, 
                 max_iter=300, 
                 tol=1e-3, 
                 batch_size=1024,  # Added batch_size to prevent OOM
                 device=None):
        """
        Args:
            bandwidth (float): The kernel radius (h). Required.
            n_initial (int): Initial number of seeds (N).
            batch_size (int): Max number of seeds to process in one GPU op.
        """
        self.bandwidth = bandwidth
        self.n_initial = n_initial
        self.l_factor = l_factor
        self.h_factor = h_factor
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        if self.bandwidth is None:
            raise ValueError("Bandwidth must be provided.")

        # 1. Data Prep
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X.float()
        
        X_tensor = X_tensor.to(self.device)
        n_samples = X_tensor.shape[0]
        
        # Pre-compute X norms: ||a-b||^2 = a^2 + b^2 - 2ab
        X_sq = torch.sum(X_tensor**2, dim=1, keepdim=True).t()

        # 2. Dynamic Seed Loop
        n_seeds = self.n_initial
        
        while True:
            # A. Select Seeds
            if n_seeds >= n_samples:
                seeds = X_tensor
                n_seeds = n_samples
            else:
                seed_indices = torch.randperm(n_samples, device=self.device)[:n_seeds]
                seeds = X_tensor[seed_indices]

            # B. Parallel Mean-Shift (Batched to prevent OOM)
            converged_seeds = self._run_parallel_mean_shift(seeds, X_tensor, X_sq)
            
            # C. Prune Modes
            unique_modes = self._prune_modes(converged_seeds)
            M = unique_modes.shape[0]
            
            # D. Dynamic N Adjustment Logic
            if n_seeds < self.l_factor * M and n_seeds < n_samples:
                n_seeds *= 2
                continue # Redo with more seeds
            
            self.cluster_centers_ = unique_modes
            break

        # 3. Final Labeling
        self.labels_ = self._assign_labels(X_tensor, self.cluster_centers_)
        
        # Move to CPU for sklearn compatibility
        self.cluster_centers_ = self.cluster_centers_.cpu().numpy()
        self.labels_ = self.labels_.cpu().numpy()
        
        return self

    def _run_parallel_mean_shift(self, seeds, X, X_sq):
        """
        Iterative Mean Shift with batching to avoid OOM.
        """
        current_seeds = seeds.clone()
        bandwidth_sq = self.bandwidth ** 2
        
        for i in range(self.max_iter):
            n_total_seeds = current_seeds.shape[0]
            new_seeds_list = []
            
            # Process seeds in batches
            for start_idx in range(0, n_total_seeds, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_total_seeds)
                seed_batch = current_seeds[start_idx:end_idx]
                
                # 1. Distances (Batch vs All X)
                seeds_sq_batch = torch.sum(seed_batch**2, dim=1, keepdim=True)
                
                # Memory efficient addmm
                # Result shape: (batch_size, n_samples)
                dist_sq = torch.addmm(
                    seeds_sq_batch + X_sq, 
                    seed_batch, 
                    X.t(), 
                    beta=1, 
                    alpha=-2
                )
                
                # 2. Flat Kernel
                within_bandwidth = dist_sq <= bandwidth_sq
                
                # 3. Update
                # Convert boolean mask to float for matmul
                mask_float = within_bandwidth.float()
                
                points_sum = torch.mm(mask_float, X)
                points_count = mask_float.sum(dim=1, keepdim=True)
                
                valid_mask = points_count.squeeze() > 0
                points_count = points_count.clamp(min=1.0)
                
                new_batch_pos = points_sum / points_count
                
                # Handle orphans (keep old position)
                if not valid_mask.all():
                    # We need to broadcast the valid_mask correctly
                    # valid_mask is (batch,), new_batch_pos is (batch, D)
                    new_batch_pos[~valid_mask] = seed_batch[~valid_mask]
                
                new_seeds_list.append(new_batch_pos)
            
            # Combine updated batches
            new_positions = torch.cat(new_seeds_list, dim=0)

            # 4. Convergence Check (Paper: Early Stopping)
            shift_dist = torch.norm(new_positions - current_seeds, dim=1)
            
            current_seeds = new_positions
            
            converged_count = (shift_dist < self.tol).sum().item()
            convergence_ratio = converged_count / n_total_seeds
            
            if convergence_ratio > self.gamma:
                break
                
        return current_seeds

    def _prune_modes(self, modes):
        """
        Greedy pruning on CPU to save GPU ops overhead for sequential logic.
        """
        if modes.shape[0] == 0:
            return modes
            
        # Move to CPU for sequential merging loop (faster for small N)
        modes_cpu = modes.cpu().numpy()
        n_modes = modes_cpu.shape[0]
        
        # Calculate distances between modes
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(modes_cpu, squared=True)
        bw_sq = self.bandwidth ** 2
        
        keep = np.ones(n_modes, dtype=bool)
        
        # Greedy suppression
        for i in range(n_modes):
            if keep[i]:
                # Find neighbors
                neighbors = dists[i] <= bw_sq
                neighbors[i] = False # Don't remove self
                keep[neighbors] = False
                
        # Return unique modes on GPU
        return modes[torch.from_numpy(keep).to(self.device)]

    def _assign_labels(self, X, centers):
        if centers.shape[0] == 0:
            return torch.zeros(X.shape[0], dtype=torch.long, device=self.device) - 1
            
        labels_list = []
        c_sq = torch.sum(centers**2, dim=1, keepdim=True).t()
        
        # Batch X to avoid OOM during inference
        inference_batch = self.batch_size * 2
        
        for i in range(0, X.shape[0], inference_batch):
            X_batch = X[i : i+inference_batch]
            x_sq = torch.sum(X_batch**2, dim=1, keepdim=True)
            dist = torch.addmm(x_sq + c_sq, X_batch, centers.t(), beta=1, alpha=-2)
            labels_list.append(torch.argmin(dist, dim=1))
            
        return torch.cat(labels_list)