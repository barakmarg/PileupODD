import torch
import numpy as np
from sklearn.cluster._mean_shift import get_bin_seeds
from sklearn.base import BaseEstimator, ClusterMixin
class MeanShiftGPU(BaseEstimator, ClusterMixin):
    def __init__(self, bandwidth=None, max_iter=100, density_subsample_ratio=0.2, seed_ratio=0.05, device=None):
        """
        Args:
            bandwidth: The kernel bandwidth (radius).
            max_iter: Max iterations for convergence.
            density_subsample_ratio: (0.0 to 1.0). Portion of points used to calculate 
                                     the density field. Lower = Faster, slightly less accurate.
                                     0.2 is usually indistinguishable from 1.0 for blobs.
            seed_ratio: (0.0 to 1.0). Portion of points used as starting seeds.
        """
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.density_subsample_ratio = density_subsample_ratio
        self.seed_ratio = seed_ratio
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_tensor):
        # X_tensor is assumed to be on GPU and Float32
        n_samples = X_tensor.shape[0]

        # ---------------------------------------------------------
        # 1. Setup Seeds and Density Reference
        # ---------------------------------------------------------
        # A. Seeds: Randomly select starting positions
        n_seeds = max(10, int(n_samples * self.seed_ratio))
        # Clamp to avoid selecting more seeds than points
        n_seeds = min(n_seeds, n_samples)
        
        seed_indices = torch.randint(0, n_samples, (n_seeds,), device=self.device)
        seeds = X_tensor[seed_indices]  # Shape: (n_seeds, 3)

        # B. Density Reference (The "Stochastic" part)
        # We compute the weighted mean against this subset, not the full 60k
        n_ref = max(100, int(n_samples * self.density_subsample_ratio))
        n_ref = min(n_ref, n_samples)
        
        ref_indices = torch.randint(0, n_samples, (n_ref,), device=self.device)
        X_ref = X_tensor[ref_indices]   # Shape: (n_ref, 3)

        # Pre-compute X_ref squared norm for the distance trick
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # Transpose for broadcasting: (3, n_ref)
        X_ref_t = X_ref.t()
        X_ref_sq = torch.sum(X_ref**2, dim=1, keepdim=True).t() # (1, n_ref)

        bandwidth_sq = self.bandwidth ** 2
        
        # ---------------------------------------------------------
        # 2. Iterative Update Loop
        # ---------------------------------------------------------
        for i in range(self.max_iter):
            # Seeds Squared Norm: Shape (n_seeds, 1)
            seeds_sq = torch.sum(seeds**2, dim=1, keepdim=True)
            
            # Distance Matrix Calculation (Uses Tensor Cores)
            # Result: (n_seeds, n_ref)
            dist_sq = torch.addmm(seeds_sq + X_ref_sq, seeds, X_ref_t, beta=1, alpha=-2)
            
            # Gaussian Kernel: exp(-0.5 * d^2 / bw^2)
            # We do in-place operations to keep memory low
            weights = torch.exp(dist_sq.div_(-2 * bandwidth_sq)) 
            
            # Normalization factor (Sum of weights)
            normalization = weights.sum(dim=1, keepdim=True)
            normalization.clamp_(min=1e-5) # Safety
            
            # Weighted Average (New Cluster Centers)
            # (Weights @ X_ref) / Sum(Weights)
            new_seeds = torch.mm(weights, X_ref).div_(normalization)
            
            # Convergence Check (every 5 iterations to save overhead)
            if i % 5 == 0:
                # Max movement of any seed
                diff = torch.norm(new_seeds - seeds, dim=1).max()
                if diff < 1e-3 * self.bandwidth:
                    seeds = new_seeds
                    break
            
            seeds = new_seeds

        # ---------------------------------------------------------
        # 3. Merge Duplicate Centers
        # ---------------------------------------------------------
        # Round to merge centers that are extremely close
        # (Round to 10% of bandwidth)
        rounding_factor = self.bandwidth / 10.0
        rounded_seeds = torch.round(seeds / rounding_factor) * rounding_factor
        unique_centers = torch.unique(rounded_seeds, dim=0)
        
        # ---------------------------------------------------------
        # 4. Final Label Assignment (Using ALL points)
        # ---------------------------------------------------------
        # Now we use the full X_tensor to assign labels to the found centers
        self.cluster_centers_ = unique_centers
        
        # Dist(All_Points, Centers)
        # X: (N, 3), Centers: (C, 3)
        c_sq = torch.sum(unique_centers**2, dim=1, keepdim=True).t()
        x_sq = torch.sum(X_tensor**2, dim=1, keepdim=True)
        
        dist_matrix = torch.addmm(x_sq + c_sq, X_tensor, unique_centers.t(), beta=1, alpha=-2)
        
        self.labels_ = torch.argmin(dist_matrix, dim=1)
        return self
