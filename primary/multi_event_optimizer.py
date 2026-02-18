import numpy as np
import polars as pl
import optuna
from typing import Callable, Tuple, List
import CLUEstering as clue
from primary.calibration import CALIBRATION
from primary.downsample import voxelize_hits, voxel_config

# ================================================================================
# MULTI-EVENT OPTIMIZER FOR CLUE CLUSTERING (Unified Parameters)
# ================================================================================
# Optimizes unified CLUE parameters (dc, rhoc, dm, ppbin) across multiple events
# with averaged scoring. Runs clustering on all events together.


class MultiEventOptimizer:
    """
    Optimizes CLUE clustering parameters across multiple events using Optuna.
    Averages the score metric across all events for robust parameter tuning.

    Uses unified parameters: dc, rhoc, dm, ppbin (no separate ECAL/HCAL)
    """

    def __init__(self,
                 calo_hits: pl.DataFrame,
                 max_events: int = None,
                 n_trials: int = 100,
                 seed: int = 42,
                 metric_fn: Callable = None,
                 calo_deps_mappings: pl.DataFrame = None,
                 particles_hard_scatter: pl.DataFrame = None,
                 max_clusters_penalty_threshold: int = 10000,):
        """
        Args:
            calo_hits: Polars DataFrame with calorimeter hits
                      (columns: event_id, x, y, z, total_energy, detector, etc.)
            max_events: Maximum number of events to use (None = all)
            n_trials: Number of Optuna trials
            seed: Random seed for reproducibility
            metric_fn: Custom metric function(calo_with_clusters, calo_deps_mappings, particles_hard_scatter) -> float
                      If None, uses default noise ratio metric
            calo_deps_mappings: Polars DataFrame with particle dependencies (required for custom metric_fn)
            particles_hard_scatter: Polars DataFrame with particles (required for custom metric_fn)
        """
        self.calo_hits = calo_hits
        self.n_trials = n_trials
        self.seed = seed
        self.calo_deps_mappings = calo_deps_mappings
        self.particles_hard_scatter = particles_hard_scatter
        self.max_clusters_penalty_threshold = max_clusters_penalty_threshold
        # Use custom metric if provided, otherwise default

        self.metric_fn = metric_fn


        # Prepare data
        self._prepare_data(max_events)

    def _prepare_data(self, max_events: int = None):
        """
        Prepare data: voxelize, calibrate energy, split by event, create point arrays
        """
        # Filter events if needed
        if max_events:
            event_ids = self.calo_hits['event_id'].unique()[:max_events]
            self.calo_hits = self.calo_hits.filter(pl.col('event_id').is_in(event_ids))
            self.particles_hard_scatter = self.particles_hard_scatter.filter(pl.col('event_id').is_in(event_ids))
            self.calo_deps_mappings = self.calo_deps_mappings.filter(pl.col('event_id').is_in(event_ids))
        self.n_events = self.calo_hits['event_id'].n_unique()
        print(f"Preparing data for {self.n_events} events...")

        # Step 1: Voxelize the hits (downsampling)
        print(f"Voxelizing hits...")
        calo_voxel = voxelize_hits(self.calo_hits)
        self.calo_voxel = calo_voxel

        # Step 2: Flatten, explode, and calibrate energy from voxelized hits
        data_flat = (
            calo_voxel.lazy()
            .select(['event_id', 'x', 'y', 'z', 'total_energy', 'detector'])
            .explode(['x', 'y', 'z', 'total_energy', 'detector'])
            .join(CALIBRATION.lazy(), on='detector')
            .with_columns([
                (pl.col('total_energy') * pl.col('calib_factor') * 1000).alias('energy'),
                pl.col('x').cast(pl.Float32),
                pl.col('y').cast(pl.Float32),
                pl.col('z').cast(pl.Float32)
            ])
            .select(['event_id', 'x', 'y', 'z', 'energy', 'detector'])
            .sort('event_id')
            .collect()
        )

        # Store data_flat for later reconstruction
        self.data_flat = data_flat

        # Step 3: Split points by event (vectorized) using voxelized coordinates
        event_counts = data_flat.group_by('event_id', maintain_order=True).len()['len'].to_numpy()
        all_points = data_flat.select(['x', 'y', 'z', 'energy']).to_numpy().astype(np.float32)
        split_indices = np.cumsum(event_counts)[:-1]

        self.points_list = np.split(all_points, split_indices)
        self.event_ids = data_flat['event_id'].to_list()

        # Prepare original hits for reconstruction mapping
        data_flat_original = (
            self.calo_hits.lazy()
            .select(['event_id', 'x', 'y', 'z', 'total_energy', 'detector'])
            .explode(['x', 'y', 'z', 'total_energy', 'detector'])
            .join(CALIBRATION.lazy(), on='detector')
            .with_columns([
                (pl.col('total_energy') * pl.col('calib_factor') * 1000).alias('energy'),
                pl.col('x').cast(pl.Float32),
                pl.col('y').cast(pl.Float32),
                pl.col('z').cast(pl.Float32)
            ])
            .select(['event_id', 'x', 'y', 'z', 'energy', 'detector'])
            .sort('event_id')
            .collect()
        )
        self.data_flat_original = data_flat_original

        total_voxels = all_points.shape[0]
        total_original = data_flat_original.shape[0]
        avg_voxels = total_voxels // self.n_events if self.n_events > 0 else 0
        avg_hits = total_original // self.n_events if self.n_events > 0 else 0
        print(f"✓ Data prepared: {self.n_events} events")
        print(f"  Voxelized: {total_voxels} voxels, avg {avg_voxels} voxels/event")
        print(f"  Original:  {total_original} hits, avg {avg_hits} hits/event")

    def _default_metric(self, calo_with_clusters: pl.DataFrame) -> float:
        """
        Default metric: average noise ratio across all events.

        Args:
            calo_with_clusters: Calo dataframe with cluster assignments

        Returns:
            Average noise ratio (0-1, lower is better)
        """
        raise   NotImplementedError("Default metric is not implemented. Please provide a custom metric function.")

    def _reconstruct_calo_with_clusters(self, cluster_ids_list: List[np.ndarray],
                                         centroids_x_list: List[np.ndarray],
                                         centroids_y_list: List[np.ndarray],
                                         centroids_z_list: List[np.ndarray]) -> pl.DataFrame:
        """
        Reconstruct calo dataframe with cluster assignments by mapping from voxelized space to original hits.
        
        Uses spatial indexing (voxel indices) to assign cluster IDs from voxels to their constituent hits.

        Args:
            cluster_ids_list: List of cluster ID arrays (one per event) for voxels
            centroids_x_list: List of centroid x arrays
            centroids_y_list: List of centroid y arrays
            centroids_z_list: List of centroid z arrays

        Returns:
            Original calo dataframe with cluster columns added, preserving all original columns
        """
        # Concatenate all event results into long 1D arrays
        final_ids = np.concatenate(cluster_ids_list)
        final_cx = np.concatenate(centroids_x_list)
        final_cy = np.concatenate(centroids_y_list)
        final_cz = np.concatenate(centroids_z_list)

        # Add columns to the flat dataframe
        data_clustered = self.data_flat.with_columns([
            pl.Series("cluster_id", final_ids),
            pl.Series("cluster_cx", final_cx),
            pl.Series("cluster_cy", final_cy),
            pl.Series("cluster_cz", final_cz)
        ])

        aggregated = (data_clustered.lazy() 
                              .select('event_id', 'x', 'y', 'z', 'detector', 'cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz')
                              .join(voxel_config.lazy(), on='detector')
                             .with_columns([
                                (pl.col("x") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_x"),
                                (pl.col("y") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_y"),
                                (pl.col("z") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_z"),  ])
                             .join(
                                 (self.calo_hits.lazy().select('event_id', 'x', 'y', 'z', 'detector')
                                 .explode(['x', 'y', 'z', 'detector'])
                                 .join(voxel_config.lazy(), on='detector')
                                .with_columns([
                                    (pl.col("x") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_x"),
                                    (pl.col("y") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_y"),
                                    (pl.col("z") / pl.col("v_size")).floor().cast(pl.Int32).alias("idx_z"),  ])),
                                    on=['event_id', 'idx_x', 'idx_y', 'idx_z', 'detector'], how='inner'
                             )
                             .select(['event_id','cluster_id', 'cluster_cx', 'cluster_cy', 'cluster_cz'])
                             .group_by('event_id', maintain_order=True)
                                 .agg([
                                    pl.col('cluster_id'),
                                    pl.col('cluster_cx'),
                                    pl.col('cluster_cy'),
                                    pl.col('cluster_cz')
                                ])

                              ).collect()

        # Add clusters to calo as columns
        calo_result = self.calo_hits.join(aggregated, on='event_id', how='left')

        return calo_result

    def _run_clustering(self, dc: float, rhoc: float, dm: float, ppbin: int = 16) -> dict:
        """
        Run CLUE clustering on all events with given parameters.

        Args:
            dc: Local density radius
            rhoc: Local density threshold
            dm: Distance to nearest higher density
            ppbin: Points per bin (default: 16)

        Returns:
            Dict with:
                'cluster_ids': List[np.ndarray] - cluster IDs per event
                'n_clusters': List[int] - number of clusters per event
                'centroids_x': List[np.ndarray] - x centroids per event
                'centroids_y': List[np.ndarray] - y centroids per event
                'centroids_z': List[np.ndarray] - z centroids per event
        """
        clusterer = clue.clusterer(dc=dc, rhoc=rhoc, dm=dm, ppbin=ppbin)
        print(f"Running CLUE with dc={dc:.2f}, rhoc={rhoc:.2f}, dm={dm:.2f}, ppbin={ppbin} on {self.n_events} events...")
        cluster_ids_list = []
        n_clusters_list = []
        centroids_x_list = []
        centroids_y_list = []
        centroids_z_list = []

        for points in self.points_list:
            # Run CLUE on this event's points
            clusterer.read_data(points.T)  # Transpose to (features, hits)
            clusterer.run_clue(backend='gpu cuda', verbose=False, block_size=1024)

            cluster_ids = clusterer.cluster_ids
            cluster_ids_list.append(cluster_ids)

            # Count valid clusters (excluding noise with ID -1)
            n_clusters = len(np.unique(cluster_ids[cluster_ids != -1]))
            n_clusters_list.append(n_clusters)

            # Extract centroids
            centroids = clusterer.cluster_centroids()  # shape: (N_clusters, 4)

            # Initialize centroid arrays with inf (for noise)
            n_points = len(cluster_ids)
            cx = np.full(n_points, float('inf'), dtype=np.float32)
            cy = np.full(n_points, float('inf'), dtype=np.float32)
            cz = np.full(n_points, float('inf'), dtype=np.float32)

            # Map centroids to each hit
            valid_mask = cluster_ids != -1
            if np.any(valid_mask):
                valid_ids = cluster_ids[valid_mask]
                cx[valid_mask] = centroids[valid_ids, 0]
                cy[valid_mask] = centroids[valid_ids, 1]
                cz[valid_mask] = centroids[valid_ids, 2]

            centroids_x_list.append(cx)
            centroids_y_list.append(cy)
            centroids_z_list.append(cz)

        return {
            'cluster_ids': cluster_ids_list,
            'n_clusters': n_clusters_list,
            'centroids_x': centroids_x_list,
            'centroids_y': centroids_y_list,
            'centroids_z': centroids_z_list
        }

    def optimize(self) -> Tuple[dict, optuna.Study]:
        """
        Run Optuna optimization across multiple events.

        Returns:
            (best_params_dict, optuna_study)

        best_params_dict contains keys: dc, rhoc, dm, ppbin
        """
        print(f"\nStarting Optuna Optimization with {self.n_trials} trials on {self.n_events} events...")
        print("Parameters: dc, rhoc, dm, ppbin (unified across all events)")

        def objective(trial):
            # Sample parameters using recommended ranges
            dc = trial.suggest_float("dc", 50.0, 150.0)
            rhoc = trial.suggest_float("rhoc", 0.0, 110.0)
            dm = trial.suggest_float("dm", 0.8*dc, dc * 2.0)  # dm >= dc

            try:
                # Run clustering on all events
                results = self._run_clustering(dc=dc, rhoc=rhoc, dm=dm, ppbin=16)
                cluster_ids_list = results['cluster_ids']

                # Reconstruct calo with cluster assignments and centroids
                calo_with_clusters = self._reconstruct_calo_with_clusters(
                    cluster_ids_list,
                    results['centroids_x'],
                    results['centroids_y'],
                    results['centroids_z']
                )

                base_score = self.metric_fn(calo_with_clusters, self.calo_deps_mappings, self.particles_hard_scatter)
                score = base_score
                # Extract cluster statistics
                n_clusters_list = results['n_clusters']
                avg_clusters = np.mean(n_clusters_list)

                # Soft constraints via penalties
                penalty = 0

                # Constraint 1: Don't fragment too much
                max_clusters = np.max(n_clusters_list)
                if max_clusters > self.max_clusters_penalty_threshold:
                    penalty += (max_clusters - self.max_clusters_penalty_threshold) * 0.001


                total_score = score + penalty

                # Store metrics for analysis
                trial.set_user_attr("score", score)
                trial.set_user_attr("base_score_metric", base_score)
                trial.set_user_attr("avg_clusters", avg_clusters)
                trial.set_user_attr("penalty", penalty)
                trial.set_user_attr("min_clusters", float(np.min(n_clusters_list)))
                trial.set_user_attr("max_clusters", float(np.max(n_clusters_list)))

                return total_score

            except Exception as e:
                print(f"Trial {trial.number} failed: {str(e)}")
                raise optuna.TrialPruned()

        # Create and run study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )

        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(objective, n_trials=self.n_trials)

        # Extract best parameters
        bp = study.best_params
        best_params = {
            'dc': bp['dc'],
            'rhoc': bp['rhoc'],
            'dm': bp['dm'],
            'ppbin': 16
        }

        # Print results
        print("\n" + "="*70)
        print("OPTIMIZATION FINISHED")
        print("="*70)
        print(f"Best Objective Value: {study.best_value:.6f}")
        print(f"\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key:10s}: {value}")

        # Run final evaluation on all events
        print("\n" + "="*70)
        print("FINAL EVALUATION (on all events)")
        print("="*70)
        final_results = self._run_clustering(**best_params)
        final_calo = self._reconstruct_calo_with_clusters(
            final_results['cluster_ids'],
            final_results['centroids_x'],
            final_results['centroids_y'],
            final_results['centroids_z']
        )

        # Compute final metrics
        if self.calo_deps_mappings is not None and self.particles_hard_scatter is not None:
            final_score = self.metric_fn(final_calo, self.calo_deps_mappings, self.particles_hard_scatter)
            print(f"Final Base Score Metric: {final_score:.6f}")
        else:
            final_score = self.metric_fn(final_calo)
            final_metrics = [
                np.sum(ids == -1) / len(ids) for ids in final_results['cluster_ids']
            ]
            print(f"Average Noise Ratio: {np.mean(final_metrics):.6f}")
            print(f"Std Dev:             {np.std(final_metrics):.6f}")
            print(f"Min Noise Ratio:     {np.min(final_metrics):.6f}")
            print(f"Max Noise Ratio:     {np.max(final_metrics):.6f}")

        print(f"Avg Clusters/Event:  {np.mean(final_results['n_clusters']):.1f}")
        print(f"Range:               {np.min(final_results['n_clusters'])} - {np.max(final_results['n_clusters'])}")

        return best_params, study


# ================================================================================
# CONVENIENCE FUNCTION FOR EASY EXECUTION
# ================================================================================

def run_multi_event_optimizer(calo_hits: pl.DataFrame,
                               max_events: int = 50,
                               n_trials: int = 100,
                               seed: int = 42,
                               metric_fn: Callable = None,
                               calo_deps_mappings: pl.DataFrame = None,
                               particles_hard_scatter: pl.DataFrame = None) -> Tuple[dict, optuna.Study]:
    """
    Simple entry point to run the multi-event optimizer.

    Usage with custom metric:
        from primary.preprocessing import number_of_clusters_per_particle

        def evaluate(calo, calo_deps_mappings, particles_hard_scatter):
            c = number_of_clusters_per_particle(
                calo_hits_with_clusters=calo,
                ancestors=calo_deps_mappings,
                particles=particles_hard_scatter,
                cut_off_percent=0.05, pt_cut=1.0, eta_cut=3.0
            )
            return len(c.filter(pl.col('num_contributing_clusters') == 0)) / len(c)

        best_params, study = run_multi_event_optimizer(
            calo_hits=calo_hits,
            max_events=100,
            n_trials=200,
            metric_fn=evaluate,
            calo_deps_mappings=calo_deps_mappings,
            particles_hard_scatter=particles_hard_scatter
        )

    Args:
        calo_hits: Polars DataFrame with calorimeter hits
        max_events: Number of events to optimize on
        n_trials: Number of Optuna trials
        seed: Random seed
        metric_fn: Custom metric function (optional)
                   Signature: metric_fn(calo_with_clusters, calo_deps_mappings, particles_hard_scatter) -> float
                            or metric_fn(calo_with_clusters) -> float
        calo_deps_mappings: Polars DataFrame with particle dependencies (for custom metric)
        particles_hard_scatter: Polars DataFrame with particles (for custom metric)

    Returns:
        (best_params_dict, optuna_study)
    """
    optimizer = MultiEventOptimizer(
        calo_hits=calo_hits,
        max_events=max_events,
        n_trials=n_trials,
        seed=seed,
        metric_fn=metric_fn,
        calo_deps_mappings=calo_deps_mappings,
        particles_hard_scatter=particles_hard_scatter
    )

    best_params, study = optimizer.optimize()

    return best_params, study


# ================================================================================
# EXAMPLE USAGE (uncomment to run)
# ================================================================================

if __name__ == "__main__":
    # Example loading and running
    # from huggingface_hub import HfFileSystem
    # import polars as pl
    #
    # fs = HfFileSystem()
    # calo_hits_list = []
    # for i in range(1):
    #     file_path = f"datasets/CERN/ColliderML-Release-1/data/ttbar_pu200_calo_hits/train-{i:05d}-of-01000.parquet"
    #     with fs.open(file_path, "rb") as f:
    #         calo_hits_list.append(pl.read_parquet(f))
    # calo_hits = pl.concat(calo_hits_list)
    #
    # best_ecal, best_hcal, study = run_multi_event_optimizer(
    #     calo_hits,
    #     max_events=50,
    #     n_trials=100
    # )
    pass
