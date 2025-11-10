import warnings
from collections import defaultdict

from sklearn.cluster import MeanShift # , get_bin_seeds
# from sklearn.cluster._mean_shift import _mean_shift_single_seed
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils import check_array, check_random_state, gen_batches

import numpy as np




def deltaR(etaphi1, etaphi2):
    eta1, phi1 = etaphi1
    eta2, phi2 = etaphi2
    d_eta = eta1 - eta2
    phi1, phi2 = (phi1+np.pi) % (2*np.pi) - np.pi, (phi2+np.pi) % (2*np.pi) - np.pi
    d_phi = np.minimum(np.abs(phi1 - phi2), 2*np.pi - np.abs(phi1 - phi2))
    dR = np.sqrt(d_eta**2 + d_phi**2)
    return dR




# copy paste of sklearn.cluster.estimate_bandwidth
# with the only change being the use of deltaR instead of euclidean distance
def estimate_bandwidth(X, *, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
    X = check_array(X)

    random_state = check_random_state(random_state)
    if n_samples is not None:
        idx = random_state.permutation(X.shape[0])[:n_samples]
        X = X[idx]
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:  # cannot fit NearestNeighbors with n_neighbors = 0
        n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs, metric=deltaR)
    nbrs.fit(X)

    bandwidth = 0.0
    for batch in gen_batches(len(X), 500):
        d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
        bandwidth += np.max(d, axis=1).sum()

    return bandwidth / X.shape[0]




# separate function for each seed's iterative loop
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter, wt=None):
    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()["radius"]
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    completed_iterations = 0
    while True:
        # Find mean of points within bandwidth
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break  # Depending on seeding strategy this condition may occur
        my_old_mean = my_mean  # save the old mean

        # new mean for phi
        # my_mean = np.mean(points_within, axis=0)
        if points_within.shape[1] != 2:
            raise ValueError(
                "MeanShiftMod only supports 2D data (eta, phi), got %dD data."
                % points_within.shape[1]
            )
        eta = points_within[:, 0]
        phi = points_within[:, 1]

        if wt is not None:
            eta_mean = np.sum(eta * wt[i_nbrs]) / np.sum(wt[i_nbrs])
            phi_mean = np.arctan2(np.sum(np.sin(phi) * wt[i_nbrs]), np.sum(np.cos(phi) * wt[i_nbrs]))
        else:
            phi_mean = np.arctan2(np.sin(phi).sum(), np.cos(phi).sum())
            eta_mean = np.mean(eta)

        my_mean = np.array([eta_mean, phi_mean])

        # If converged or at max_iter, adds the cluster
        if (

            # need to use deltaR instead of euclidean distance
            # np.linalg.norm(my_mean - my_old_mean) < stop_thresh
            deltaR(my_mean, my_old_mean) < stop_thresh
            or completed_iterations == max_iter
        ):
            break
        completed_iterations += 1
    return tuple(my_mean), len(points_within), completed_iterations



def get_bin_seeds(X, bin_size=0.1, min_bin_freq=1):
    if bin_size == 0:
        return X

    # Bin points
    bin_sizes = defaultdict(int)
    for point in X:
        binned_point = np.round(point / bin_size)
        bin_sizes[tuple(binned_point)] += 1

    # Select only those bins as seeds which have enough members
    bin_seeds = np.array(
        [point for point, freq in bin_sizes.items() if freq >= min_bin_freq],
        dtype=np.float32,
    )
    if len(bin_seeds) == len(X):
        warnings.warn(
            "Binning data failed with provided bin_size=%f, using data points as seeds."
            % bin_size
        )
        return X
    bin_seeds = bin_seeds * bin_size

    # Update bin seeds to be the geometric center of each bin
    distances = np.zeros((len(bin_seeds), len(X)))
    for i, seed in enumerate(bin_seeds):
        for j, point in enumerate(X):
            distances[i, j] = deltaR(seed, point)
    closest_idx = np.argmin(distances, axis=0)
    for i, seed in enumerate(bin_seeds):
        closest_points = X[closest_idx == i]
        if len(closest_points) > 0:
            bin_seeds[i] = np.mean(closest_points, axis=0)
    return bin_seeds



class MeanShiftMod(MeanShift):
    """Modified Mean shift clustering.

    Changes:
        - Euclidean distance is replaced by deltaR
        - Only modifying the fit method
    """

    def __init__(
        self,
        *,
        bandwidth=None,
        seeds=None,
        bin_seeding=False,
        min_bin_freq=1,
        cluster_all=True,
        n_jobs=None,
        max_iter=300,
    ):
        super().__init__(
        bandwidth = bandwidth,
        seeds = seeds,
        bin_seeding = bin_seeding,
        cluster_all = cluster_all,
        min_bin_freq = min_bin_freq,
        n_jobs = n_jobs,
        max_iter = max_iter
    )
        
    def fit(self, X, y=None, wt=None):
        """Perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        wt : array-like of shape (n_samples,), default=None
            Energy values

        Returns
        -------
        self : object
               Fitted instance.
        """
        self._validate_params()
        if np.isnan(X).any():
            warnings.warn("Input contains NaNs. Replacing them with zeros...")
            X = np.nan_to_num(X)
        X = self._validate_data(X)
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, n_jobs=self.n_jobs)

        seeds = self.seeds
        if seeds is None:
            if self.bin_seeding:
                seeds = get_bin_seeds(X, bandwidth, self.min_bin_freq)
            else:
                seeds = X

        n_samples, n_features = X.shape
        center_intensity_dict = {}

        # We use n_jobs=1 because this will be used in nested calls under
        # parallel calls to _mean_shift_single_seed so there is no need for
        # for further parallelism.
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1, metric=deltaR).fit(X)

        # execute iterations on all seeds in parallel
        all_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_mean_shift_single_seed)(seed, X, nbrs, self.max_iter, wt)
            for seed in seeds
        )
        # copy results in a dictionary
        for i in range(len(seeds)):
            if all_res[i][1]:  # i.e. len(points_within) > 0
                center_intensity_dict[all_res[i][0]] = all_res[i][1]

        self.n_iter_ = max([x[2] for x in all_res])

        if not center_intensity_dict:
            # nothing near seeds
            raise ValueError(
                "No point was within bandwidth=%f of any seed. Try a different seeding"
                " strategy                              or increase the bandwidth."
                % bandwidth
            )

        # POST PROCESSING: remove near duplicate points
        # If the distance between two kernels is less than the bandwidth,
        # then we have to remove one because it is a duplicate. Remove the
        # one with fewer points.

        sorted_by_intensity = sorted(
            center_intensity_dict.items(),
            key=lambda tup: (tup[1], tup[0]),
            reverse=True,
        )
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        unique = np.ones(len(sorted_centers), dtype=bool)
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=self.n_jobs, metric=deltaR).fit(
            sorted_centers
        )
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center], return_distance=False)[
                    0
                ]
                unique[neighbor_idxs] = 0
                unique[i] = 1  # leave the current point as unique
        cluster_centers = sorted_centers[unique]

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs, metric=deltaR).fit(cluster_centers)
        labels = np.zeros(n_samples, dtype=int)
        distances, idxs = nbrs.kneighbors(X)
        if self.cluster_all:
            labels = idxs.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= bandwidth
            labels[bool_selector] = idxs.flatten()[bool_selector]

        self.cluster_centers_, self.labels_ = cluster_centers, labels
        return self
