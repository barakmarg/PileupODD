from datasets import load_dataset, DownloadConfig
import os
import numpy as np


def load_hf_files(datasets=None, custom_cache_path=None, n_rows=1000):
        # Download all builder configs for multiple ColliderML datasets

    n_rows = 1000  # number of rows to download per config
    custom_cache_path = custom_cache_path or "/storage/agrp/barakma/h_faces_cache"
    os.environ["HF_DATASETS_CACHE"] = custom_cache_path           # datasets cache
    os.environ["HF_HUB_CACHE"] = os.path.join(custom_cache_path, "hub")  # hub cache (used by streaming and scripts)
    # List the dataset names you want to pull
    colliderml_datasets = datasets or [
        "OpenDataDetector/ColliderML_ttbar_pu0",
        #"OpenDataDetector/ColliderML_higgs_pu0",
    ]
    # Known configs exposed by the builder (from the error message)
    builder_configs = ["particles", "tracker_hits", "calo_hits", "tracks"]
    dl_cfg = DownloadConfig(
        cache_dir=custom_cache_path,
        local_files_only=True,   # do not hit the network; fail if not cached
        resume_download=True,
    )
    all_datasets = {}
    for ds_name in colliderml_datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        all_datasets[ds_name] = {}
        for cfg in builder_configs:
            print(f"Downloading config: {cfg}")
            # Each call returns a DatasetDict with train (and possibly other splits)
            all_datasets[ds_name][cfg] = load_dataset(
                ds_name,
                cfg,
                cache_dir=custom_cache_path,
                download_config=dl_cfg,
                split={"train": f"train[:{n_rows}]"},

            )
            # Basic info summary
            splits = list(all_datasets[ds_name][cfg].keys())
            print(f"  -> got splits: {splits}")
            for split in splits[:1]:  # just first split for brevity
                print(
                    f"     {cfg}/{split}: {len(all_datasets[ds_name][cfg][split])} rows"
                )
    # convert to numpy
    for ds_name in all_datasets:
        for ds_part in all_datasets[ds_name]:
            all_datasets[ds_name][ds_part]['train'] = all_datasets[ds_name][ds_part]['train'].with_format("numpy")

    return all_datasets


def to_eta(theta):
    """Convert polar angle theta to pseudorapidity eta."""
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = -np.log(np.tan(theta / 2.0))
    return eta

def convert_to_cartesian_eta_phi(data):
    """
    Converts Cartesian coordinates (x, y, z) to pseudorapidity (eta)
    and azimuthal angle (phi).

    This function is optimized for performance using NumPy's vectorized
    operations, making it suitable for large arrays of data.

    Parameters:
    -----------
    data : dict or pandas.DataFrame
        A data structure containing 'x', 'y', and 'z' coordinates as
        array-like objects (e.g., numpy arrays, pandas Series).

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two NumPy arrays: (eta, phi).
        - eta (pseudorapidity) ranges from -inf to +inf.
        - phi (azimuthal angle) ranges from -pi to +pi.
    """
    # Extract coordinate arrays
    x = data['x']
    y = data['y']
    z = data['z']

    # Ensure inputs are NumPy arrays for vectorized operations.
    # This also handles lists, pandas Series, etc., gracefully.
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # --- Calculate Phi (Azimuthal Angle) ---
    # np.arctan2 is the two-argument arctangent which correctly handles
    # all quadrants and returns values in the range [-pi, pi].
    phi = np.arctan2(y, x)

    # --- Calculate Eta (Pseudorapidity) ---
    # This requires the polar angle, theta.
    
    # First, calculate the transverse radius in the x-y plane.
    # Using np.hypot is slightly more numerically stable than np.sqrt(x**2 + y**2)
    rho = np.hypot(x, y)
    
    # Calculate the polar angle, theta.
    theta = np.arctan2(rho, z)
    
    # Now calculate pseudorapidity using its definition: eta = -ln(tan(theta/2)).
    # We suppress runtime warnings for division by zero, which occurs for
    # particles exactly on the beam axis (theta = 0 or pi). NumPy handles
    # these edge cases correctly by returning +/- infinity, which is the
    # correct physical limit for eta.
    eta = to_eta(theta)

    return eta, phi


def test_clustering():
    datasets = load_hf_files(
        datasets=[
            "OpenDataDetector/ColliderML_ttbar_pu0",
            #"OpenDataDetector/ColliderML_higgs_pu0",
        ],
        n_rows=10,
    )
    dataset = datasets["OpenDataDetector/ColliderML_ttbar_pu0"]
    for i in range(len(dataset['tracker_hits']['train'])):
        calo_hits = dataset['calo_hits']['train'][i]

        # perform clustering on tracks
        eta_calo, phi_calo = convert_to_cartesian_eta_phi(calo_hits)
        # get tracks cords
        tracks = dataset['tracks']['train'][i]
        phi_tracks = tracks['phi']
        eta_tracks = to_eta(tracks['theta'])





