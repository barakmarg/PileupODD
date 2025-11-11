import numpy as np
from typing import List

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


def get_points_for_clustering(all_datasets, dataset_name="OpenDataDetector/ColliderML_ttbar_pu0", until_index=1) ->List[np.ndarray]:

    dataset = all_datasets[dataset_name]
    calo_hits = dataset['calo_hits']['train'].with_format('numpy')
    tracks = dataset['tracks']['train'].with_format('numpy')
    lst_points = []
    for i in range(len(dataset['tracker_hits']['train'])):
        # perform clustering on tracks
        e_mask = calo_hits[i]['total_energy'] > 3.0  # energy threshold
        eta_calo, phi_calo = convert_to_cartesian_eta_phi(calo_hits[i])
        # get tracks cords
        phi_tracks = tracks[i]['phi']
        eta_tracks = to_eta(tracks[i]['theta'])
        if i >= until_index:
            break

        # concat eta's and phi's
        all_eta = np.concatenate([eta_calo, eta_tracks])
        all_phi = np.concatenate([phi_calo, phi_tracks])

        eta = np.asarray(all_eta).reshape(-1)
        phi = np.asarray(all_phi ).reshape(-1)
        points = np.column_stack((eta, phi))   # shape (N, 2)
        lst_points.append(points)
    return lst_points
        
    