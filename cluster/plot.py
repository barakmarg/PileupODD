# ...existing code...
import numpy as np
import plotly.express as px
from cluster.play import convert_to_cartesian_eta_phi, to_eta

# ...existing code...
# 3D interactive calo cluster plot (per event) colored by MeanShift cluster

import numpy as np
import plotly.express as px
from cluster.helpers.meanshift_mod import MeanShiftMod
from cluster.play import to_eta, convert_to_cartesian_eta_phi, get_points_for_clustering



def plot_calo_clusters_3d_given(event_idx, points_clustring_info_list:list, ms: MeanShiftMod, all_datasets, dataset_name="OpenDataDetector/ColliderML_ttbar_pu0", show=True):
    calo = all_datasets[dataset_name]["calo_hits"]["train"].with_format("numpy")
    ev = calo[event_idx]
    labels = ms.labels_
    data = points_clustring_info_list[event_idx]['points']
    mask_calo = points_clustring_info_list[event_idx]['mask_calo']
    e_mask = points_clustring_info_list[event_idx]['e_mask']
    e = np.asarray(ev["total_energy"], dtype=float)

    x = ev["x"][e_mask]
    y = ev["y"][e_mask]
    z = ev["z"][e_mask]
    # Selected per-hit metadata aligned with x,y,z
    eta =  data[:, 0][mask_calo]
    phi =  data[:, 1][mask_calo]
    e_sel = e[e_mask]
    labels = labels[mask_calo]

    # Robust marker sizing based on energy
    if e_sel.size:
        lo = np.percentile(e_sel, 5)
        hi = np.percentile(e_sel, 95)
        sizes = np.clip(e_sel, lo, hi)
    else:
        sizes = np.ones_like(e_sel, dtype=float)

    fig = px.scatter_3d(
        x=x, y=y, z=z,
        color=labels.astype(str),
        size=sizes,
        size_max=14,
        opacity=0.8,
        hover_data={
            "cluster": labels,
            "energy": e_sel,
            "eta": eta,
            "phi": phi
        },
        title=f"Calo hits MeanShift clusters (event {event_idx}, n_clusters={len(np.unique(labels))})"
    )
    fig.update_layout(scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z"
    ))
    if show:
        fig.show()
    return labels, ms.cluster_centers_

def plot_calo_clusters_3d(all_datasets, dataset_name="OpenDataDetector/ColliderML_ttbar_pu0", event_idx=0, bandwidth=0.4, energy_threshold=0.0001, show=True):
    calo = all_datasets[dataset_name]["calo_hits"]["train"].with_format("numpy")
    points_list = get_points_for_clustering(all_datasets, dataset_name=dataset_name, energy_threshold=energy_threshold, until_index=event_idx+1)
    data = points_list[event_idx]['points']

    ms = MeanShiftMod(bandwidth=bandwidth, bin_seeding=True).fit(data)
    return plot_calo_clusters_3d_given(event_idx, points_list, ms, all_datasets, dataset_name=dataset_name, show=show)




# Example usage:
# labels_ev0, centers_ev0 = plot_calo_clusters_3d(all_datasets, event_idx=0, bandwidth=0.4, energy_threshold=0.00005)
# ...existing code...