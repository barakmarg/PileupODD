import uproot
import numpy as np
from tqdm import tqdm
import warnings
from cluster.helpers.tree_writer import TreeWriter

from cluster.helpers.meanshift_mod import MeanShiftMod


def deltaR(eta1, phi1, eta2, phi2):
    eta1, phi1, eta2, phi2 = eta1.astype(float), phi1.astype(float), eta2.astype(float), phi2.astype(float)
    d_eta = eta1 - eta2
    phi1, phi2 = (phi1+np.pi) % (2*np.pi) - np.pi, (phi2+np.pi) % (2*np.pi) - np.pi
    d_phi = np.minimum(np.abs(phi1 - phi2), 2*np.pi - np.abs(phi1 - phi2))
    dR = np.sqrt(d_eta**2 + d_phi**2)
    return dR



class MSEventSplitter:

    def __init__(self, input_filepath, tree_name, entry_start, entry_stop, input_br_map_dict, 
            output_filepath, output_br_dict, bandwidth, is_inference, topo_part_frac_e_th, 
            eventNumber_offset=0, chunk_size=100, n_proc=1):
        
        self.input_filepath = input_filepath
        self.tree_name = tree_name
        self.entry_start = entry_start
        self.entry_stop = entry_stop
        self.input_br_map_dict = input_br_map_dict

        self.output_filepath = output_filepath
        self.output_br_dict = output_br_dict
        self.bandwidth = bandwidth
        self.is_inference = is_inference
        self.topo_part_e_th = topo_part_frac_e_th
        self.eventNumber_offset = eventNumber_offset

        self.chunk_size = chunk_size
        self.n_proc = n_proc


    def load_tree(self):
        f = uproot.open(self.input_filepath)
        tree = f[self.tree_name]
        self.entry_stop = min(self.entry_stop or tree.num_entries, tree.num_entries)
        self.n_events = self.entry_stop - self.entry_start
        return tree
    
    def load_branches(self, tree, inp_br_map):
        self.branch_data = {}
        for k, v in tqdm(inp_br_map.items(), desc='Loading branches'):
            self.branch_data[v] = tree[k].array(
                library='np', entry_start=self.entry_start, entry_stop=self.entry_stop)

    def branch_data_update(self):
        pass

    def prep_output_tree(self):
        self.tree_writer_obj = TreeWriter(self.output_filepath, 'EventTree', self.chunk_size)

    def get_track_topo_part_mask_ev(self, ev_i, ms):
        raise NotImplementedError('get_track_topo_mask_ev method must be implemented in the derived class')

    def decorate_mscluster(self, track_mask, cluster_mask, ev_i):
        pass

    def split_events(self):
        tree = self.load_tree()
        self.load_branches(tree, self.input_br_map_dict)
        self.branch_data_update()

        self.prep_output_tree()
        print('output file will be saved to', self.output_filepath)

        if self.bandwidth >= 0:
            # Set up the mean shift clustering
            ms = MeanShiftMod(bandwidth=self.bandwidth, bin_seeding=True)
        else:
            print('bandwidth < 0 ==> events will not be segmented')
            ms = None

        for ev_i in tqdm(range(self.n_events)):

            mask_track_ev, mask_topo_ev, mask_part_ev = self.get_track_topo_part_mask_ev(ev_i)

            n_topo_ev, n_tracks_ev = mask_topo_ev.sum(), mask_track_ev.sum()
            if n_topo_ev + n_tracks_ev == 0:
                continue

            # MeanShift Clustering
            data_c = np.concatenate([self.branch_data['topo_eta'][ev_i][mask_topo_ev].reshape(-1,1), self.branch_data['topo_phi'][ev_i][mask_topo_ev].reshape(-1,1)], axis=1)
            data_t = np.concatenate([self.branch_data['track_eta_int'][ev_i][mask_track_ev].reshape(-1,1), self.branch_data['track_phi_int'][ev_i][mask_track_ev].reshape(-1,1)], axis=1)
            data = np.concatenate([data_c, data_t], axis=0)
            weights = np.concatenate([self.branch_data['topo_e'][ev_i][mask_topo_ev], self.branch_data['track_pt'][ev_i][mask_track_ev]], axis=0)

            # MS Clustering
            if ms is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ms_clustering = ms.fit(data) #, wt=weights)
                labels = ms_clustering.labels_
                num_ms_clusters = len(set(labels))
            else:
                labels = np.zeros(len(data), dtype=int)
                num_ms_clusters = 1


            # segmenting the events based on ms output
            for ms_i in range(num_ms_clusters):

                # length <= unfiltered data
                track_mask_ms = labels[n_topo_ev:] == ms_i
                topo_mask_ms  = labels[:n_topo_ev] == ms_i

                # we want this mask to have teh same length as unfileter data
                track_mask_ev_msi = np.zeros_like(mask_track_ev, dtype=bool)
                track_mask_ev_msi[mask_track_ev] = track_mask_ms

                topo_mask_ev_msi = np.zeros_like(mask_topo_ev, dtype=bool)
                topo_mask_ev_msi[mask_topo_ev] = topo_mask_ms


                self.decorate_mscluster(track_mask_ev_msi, topo_mask_ev_msi, mask_part_ev, ev_i, ms_i)
            

        # write leftover chunk
        self.tree_writer_obj.write()

        # show tree
        self.tree_writer_obj.f[self.tree_writer_obj.tree_name].show(name_width=30, typename_width=12)
        self.tree_writer_obj.close()