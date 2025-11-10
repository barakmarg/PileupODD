import uproot
from tqdm import tqdm
import numpy as np
import awkward as ak

class TreeWriter:
    def __init__(self, path, tree_name, chunk_size):
        self.f = uproot.recreate(path)
        self.tree_name = tree_name

        self.fill_one_event_counter = 0
        self.chunk_size = chunk_size
        self.data = {}


    def reset_chunk(self):
        self.data = {}
        self.fill_one_event_counter = 0     


    def type_64_to_32(self, data):
        if np.issubdtype(data[0].dtype, np.integer):
            return ak.values_astype(ak.Array(data), 'int32')
        elif np.issubdtype(data[0].dtype, np.floating):
            return ak.values_astype(ak.Array(data), 'float32')
        else:
            raise ValueError(f'Unsupported data type: {data[0].dtype}')


    def write(self):
        if self.data == {}:
            return

        for k, v in self.data.items():
            if type(v) == dict:
                self.data[k] = ak.zip(v)
                for subkey, subvalue in v.items():
                    self.data[k][subkey] = self.type_64_to_32(subvalue)
            else:
                self.data[k] = self.type_64_to_32(np.array(v))

        # first time
        if self.tree_name not in self.f:
            self.f[self.tree_name] = self.data
        else:
            self.f[self.tree_name].extend(self.data)



    def fill_one_event(self, data_one_event):

        # create if not exist
        if self.data == {}:
            for k, v in data_one_event.items():
                if type(v) == dict:
                    self.data[k] = {}
                    for kk, vv in v.items():
                        self.data[k][kk] = []
                elif type(v) == np.ndarray: # list
                    self.data[k] = []
                elif type(v) in [int, float]: # scalar
                    self.data[k] = []
                else:
                    raise ValueError(f'Unsupported data type: {type(v)}')

        # fill
        for k, v in data_one_event.items():
            if type(v) == dict:
                for kk, vv in v.items():
                    self.data[k][kk].append(vv)
            else: # same treatment for list and scalar
                self.data[k].append(v)
        self.fill_one_event_counter += 1

        # write if chunk is full
        if self.fill_one_event_counter == self.chunk_size:
            self.write()
            self.reset_chunk()

    
    def close(self):
        self.f.close()

