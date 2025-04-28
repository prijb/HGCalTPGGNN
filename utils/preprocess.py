# Preprocesses data (Gets X, y, w, u form) for generic ML training
# Processing
import yaml
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os
import sys
# Deep learning
import torch
import torch_geometric
from torch_geometric.data import Data
# Storage
import pickle
# Memory management
import psutil
import gc

# Aesthetic
from tqdm import tqdm

def print_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024**2  # Convert to MB
    print(f"{stage} - Memory usage: {mem_usage:.2f} MB")

"""
Returns the following per cluster data structures
X: Trigger cell information
Row 
        Subrow  pt  eta  phi ....
cl3d0   tc0  ..  ..  ..
        tc1  ..  ..  ..
        tc2  ..  ..  ..
cl3d1   tc0  ..  ..  ..
        tc1  ..  ..  ..
        tc2  ..  ..  ..
...

y:  Cluster targets
Row class/label
cl3d0 1
cl3d1 1
...

w: Weights per cluster

u: Global variables per cluster
Row pt eta phi
cl3d0  ..  ..  ..
cl3d1  ..  ..  ..
...
"""
class Preprocessor():
    def __init__(self, file_list, tree_name="l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple", cache_dir="cachedir", use_existing_cache=False, batch_size=10000, class_label=0):
        self.file_list = file_list
        self.tree_name = tree_name
        self.cache_dir = cache_dir
        self.use_existing_cache = use_existing_cache  
        self.batch_size = batch_size
        self.class_label = class_label  

    def __len__(self):
        return len(self.file_list)
    
    # Preprocess and cache each file
    def cache_file(self, input_path, output_path):

        # Open the file
        f = uproot.open(input_path)
        t = f[self.tree_name]

        # Stores X, y, w, u
        data_dict = {}

        var_dict = {
            "gen": ["pt", "eta", "phi", "energy", "charge", "pdgid", "status", "daughters"],
            "genpart": ["pt", "eta", "phi", "energy", "pid", "reachedEE", "ovx", "ovy", "ovz", "dvx", "dvy", "dvz", "mother", "exphi", "exeta"],
            "tc": ["n", "id", "subdet", "zside", "layer", "waferu", "waferv", "cellu", "cellv", "pt", "energy", "eta", "phi", "x", "y", "z", "cluster_id", "multicluster_id", "multicluster_pt"],
            "cl3d": ["n", "id", "pt", "energy", "eta", "phi", "hoe", "bdteg"],
        }

        var_list = []
        for key in var_dict:
            for var in var_dict[key]:
                var_list.append(f"{key}_{var}")

        tc_vars = [var for var in var_list if "tc_" in var]
        cl3d_vars = [var for var in var_list if "cl3d_" in var]

        remove_tc_vars = ["tc_n", "tc_id", "tc_cluster_id", "tc_multicluster_id", "tc_multicluster_pt"]
        remove_cl3d_vars = ["cl3d_n"]
        for var in remove_tc_vars:
            tc_vars.remove(var)
        for var in remove_cl3d_vars:
            cl3d_vars.remove(var)

        # Read the data
        print(f"Reading variables: {var_list}")

        # Print memory usage for debugging
        #print_memory_usage(f"Before loading arrays")

        total_events_file = t.num_entries
        if total_events_file < self.batch_size:
            n_batches = 1
        else:
            n_batches = total_events_file//self.batch_size + 1
        print(f"Splitting file into {n_batches} batches")

        for i_batch in tqdm(range(n_batches)):
            events = t.arrays(var_list, library="ak", entry_start=i_batch*self.batch_size, entry_stop=(i_batch+1)*self.batch_size)
            #events = t.arrays(var_list, library="ak", entry_stop=100)
            total_events = len(events)

            # Filter out events
            mask = events["cl3d_n"] > 0
            events = events[mask]

            tc_multicluster_id = events["tc_multicluster_id"]

            # Zip the variables
            tc_dict = {}
            cl3d_dict = {}
            for var in tc_vars:
                var_key = var.split("_")[1:]
                var_key = "_".join(var_key)
                tc_dict[var_key] = events[var]
            for var in cl3d_vars:
                var_key = var.split("_")[1:]
                var_key = "_".join(var_key)
                cl3d_dict[var_key] = events[var]
            tc = ak.zip(tc_dict)
            cl3d = ak.zip(cl3d_dict)

            # Filter out entries (jagged, use zipped collections)
            cl3d_mask = cl3d.pt > 10
            cl3d = cl3d[cl3d_mask]

            # TO DO: Genmatch clusters (either to gen photon, or BSM decay)

            # Print memory usage for debugging
            #print_memory_usage(f"Arrays loaded and filtered")
            
            # Dataset structure is per cluster
            tcs_per_cl3d = ak.Array([])
            tcs_per_cl3d_globals = ak.Array([])

            for evt in tqdm(range(len(events))):
                # Skip zero cluster events
                if len(cl3d[evt]) == 0:
                    continue
                
                tc_event = tc[evt]
                tc_event_multicluster_id = tc_multicluster_id[evt]
                cl3d_event = cl3d[evt]
                cl3d_ids = cl3d_event.id
                for i_cl3d, cl3d_id in enumerate(cl3d_ids):
                    # Get the TCs associated with the cl3d
                    tc_cl3d_mask = tc_event_multicluster_id == cl3d_id
                    tcs_per_cl3d_event = tc_event[tc_cl3d_mask]
                    # Sort in terms of descending energy
                    energy_order = ak.argsort(tcs_per_cl3d_event.energy, axis=-1, ascending=False)
                    tcs_per_cl3d_event = tcs_per_cl3d_event[energy_order]
                    # Globals
                    cl3d_pt = cl3d_event.pt[i_cl3d]
                    cl3d_eta = cl3d_event.eta[i_cl3d]
                    cl3d_phi = cl3d_event.phi[i_cl3d]
                    tcs_per_cl3d = ak.concatenate([tcs_per_cl3d, [tcs_per_cl3d_event]])
                    tcs_per_cl3d_globals = ak.concatenate([tcs_per_cl3d_globals, [ak.Array([cl3d_pt, cl3d_eta, cl3d_phi])]])

            # Print memory usage for debugging
            #print_memory_usage(f"Event loop done")

            # Class label
            y = np.ones(len(tcs_per_cl3d))*self.class_label
            w = np.ones(len(tcs_per_cl3d))
            
            # Debugging
            #print(f"Total events: {total_events}")
            #print(f"X with shape {len(tcs_per_cl3d)}: {tcs_per_cl3d}")
            #print(f"y with shape {len(y)}: {y}")
            #print(f"u with shape {len(tcs_per_cl3d_globals)}: {tcs_per_cl3d_globals}")

            # Convert to pandas dataframes
            df_X = ak.to_dataframe(tcs_per_cl3d)
            df_y = pd.DataFrame(y)
            df_w = pd.DataFrame(w)
            df_u = pd.DataFrame(tcs_per_cl3d_globals, columns=["pt", "eta", "phi"])

            # Print memory usage for debugging
            #print_memory_usage(f"Dataframes loaded")

            # Store the data
            data_dict["X"] = df_X
            data_dict["y"] = df_y
            data_dict["w"] = df_w
            data_dict["u"] = df_u
            output_path_batch = output_path.replace(".pkl", f"_{i_batch}.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(data_dict, f)

        return None
    
    # Cache all files
    def cache_files(self):
        if self.use_existing_cache:
            print(f"Using existing cache at {self.cache_dir}")
            return None
        else:
            print(f"Caching files to {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
            # Clear cache directory
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            for i, file in enumerate(tqdm(self.file_list)):
                input_path = file
                output_path = os.path.join(self.cache_dir, f"file_{i}.pkl")
                self.cache_file(input_path, output_path)
    
    """
    # Load the concatenated data from the cache
    def get_data_dict(self):
        X = None
        y = None
        w = None
        u = None
        print(f"Loading data from {self.cache_dir}")
        for i, file in enumerate(tqdm(os.listdir(self.cache_dir), total=len(os.listdir(self.cache_dir)))):
            
            cache_file = os.path.join(self.cache_dir, file)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                X_i = data["X"]
                y_i = data["y"]
                w_i = data["w"]
                u_i = data["u"]
                if i==0:
                    X = X_i
                    y = y_i
                    w = w_i
                    u = u_i
                else:
                    X = pd.concat([X, X_i])
                    y = pd.concat([y, y_i])
                    w = pd.concat([w, w_i])
            
        return X, y, w, u
        """
    
    # Load the concatenated data from the cache
    # FIX: Preserve indices
    def get_data_dict(self):
        X_chunks, y_chunks, w_chunks, u_chunks = [], [], [], []
        next_entry_id = 0

        print(f"Loading data from {self.cache_dir}")
        for i, file in enumerate(tqdm(os.listdir(self.cache_dir), total=len(os.listdir(self.cache_dir)))):
            
            cache_file = os.path.join(self.cache_dir, file)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            
            X_i = data["X"]
            y_i = data["y"]
            w_i = data["w"]
            u_i = data["u"]
            n_clusters = y_i.shape[0]

            old_entries = X_i.index.levels[0]
            new_entries = old_entries + next_entry_id
            X_i = X_i.copy()
            X_i.index = X_i.index.set_levels(
                [new_entries, X_i.index.levels[1]],  # [new_entry_ids, same subentry level]
                level=[0, 1]
            )
            y_i = y_i.copy(); y_i.index = y_i.index + next_entry_id
            w_i = w_i.copy(); w_i.index = w_i.index + next_entry_id
            u_i = u_i.copy(); u_i.index = u_i.index + next_entry_id
            next_entry_id += n_clusters
            
            X_chunks.append(X_i)
            y_chunks.append(y_i)
            w_chunks.append(w_i)
            u_chunks.append(u_i)

        X = pd.concat(X_chunks)
        y = pd.concat(y_chunks)
        w = pd.concat(w_chunks)
        u = pd.concat(u_chunks)
        return X, y, w, u


        

        

# Testing
#cwd = os.getcwd()
#preprocessor_test = Preprocessor([f"{cwd}/ntuples/photon_gun.root"], cache_dir=f"{cwd}/cache/photon_gun", use_existing_cache=True)
#preprocessor_test.cache_file(f"{cwd}/ntuples/photon_gun.root", f"{cwd}/cache//photon_gun.pkl")
#preprocessor_test.cache_files()

#X, y, w, u = preprocessor_test.get_data_dict()
#print(f"X with shape {X.shape}: {X}")
#print(f"X pt, eta, phi and energy: {X.loc[0, ['pt', 'eta', 'phi', 'energy']]}")
#print(f"y with shape {y.shape}: {y}")
#print(f"w with shape {w.shape}: {w}")
#print(f"u with shape {u.shape}: {u}")
