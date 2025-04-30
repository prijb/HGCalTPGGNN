# Takes the X, y, w and u from the cache and returns it for a BDT
import numpy as np
import awkward as ak    
import torch
from torch.utils.data import Dataset

# System
import os
import sys

# Plotting
import matplotlib.pyplot as plt

# Aesthetic
from tqdm import tqdm

"""
Dataset format:
* Each row is a cluster
* Each cluster has N (~20) trigger cells which are already ordered by decreasing energy
* Global features added at the end -> Gives N*14 + 3 columns per row
"""
class BDTDataset(Dataset):
    def __init__(self, X, y, w, u, max_tc=20):
        self.max_tc = max_tc
        X_list = []
        
        feature_list = []
        tc_feature_names = X.columns.values
        global_feature_names = u.columns.values
        
        # Make a list of long vector variables
        for var in tc_feature_names:
            for i in range(self.max_tc):
                feature_list.append(f"{var}_{i}")
        for var in global_feature_names:
            feature_list.append(f"{var}")
        #print(f"Long vector feature list: {feature_list}")

        self.tc_feature_names = tc_feature_names
        self.global_feature_names = global_feature_names
        self.feature_list = feature_list

        # Make X, y and w
        self.x = np.full((len(y), len(feature_list)), -999.0)
        self.y = y
        self.W = w

        #print("Number of clusters:", len(y))
        #print("Number of features per cluster:", len(feature_list))
        #print(f"Shape of X: {self.X.shape}")
        #print(f"Shape of y: {self.y.shape}")
        #print(f"Shape of w: {w.shape}")
        #print(f"Shape of u: {u.shape}")

        # Fill the long-vector X data in a columnar way
        tc_multiplicity = X.groupby(level="entry").size()
        """
        var_index = 0
        for i_var in range(len(tc_feature_names)):
            var = tc_feature_names[i_var]   
            for i_tc in range(self.max_tc):
                # Fill the rows of the X array which have more than i_tc TCs
                mask = tc_multiplicity > i_tc
                mask_idx = tc_multiplicity[mask].index
                self.X[mask, var_index] = (X.loc[mask_idx]).xs(i_tc, level="subentry")[var]
                var_index += 1
        # Fill with the global features
        for i_var in range(len(global_feature_names)):
            var = global_feature_names[i_var]
            # Fill the indices of the X array which are chosen by this mask
            self.X[:, var_index] = u[var]
            var_index += 1
        """
        for i_var in tqdm(range(len(feature_list))):
            # Fill tc features
            if i_var < len(tc_feature_names) * self.max_tc:
                var = feature_list[i_var]
                i_tc = int(var.split("_")[-1])
                # The var name is the form var_i, so we keep the first part (join the _'s before the last one)
                var = "_".join(var.split("_")[:-1])
                # Fill the rows of the X array which have more than i_tc TCs
                mask = tc_multiplicity > i_tc
                mask_idx = tc_multiplicity[mask].index
                self.x[mask, i_var] = (X.loc[mask_idx]).xs(i_tc, level="subentry")[var]
            # Fill global features
            else:
                var = feature_list[i_var]
                # Fill the indices of the X array which are chosen by this mask
                self.x[:, i_var] = u[var]

        #print(f"First cluster: {self.x[0]}")       
        #print(len(feature_list))

        # Convert to torch tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y.values, dtype=torch.float32)
        self.W = torch.tensor(self.W.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.W[idx]