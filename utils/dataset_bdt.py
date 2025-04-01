# Takes the X, y, w and u from the cache and returns it for a BDT
import numpy as np
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
        print("Creating data objects")
        for idx in tqdm(range(len(y))):
            nodes = X.xs(idx, level="entry")   

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx], self.u[idx]