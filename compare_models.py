# This code compares two models on their respective datasets.
import os 
import pickle
import sys

# Data loaders
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import ConcatDataset
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams["figure.figsize"] = (12.5, 10)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Timing
import time
start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache_dataset_dir = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/datasets"
model_dir = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/models"

# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make plotdir if it doesn't exist
os.makedirs(f"{cwd}/plots/training", exist_ok=True)

from utils.preprocess import Preprocessor
from utils.dataset_bdt import BDTDataset
from utils.dataset_graph import GraphDataset

# Load the datasets
dataset_photon_gun_bdt_test = torch.load(os.path.join(cache_dataset_dir, "dataset_photon_gun_bdt_test.pt"))
dataset_photon_gun_gnn_test = torch.load(os.path.join(cache_dataset_dir, "dataset_photon_gun_gnn_test.pt"))
dataset_llp_ctau_1000_bdt_test = torch.load(os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_bdt_test.pt"))
dataset_llp_ctau_1000_gnn_test = torch.load(os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_gnn_test.pt"))

test_dataset_bdt = ConcatDataset([dataset_photon_gun_bdt_test, dataset_llp_ctau_1000_bdt_test])
test_dataset_gnn = ConcatDataset([dataset_photon_gun_gnn_test, dataset_llp_ctau_1000_gnn_test])

# Load the models
from mva.bdt import BDTClassifier
from mva.gnn import SimpleGNN

model_bdt = None
model_gnn = SimpleGNN(in_channels=14, global_features=3).to(device)

print(f"Loading BDT model from {model_dir}/bdt_model.pkl")
with open(f"{model_dir}/bdt_model.pkl", "rb") as f:
    model_bdt = pickle.load(f)

print(f"Loading GNN model from {model_dir}/gnn_model.pth")
model_gnn.load_state_dict(torch.load(f"{model_dir}/gnn_model.pth", weights_only=False, map_location=device))

# Manual regularisation numbers (NOTE: Load from file)
X_mean = torch.tensor([
     1.001371,  0.001673, 12.594896,  0.020278,
    -0.018806,  3.669523,  3.337332,  0.548093,
     2.820588,  0.003166,  0.001938,  0.163391,
    -0.143133,  0.568945
], dtype=torch.float32, device=device)

X_std = torch.tensor([
    0.044679,  0.999999,  5.320169,  5.217934,
    5.213455,  2.101008,  2.104588,  1.311103,
    7.463643,  2.259842,  1.811005, 62.276955,
   62.235554, 340.090455
], dtype=torch.float32, device=device)

# u features: [pt, eta, phi]
u_mean = torch.tensor([
    50.394451,
     0.002212,
    -0.001279
], dtype=torch.float32, device=device)

u_std = torch.tensor([
    26.147342,
     2.260366,
     1.810458
], dtype=torch.float32, device=device)

print(f"Evaluating BDT")
test_loader_bdt = DataLoader(test_dataset_bdt, batch_size=len(test_dataset_bdt), shuffle=True)
X_test, y_test, w_test = next(iter(test_loader_bdt))
X_test = X_test.detach().numpy()
y_test = y_test.detach().numpy()
w_test = w_test.detach().numpy()
y_pred_bdt = model_bdt.predict(X_test)
y_pred_proba_bdt = model_bdt.predict_proba(X_test)

print(f"Evaluating GNN")
test_loader_gnn = DataLoader(test_dataset_gnn, batch_size=32, shuffle=True)
model_gnn.eval()
test_losses = []
all_outs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader_gnn:
        batch = batch.to(device)
        # Regularise
        batch.x = (batch.x - X_mean) / X_std
        batch.u = (batch.u - u_mean) / u_std
        out = model_gnn(batch.x, batch.edge_index, batch.batch, batch.u)
        preds = (out > 0.0).float().cpu().numpy()
        labels = batch.y.long().view(-1).cpu().numpy()

        # Apply sigmod to get probabilities
        out = torch.sigmoid(out)
        all_outs.extend(out.cpu().numpy())
        all_preds.extend(preds)
        all_labels.extend(labels)

# ROC curves
from sklearn.metrics import roc_curve, auc
fpr_bdt, tpr_bdt, thresholds_bdt = roc_curve(y_test, y_pred_proba_bdt[:, 1])
fpr_bdt_pred, tpr_bdt_pred, thresholds_bdt_pred = roc_curve(y_test, y_pred_bdt)
fpr_gnn, tpr_gnn, thresholds_gnn = roc_curve(all_labels, all_outs)
fpr_gnn_pred, tpr_gnn_pred, thresholds_gnn_pred = roc_curve(all_labels, all_preds)

# AUC
roc_auc_bdt = auc(fpr_bdt, tpr_bdt)
roc_auc_gnn = auc(fpr_gnn, tpr_gnn)

# 0.5 threshold
fpr_bdt_05 = fpr_bdt_pred[thresholds_bdt_pred == 1]
tpr_bdt_05 = tpr_bdt_pred[thresholds_bdt_pred == 1]
fpr_gnn_05 = fpr_gnn_pred[thresholds_gnn_pred == 1]
tpr_gnn_05 = tpr_gnn_pred[thresholds_gnn_pred == 1]

fig, ax = plt.subplots()
ax.plot(fpr_bdt, tpr_bdt, label=f'BDT ROC curve (area = {roc_auc_bdt:.4f})', color='tab:blue')
ax.plot(fpr_bdt_05, tpr_bdt_05, 'bo', label='BDT threshold = 0.5')
ax.plot(fpr_gnn, tpr_gnn, label=f'GNN ROC curve (area = {roc_auc_gnn:.4f})', color='tab:orange')
ax.plot(fpr_gnn_05, tpr_gnn_05, 'ro', label='GNN threshold = 0.5')
ax.plot(np.logspace(-5, 0, 100), np.logspace(-5, 0, 100), 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xscale('log')
ax.set_xlim(1e-5, 1)
ax.set_ylim(0.0, None)
ax.legend()
hep.cms.label(data=False, llabel="Private Work", rlabel=r"LLP $c\tau \, = \, 1000$ mm", ax=ax)
plt.savefig(f"{cwd}/plots/training/roc_curve_comparison.png")

# Plot the MVA prediction score for both models
y_test = np.array(y_test).reshape(-1)
print(y_pred_proba_bdt[:, 1])
print(y_test)
y_prob_bdt_photon = y_pred_proba_bdt[:, 1][y_test == 0]
y_prob_bdt_llp = y_pred_proba_bdt[:, 1][y_test == 1]
y_prob_gnn_photon = np.array(all_outs)[np.array(all_labels) == 0]
y_prob_gnn_llp = np.array(all_outs)[np.array(all_labels) == 1]

fig, ax = plt.subplots()
ax.hist(y_prob_bdt_photon, bins=50, range=(0, 1), histtype='step', label='Photon Gun', color='tab:blue')
ax.hist(y_prob_bdt_llp, bins=50, range=(0, 1), histtype='step', label='LLP', color='tab:orange')
ax.set_xlabel('Probability')
ax.set_ylabel('Number of Events')
ax.legend()
hep.cms.label(data=False, llabel="Private Work", rlabel=r"BDT", ax=ax)
plt.savefig(f"{cwd}/plots/training/mva_score_bdt.png")

fig, ax = plt.subplots()
ax.hist(y_prob_gnn_photon, bins=50, range=(0, 1), histtype='step', label='Photon Gun', color='tab:blue')
ax.hist(y_prob_gnn_llp, bins=50, range=(0, 1), histtype='step', label='LLP', color='tab:orange')
ax.set_xlabel('Probability')
ax.set_ylabel('Number of Events')
ax.legend()
hep.cms.label(data=False, llabel="Private Work", rlabel=r"GNN", ax=ax)
plt.savefig(f"{cwd}/plots/training/mva_score_gnn.png")