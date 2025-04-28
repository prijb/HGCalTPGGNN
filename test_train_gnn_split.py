# Train a GNN classifier using a train-validation-test split dataset
import os 
import sys

# Data loaders
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import ConcatDataset
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm
from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Timing
import time
start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Consider just the first 3000 clusters
n_skim = -1
# This is for the "dataset" object
use_cache_dataset = False
cache_dataset_dir = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/datasets"

# Additional imports
# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make plotdir if it doesn't exist
os.makedirs(f"{cwd}/plots/training", exist_ok=True)

from utils.preprocess import Preprocessor
from utils.dataset_graph import GraphDataset

photon_gun_training = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/photon_gun/train"
photon_gun_validation = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/photon_gun/validation"
photon_gun_test = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/photon_gun/test"

llp_ctau_1000_training = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/llp_ctau_1000/train"
llp_ctau_1000_validation = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/llp_ctau_1000/validation"
llp_ctau_1000_test = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/llp_ctau_1000/test"

# Load preprocessed data
preprocessor_photon_gun_training = Preprocessor(os.listdir(photon_gun_training), cache_dir=photon_gun_training, use_existing_cache=True, batch_size=10000, class_label=0)
preprocessor_photon_gun_validation = Preprocessor(os.listdir(photon_gun_validation), cache_dir=photon_gun_validation, use_existing_cache=True, batch_size=10000, class_label=0)
preprocessor_photon_gun_test = Preprocessor(os.listdir(photon_gun_test), cache_dir=photon_gun_test, use_existing_cache=True, batch_size=10000, class_label=0)

preprocessor_llp_ctau_1000_training = Preprocessor(os.listdir(llp_ctau_1000_training), cache_dir=llp_ctau_1000_training, use_existing_cache=True, batch_size=10000, class_label=1)
preprocessor_llp_ctau_1000_validation = Preprocessor(os.listdir(llp_ctau_1000_validation), cache_dir=llp_ctau_1000_validation, use_existing_cache=True, batch_size=10000, class_label=1)
preprocessor_llp_ctau_1000_test = Preprocessor(os.listdir(llp_ctau_1000_test), cache_dir=llp_ctau_1000_test, use_existing_cache=True, batch_size=10000, class_label=1)

if use_cache_dataset == False:
    # Get the data for the train, test and validation sets
    X_photon_gun_training, y_photon_gun_training, w_photon_gun_training, u_photon_gun_training = preprocessor_photon_gun_training.get_data_dict()
    X_photon_gun_validation, y_photon_gun_validation, w_photon_gun_validation, u_photon_gun_validation = preprocessor_photon_gun_validation.get_data_dict()
    X_photon_gun_test, y_photon_gun_test, w_photon_gun_test, u_photon_gun_test = preprocessor_photon_gun_test.get_data_dict()

    X_llp_ctau_1000_training, y_llp_ctau_1000_training, w_llp_ctau_1000_training, u_llp_ctau_1000_training = preprocessor_llp_ctau_1000_training.get_data_dict()
    X_llp_ctau_1000_validation, y_llp_ctau_1000_validation, w_llp_ctau_1000_validation, u_llp_ctau_1000_validation = preprocessor_llp_ctau_1000_validation.get_data_dict()
    X_llp_ctau_1000_test, y_llp_ctau_1000_test, w_llp_ctau_1000_test, u_llp_ctau_1000_test = preprocessor_llp_ctau_1000_test.get_data_dict()

    # Skim datasets
    if (len(y_photon_gun_training) > n_skim) & (n_skim > 0):
        idx = np.arange(n_skim)
        X_photon_gun_training = X_photon_gun_training.loc[idx]
        y_photon_gun_training = y_photon_gun_training.loc[idx]
        w_photon_gun_training = w_photon_gun_training.loc[idx]
        u_photon_gun_training = u_photon_gun_training.loc[idx]
    if (len(y_photon_gun_validation) > n_skim) & (n_skim > 0):
        idx = np.arange(n_skim)
        X_photon_gun_validation = X_photon_gun_validation.loc[idx]
        y_photon_gun_validation = y_photon_gun_validation.loc[idx]
        w_photon_gun_validation = w_photon_gun_validation.loc[idx]
        u_photon_gun_validation = u_photon_gun_validation.loc[idx]
    if (len(y_photon_gun_test) > n_skim) & (n_skim > 0):
        idx = np.arange(n_skim)
        X_photon_gun_test = X_photon_gun_test.loc[idx]
        y_photon_gun_test = y_photon_gun_test.loc[idx]
        w_photon_gun_test = w_photon_gun_test.loc[idx]
        u_photon_gun_test = u_photon_gun_test.loc[idx]
    if (len(y_llp_ctau_1000_training) > n_skim) & (n_skim > 0):
        idx = np.arange(n_skim)
        X_llp_ctau_1000_training = X_llp_ctau_1000_training.loc[idx]
        y_llp_ctau_1000_training = y_llp_ctau_1000_training.loc[idx]
        w_llp_ctau_1000_training = w_llp_ctau_1000_training.loc[idx]
        u_llp_ctau_1000_training = u_llp_ctau_1000_training.loc[idx]
    if (len(y_llp_ctau_1000_validation) > n_skim) & (n_skim > 0):
        idx = np.arange(n_skim)
        X_llp_ctau_1000_validation = X_llp_ctau_1000_validation.loc[idx]
        y_llp_ctau_1000_validation = y_llp_ctau_1000_validation.loc[idx]
        w_llp_ctau_1000_validation = w_llp_ctau_1000_validation.loc[idx]
        u_llp_ctau_1000_validation = u_llp_ctau_1000_validation.loc[idx]
    if (len(y_llp_ctau_1000_test) > n_skim) & (n_skim > 0):
        idx = np.arange(n_skim)
        X_llp_ctau_1000_test = X_llp_ctau_1000_test.loc[idx]
        y_llp_ctau_1000_test = y_llp_ctau_1000_test.loc[idx]
        w_llp_ctau_1000_test = w_llp_ctau_1000_test.loc[idx]
        u_llp_ctau_1000_test = u_llp_ctau_1000_test.loc[idx]


    # Scale up the LLP weights
    # Get the weights for the train, test and validation sets
    w_sum_photon_gun_training = w_photon_gun_training.sum()
    w_sum_photon_gun_validation = w_photon_gun_validation.sum()
    w_sum_photon_gun_test = w_photon_gun_test.sum()

    w_sum_llp_ctau_1000_training = w_llp_ctau_1000_training.sum()
    w_sum_llp_ctau_1000_validation = w_llp_ctau_1000_validation.sum()
    w_sum_llp_ctau_1000_test = w_llp_ctau_1000_test.sum()

    # All the pos weights are similar (~2.0-2.1)
    pos_weight_training = (w_sum_photon_gun_training / w_sum_llp_ctau_1000_training).values[0]
    pos_weight_validation = (w_sum_photon_gun_validation / w_sum_llp_ctau_1000_validation).values[0]
    pos_weight_test = (w_sum_photon_gun_test / w_sum_llp_ctau_1000_test).values[0]
    pos_weight_total = ((w_sum_photon_gun_training + w_sum_photon_gun_validation + w_sum_photon_gun_test) / (w_sum_llp_ctau_1000_training + w_sum_llp_ctau_1000_validation + w_sum_llp_ctau_1000_test)).values[0]

    # Create the datasets
    dataset_photon_gun_training = GraphDataset(X_photon_gun_training, y_photon_gun_training, w_photon_gun_training, u_photon_gun_training, use_knn=True, k=5)
    dataset_photon_gun_validation = GraphDataset(X_photon_gun_validation, y_photon_gun_validation, w_photon_gun_validation, u_photon_gun_validation, use_knn=True, k=5)
    dataset_photon_gun_test = GraphDataset(X_photon_gun_test, y_photon_gun_test, w_photon_gun_test, u_photon_gun_test, use_knn=True, k=5)
    dataset_llp_ctau_1000_training = GraphDataset(X_llp_ctau_1000_training, y_llp_ctau_1000_training, w_llp_ctau_1000_training, u_llp_ctau_1000_training, use_knn=True, k=5)
    dataset_llp_ctau_1000_validation = GraphDataset(X_llp_ctau_1000_validation, y_llp_ctau_1000_validation, w_llp_ctau_1000_validation, u_llp_ctau_1000_validation, use_knn=True, k=5)
    dataset_llp_ctau_1000_test = GraphDataset(X_llp_ctau_1000_test, y_llp_ctau_1000_test, w_llp_ctau_1000_test, u_llp_ctau_1000_test, use_knn=True, k=5)
    print(f"\nGenerated datasets from scratch and saving to cache at {cache_dataset_dir}")
    torch.save(dataset_photon_gun_training, os.path.join(cache_dataset_dir, "dataset_photon_gun_gnn_training.pt"))
    torch.save(dataset_photon_gun_validation, os.path.join(cache_dataset_dir, "dataset_photon_gun_gnn_validation.pt"))
    torch.save(dataset_photon_gun_test, os.path.join(cache_dataset_dir, "dataset_photon_gun_gnn_test.pt"))
    torch.save(dataset_llp_ctau_1000_training, os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_gnn_training.pt"))
else:
    print(f"\nLoading datasets from cache at {cache_dataset_dir}")
    dataset_photon_gun_training = torch.load(os.path.join(cache_dataset_dir, "dataset_photon_gun_gnn_training.pt"))
    dataset_photon_gun_validation = torch.load(os.path.join(cache_dataset_dir, "dataset_photon_gun_gnn_validation.pt"))
    dataset_photon_gun_test = torch.load(os.path.join(cache_dataset_dir, "dataset_photon_gun_gnn_test.pt"))
    dataset_llp_ctau_1000_training = torch.load(os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_gnn_training.pt"))

# Scale up the LLP weights
w_sum_photon_gun_training = dataset_photon_gun_training.W.sum()
w_sum_llp_ctau_1000_training = dataset_llp_ctau_1000_training.W.sum()
print(f"Photon gun training weights: {w_sum_photon_gun_training}")
print(f"LLP ctau 1000 training weights: {w_sum_llp_ctau_1000_training}")
pos_weight_training = (w_sum_photon_gun_training / w_sum_llp_ctau_1000_training).item()

# Concatenate the datasets
train_dataset = ConcatDataset([dataset_photon_gun_training, dataset_llp_ctau_1000_training])
valid_dataset = ConcatDataset([dataset_photon_gun_validation, dataset_llp_ctau_1000_validation])
test_dataset = ConcatDataset([dataset_photon_gun_test, dataset_llp_ctau_1000_test])

print(f"Training on {len(train_dataset)} clusters of which {len(dataset_photon_gun_training)} are photon gun and {len(dataset_llp_ctau_1000_training)} are LLP clusters")
print(f"Validation on {len(valid_dataset)} clusters of which {len(dataset_photon_gun_validation)} are photon gun and {len(dataset_llp_ctau_1000_validation)} are LLP clusters")
print(f"Testing on {len(test_dataset)} clusters of which {len(dataset_photon_gun_test)} are photon gun and {len(dataset_llp_ctau_1000_test)} are LLP clusters")
print(f"Positive weight for training: {pos_weight_training:.2f}")

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# GNN training
from mva.gnn import SimpleGNN

model = SimpleGNN(in_channels=14, global_features=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# BCEWithLogitsLoss expects targets as float, and we include pos_weight to weight positives (LLP class)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_training], dtype=torch.float).to(device))
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], dtype=torch.float))

# Training 
num_epochs = 30
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss_train = 0.0
    for batch_i in train_loader:
        batch_i = batch_i.to(device)
        optimizer.zero_grad()
        # Forward pass including batch_i.u
        out = model(batch_i.x, batch_i.edge_index, batch_i.batch, batch_i.u)
        # Reshape to (batch_i_size, 1) to match out shape
        y = batch_i.y.view(-1, 1).float()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss_train += loss.item() * batch_i.num_graphs

    # Validation loss
    model.eval()
    total_loss_valid = 0.0
    with torch.no_grad():
        for batch_j in valid_loader:
            batch_j = batch_j.to(device)
            optimizer.zero_grad()
            out = model(batch_j.x, batch_j.edge_index, batch_j.batch, batch_j.u)
            y = batch_j.y.view(-1, 1).float()
            loss = criterion(out, y)
            total_loss_valid += loss.item() * batch_j.num_graphs
    
    total_loss_train /= len(train_loader.dataset)
    train_losses.append(total_loss_train)
    total_loss_valid /= len(valid_loader.dataset)
    valid_losses.append(total_loss_valid)
    #print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_loss_train:.4f}")
    #print(f"Epoch [{epoch+1}/{num_epochs}] - valid Loss: {total_loss_valid:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_loss_train:.4f} - Validation Loss: {total_loss_valid:.4f}")


fig, ax = plt.subplots()
ax.plot(train_losses, label="Train")
ax.plot(valid_losses, label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.savefig(f"{cwd}/plots/training/training_loss_curve_gnn.png")

# Evaluate on the test set
model.eval()
test_losses = []
all_outs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.u)
        # If <= 0.0, classify as photon gun (0), else LLP (1) (using BCEWithLogitsLoss)
        preds = (out > 0.0).float().cpu().numpy()
        labels = batch.y.long().view(-1).cpu().numpy()
        
        # Apply sigmod to get probabilities
        out = torch.sigmoid(out)
        all_outs.extend(out.cpu().numpy())
        all_preds.extend(preds)
        all_labels.extend(labels)
    
# Draw a confusion matrix
# Row norm
cm = confusion_matrix(all_labels, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PhotonGun (0)", "LLP (1)"])
disp.plot(values_format='.2f')
plt.title("Confusion Matrix (Row Norm)")
plt.tight_layout()
plt.savefig(f"{cwd}/plots/training/confusion_matrix_rownorm_gnn.png")

# Column norm
cm = confusion_matrix(all_labels, all_preds, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PhotonGun (0)", "LLP (1)"])
disp.plot(values_format='.2f')
plt.title("Confusion Matrix (Column Norm)")
plt.tight_layout()
plt.savefig(f"{cwd}/plots/training/confusion_matrix_colnorm_gnn.png")

# No norm
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PhotonGun (0)", "LLP (1)"])
disp.plot(values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{cwd}/plots/training/confusion_matrix_gnn.png")

# ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(all_labels, all_outs)
# Get the FPR and TPR for the prediction threshold (aka all_preds)
fpr_pred, tpr_pred, thresholds_pred = roc_curve(all_labels, all_preds)
fpr_05 = fpr_pred[thresholds_pred == 1]
tpr_05 = tpr_pred[thresholds_pred == 1]
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'GNN ROC curve (area = {roc_auc:.2f})')
ax.plot(fpr_05, tpr_05, 'ro', label='Threshold = 0.5')
ax.plot(np.logspace(-5, 0, 100), np.logspace(-5, 0, 100), 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.savefig(f"{cwd}/plots/training/roc_curve_gnn.png")

# Draw the first five clusters from each dataset
#for i in range(5):
#    dataset_photon_gun.plot_data_3d(i, plot_dir=f"{cwd}/plots/training/photon_gun", draw_edges=True)
#    dataset_llp_ctau_1000.plot_data_3d(i, plot_dir=f"{cwd}/plots/training/llp_ctau_1000", draw_edges=True)

end = time.time()
print(f"Code executed in {end - start:.2f} seconds")