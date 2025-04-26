# Mix a signal and background class and train a GNN classifier
import os 
import sys

# GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Additional imports
# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make plotdir if it doesn't exist
os.makedirs(f"{cwd}/plots/training", exist_ok=True)


from utils.preprocess import Preprocessor
from utils.dataset_graph import GraphDataset

preprocessor_photon_gun = Preprocessor([f"{cwd}/ntuples/photon_gun.root"], cache_dir=f"{cwd}/cache/photon_gun", use_existing_cache=True, batch_size=10000, class_label=0)
preprocessor_llp_ctau_1000 = Preprocessor([f"{cwd}/ntuples/llp_ctau_1000.root"], cache_dir=f"{cwd}/cache/llp_ctau_1000", use_existing_cache=True, batch_size=10000, class_label=1)

preprocessor_photon_gun.cache_files()
preprocessor_llp_ctau_1000.cache_files()

X_photon_gun, y_photon_gun, w_photon_gun, u_photon_gun = preprocessor_photon_gun.get_data_dict()
X_llp_ctau_1000, y_llp_ctau_1000, w_llp_ctau_1000, u_llp_ctau_1000 = preprocessor_llp_ctau_1000.get_data_dict()

#print(X_photon_gun.xs(0, level="subentry"))
#print(len(X_photon_gun.xs(0, level="entry").iloc[0, :]))

# Scale up the LLP ctau weights
w_sum_photon_gun = w_photon_gun.sum()
w_sum_llp_ctau_1000 = w_llp_ctau_1000.sum()
w_llp_ctau_1000 = w_llp_ctau_1000 * w_sum_photon_gun / w_sum_llp_ctau_1000
pos_weight = (w_sum_photon_gun / w_sum_llp_ctau_1000).values[0]

# Create the datasets
dataset_photon_gun = GraphDataset(X_photon_gun, y_photon_gun, w_photon_gun, u_photon_gun, use_knn=True, k=5)
dataset_llp_ctau_1000 = GraphDataset(X_llp_ctau_1000, y_llp_ctau_1000, w_llp_ctau_1000, u_llp_ctau_1000, use_knn=True, k=5)

# Split the datasets (50/50)
n_photon_gun = len(dataset_photon_gun)  
n_llp_ctau_1000 = len(dataset_llp_ctau_1000)
n_train_photon_gun = int(0.5*n_photon_gun)
n_train_llp_ctau_1000 = int(0.5*n_llp_ctau_1000)

train_photon_gun = dataset_photon_gun[:n_train_photon_gun]
train_llp_ctau_1000 = dataset_llp_ctau_1000[:n_train_llp_ctau_1000]
test_photon_gun = dataset_photon_gun[n_train_photon_gun:]
test_llp_ctau_1000 = dataset_llp_ctau_1000[n_train_llp_ctau_1000:]

train_dataset = train_photon_gun + train_llp_ctau_1000
test_dataset = test_photon_gun + test_llp_ctau_1000

# Iterate over the combined dataset until the end
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print(f"Training on {len(train_dataset)} clusters of which {n_train_photon_gun} are photon gun and {n_train_llp_ctau_1000} are LLP clusters")
print(f"Scaling up the LLP ctau weights by a factor of {pos_weight:.2f}")

# GNN training
from mva.gnn import SimpleGNN

model = SimpleGNN(in_channels=14, global_features=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# BCEWithLogitsLoss expects targets as float, and we include pos_weight to weight positives (LLP class)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float))
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], dtype=torch.float))

# Training
num_epochs = 10
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss_train = 0.0
    for batch_i in train_loader:
        optimizer.zero_grad()
        # Forward pass including batch_i.u
        out = model(batch_i.x, batch_i.edge_index, batch_i.batch, batch_i.u)
        # Reshape to (batch_i_size, 1) to match out shape
        y = batch_i.y.view(-1, 1).float()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss_train += loss.item() * batch_i.num_graphs

    # Test loss
    model.eval()
    total_loss_test = 0.0
    with torch.no_grad():
        for batch_j in test_loader:
            optimizer.zero_grad()
            out = model(batch_j.x, batch_j.edge_index, batch_j.batch, batch_j.u)
            y = batch_j.y.view(-1, 1).float()
            loss = criterion(out, y)
            total_loss_test += loss.item() * batch_j.num_graphs
    
    total_loss_train /= len(train_loader.dataset)
    train_losses.append(total_loss_train)
    total_loss_test /= len(test_loader.dataset)
    test_losses.append(total_loss_test)
    #print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_loss_train:.4f}")
    #print(f"Epoch [{epoch+1}/{num_epochs}] - Test Loss: {total_loss_test:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_loss_train:.4f} - Test Loss: {total_loss_test:.4f}")


fig, ax = plt.subplots()
ax.plot(train_losses, label="Train")
ax.plot(test_losses, label="Test")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.savefig(f"{cwd}/plots/training/training_loss_curve_gnn.png")

########################################
# 5) Evaluate on the test set
########################################
model.eval()
eval_losses = []
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index, batch.batch, batch.u)
        # If <= 0.0, classify as photon gun (0), else LLP (1) (using BCEWithLogitsLoss)
        preds = (out > 0.0).float().cpu().numpy()
        labels = batch.y.long().view(-1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)

########################################
# 6) Draw a confusion matrix
########################################
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["PhotonGun (0)", "LLP (1)"])
disp.plot(values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{cwd}/plots/training/confusion_matrix_gnn.png")


########################################
# 7) Draw a ROC curve
########################################
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.savefig(f"{cwd}/plots/training/roc_curve_gnn.png")

# Draw the first five clusters from each dataset
#for i in range(5):
#    dataset_photon_gun.plot_data_3d(i, plot_dir=f"{cwd}/plots/training/photon_gun", draw_edges=True)
#    dataset_llp_ctau_1000.plot_data_3d(i, plot_dir=f"{cwd}/plots/training/llp_ctau_1000", draw_edges=True)