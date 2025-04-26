# Train a BDT classifier 
import os
import sys

# Data loaders
import torch
import torch.utils
from torch.utils.data import DataLoader, Subset, ConcatDataset

# Stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Additional imports
# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make plotdir if it doesn't exist
os.makedirs(f"{cwd}/plots/training", exist_ok=True)

from utils.preprocess import Preprocessor
from utils.dataset_bdt import BDTDataset

preprocessor_photon_gun = Preprocessor([f"{cwd}/ntuples/photon_gun.root"], cache_dir=f"{cwd}/cache/photon_gun", use_existing_cache=True, batch_size=10000, class_label=0)
preprocessor_llp_ctau_1000 = Preprocessor([f"{cwd}/ntuples/llp_ctau_1000.root"], cache_dir=f"{cwd}/cache/llp_ctau_1000", use_existing_cache=True, batch_size=10000, class_label=1)

preprocessor_photon_gun.cache_files()
preprocessor_llp_ctau_1000.cache_files()

X_photon_gun, y_photon_gun, w_photon_gun, u_photon_gun = preprocessor_photon_gun.get_data_dict()
X_llp_ctau_1000, y_llp_ctau_1000, w_llp_ctau_1000, u_llp_ctau_1000 = preprocessor_llp_ctau_1000.get_data_dict()

# Scale up the LLP ctau weights
w_sum_photon_gun = w_photon_gun.sum()
w_sum_llp_ctau_1000 = w_llp_ctau_1000.sum()
w_llp_ctau_1000 = w_llp_ctau_1000 * w_sum_photon_gun / w_sum_llp_ctau_1000
pos_weight = (w_sum_photon_gun / w_sum_llp_ctau_1000).values[0]

# Testing multiindexing
# Takes the first TC of each cluster (if it exists)
#print(X_llp_ctau_1000.xs(0, level="subentry"))
# Takes all the TCs of the first cluster
#print(X_llp_ctau_1000.xs(0, level="entry"))
# Get the number of TCs per cluster
#X_llp_ctau_1000_mult = X_llp_ctau_1000.groupby(level="entry").size()
# Slice so that we only take events with more than 90 TCs
#mask = X_llp_ctau_1000_mult > 150
#X_llp_ctau_1000_idx = X_llp_ctau_1000_mult[mask].index
#X_llp_ctau_1000_chosen = X_llp_ctau_1000.loc[X_llp_ctau_1000_idx]
#print(mask)
#print(X_llp_ctau_1000_idx)
#print(X_llp_ctau_1000_chosen)
#print(X_llp_ctau_1000_chosen.xs(5, level="subentry")["energy"])


# Create the datasets
dataset_photon_gun = BDTDataset(X_photon_gun, y_photon_gun, w_photon_gun, u_photon_gun, max_tc=20)
dataset_llp_ctau_1000 = BDTDataset(X_llp_ctau_1000, y_llp_ctau_1000, w_llp_ctau_1000, u_llp_ctau_1000, max_tc=20)

# Split the datasets (50/50)
n_photon_gun = len(dataset_photon_gun)
n_llp_ctau_1000 = len(dataset_llp_ctau_1000)
n_train_photon_gun = int(0.5*n_photon_gun)
n_train_llp_ctau_1000 = int(0.5*n_llp_ctau_1000)

# Make subsets
train_photon_gun = Subset(dataset_photon_gun, list(range(n_train_photon_gun)))
train_llp_ctau_1000 = Subset(dataset_llp_ctau_1000, list(range(n_train_llp_ctau_1000)))
test_photon_gun = Subset(dataset_photon_gun, list(range(n_train_photon_gun, n_photon_gun)))
test_llp_ctau_1000 = Subset(dataset_llp_ctau_1000, list(range(n_train_llp_ctau_1000, n_llp_ctau_1000)))

# Concatenate the datasets
train_dataset = ConcatDataset([train_photon_gun, train_llp_ctau_1000])
test_dataset = ConcatDataset([test_photon_gun, test_llp_ctau_1000])

print(f"Training on {len(train_dataset)} clusters of which {n_train_photon_gun} are photon gun and {n_train_llp_ctau_1000} are LLP clusters")
print(f"Scaling up the LLP ctau weights by a factor of {pos_weight:.2f}")


# BDT trains over the entire training dataset
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

# BDT params
bdt_model_params = {
    "subsample": 1.0,
    "max_depth": 10,
    "n_estimators": 100,
    "eta": 0.05,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "scale_pos_weight": pos_weight,
}

print(f"Training BDT with params: {bdt_model_params}")
from mva.bdt import BDTClassifier
model = BDTClassifier(train_params=bdt_model_params, do_eval=True)

X_train, y_train, w_train = next(iter(train_loader))
X_test, y_test, w_test = next(iter(test_loader))

# Detach for BDT
X_train = X_train.detach().numpy()
y_train = y_train.detach().numpy()
w_train = w_train.detach().numpy()
X_test = X_test.detach().numpy()
y_test = y_test.detach().numpy()
w_test = w_test.detach().numpy()

print("Training BDT with")
print(f"X: {X_train.shape}")
print(f"y: {y_train.shape}")

model.train([X_train, y_train, w_train], [X_test, y_test, w_test])

epochs, results = model.evaluate(X_test, y_test)
print(f"BDT training finished after {epochs} epochs")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Plot the evaluation performance
if model.do_eval:
    print("Plotting model evaluation")
    x_axis = range(0, epochs)

    # Loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Logloss')
    ax.legend()
    plt.savefig(f"{cwd}/plots/training/training_loss_curve_bdt.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PhotonGun (0)", "LLP (1)"])
    disp.plot(values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{cwd}/plots/training/confusion_matrix_bdt.png")

    # ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.savefig(f"{cwd}/plots/training/roc_curve_bdt.png")


