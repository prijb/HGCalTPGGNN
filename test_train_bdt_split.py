# Train a BDT classifier using a train-validation-test split dataset
import os
import sys

# Data loaders
import torch
import torch.utils
from torch.utils.data import DataLoader, Subset, ConcatDataset

# Stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Timing
import time
start = time.time()

# Consider just the first 3000 clusters
n_skim = 3000
# This is for the "dataset" object
use_cache_dataset = True
cache_dataset_dir = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/datasets"

# Additional imports
# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make plotdir if it doesn't exist
os.makedirs(f"{cwd}/plots/training", exist_ok=True)

from utils.preprocess import Preprocessor
from utils.dataset_bdt import BDTDataset

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

# Get the data for the train, test and validation sets
if not use_cache_dataset:
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
    #w_sum_photon_gun_training = w_photon_gun_training.sum()
    #w_sum_photon_gun_validation = w_photon_gun_validation.sum()
    #w_sum_photon_gun_test = w_photon_gun_test.sum()

    #w_sum_llp_ctau_1000_training = w_llp_ctau_1000_training.sum()
    #w_sum_llp_ctau_1000_validation = w_llp_ctau_1000_validation.sum()
    #w_sum_llp_ctau_1000_test = w_llp_ctau_1000_test.sum()

    # All the pos weights are similar (~2.0-2.1)
    #pos_weight_training = (w_sum_photon_gun_training / w_sum_llp_ctau_1000_training).values[0]
    #pos_weight_validation = (w_sum_photon_gun_validation / w_sum_llp_ctau_1000_validation).values[0]
    #pos_weight_test = (w_sum_photon_gun_test / w_sum_llp_ctau_1000_test).values[0]
    #pos_weight_total = ((w_sum_photon_gun_training + w_sum_photon_gun_validation + w_sum_photon_gun_test) / (w_sum_llp_ctau_1000_training + w_sum_llp_ctau_1000_validation + w_sum_llp_ctau_1000_test)).values[0]

    # Make the BDT datasets
    dataset_photon_gun_training = BDTDataset(X_photon_gun_training, y_photon_gun_training, w_photon_gun_training, u_photon_gun_training)
    dataset_photon_gun_validation = BDTDataset(X_photon_gun_validation, y_photon_gun_validation, w_photon_gun_validation, u_photon_gun_validation)
    dataset_photon_gun_test = BDTDataset(X_photon_gun_test, y_photon_gun_test, w_photon_gun_test, u_photon_gun_test)
    dataset_llp_ctau_1000_training = BDTDataset(X_llp_ctau_1000_training, y_llp_ctau_1000_training, w_llp_ctau_1000_training, u_llp_ctau_1000_training)
    dataset_llp_ctau_1000_validation = BDTDataset(X_llp_ctau_1000_validation, y_llp_ctau_1000_validation, w_llp_ctau_1000_validation, u_llp_ctau_1000_validation)
    dataset_llp_ctau_1000_test = BDTDataset(X_llp_ctau_1000_test, y_llp_ctau_1000_test, w_llp_ctau_1000_test, u_llp_ctau_1000_test)
    print(f"\nGenerated datasets from scratch and saving to cache at {cache_dataset_dir}")
    torch.save(dataset_photon_gun_training, os.path.join(cache_dataset_dir, "dataset_photon_gun_bdt_training.pt"))
    torch.save(dataset_photon_gun_validation, os.path.join(cache_dataset_dir, "dataset_photon_gun_bdt_validation.pt"))
    torch.save(dataset_photon_gun_test, os.path.join(cache_dataset_dir, "dataset_photon_gun_bdt_test.pt"))
    torch.save(dataset_llp_ctau_1000_training, os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_bdt_training.pt"))
    torch.save(dataset_llp_ctau_1000_validation, os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_bdt_validation.pt"))
    torch.save(dataset_llp_ctau_1000_test, os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_bdt_test.pt"))
else:
    print(f"\nLoading datasets from cache at {cache_dataset_dir}")
    dataset_photon_gun_training = torch.load(os.path.join(cache_dataset_dir, "dataset_photon_gun_bdt_training.pt"))
    dataset_photon_gun_validation = torch.load(os.path.join(cache_dataset_dir, "dataset_photon_gun_bdt_validation.pt"))
    dataset_photon_gun_test = torch.load(os.path.join(cache_dataset_dir, "dataset_photon_gun_bdt_test.pt"))
    dataset_llp_ctau_1000_training = torch.load(os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_bdt_training.pt"))
    dataset_llp_ctau_1000_validation = torch.load(os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_bdt_validation.pt"))
    dataset_llp_ctau_1000_test = torch.load(os.path.join(cache_dataset_dir, "dataset_llp_ctau_1000_bdt_test.pt"))

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

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
validation_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)

# BDT params
bdt_model_params = {
    "subsample": 0.5,
    "max_depth": 5,
    "gamma": 1.0,
    "n_estimators": 200,
    "eta": 0.03,
    "reg_lambda": 1.0,
    "reg_alpha": 1.0,
    "scale_pos_weight": pos_weight_training,
}

print(f"Training BDT with params: {bdt_model_params}")
from mva.bdt import BDTClassifier
model = BDTClassifier(train_params=bdt_model_params, do_eval=True)

X_train, y_train, w_train = next(iter(train_loader))
X_validation, y_validation, w_validation = next(iter(validation_loader))

# Detach for BDT
X_train = X_train.detach().numpy()
y_train = y_train.detach().numpy()
w_train = w_train.detach().numpy()
X_validation = X_validation.detach().numpy()
y_validation = y_validation.detach().numpy()
w_validation = w_validation.detach().numpy()

print("Training BDT with")
print(f"X: {X_train.shape}")
print(f"y: {y_train.shape}")
print(f"w: {w_train.shape}")

model.train([X_train, y_train, w_train], [X_validation, y_validation, w_validation])

epochs, results = model.evaluate(X_validation, y_validation)
print(f"BDT training finished after {epochs} epochs")

# Get the predictions from the test dataset
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
X_test, y_test, w_test = next(iter(test_loader))
X_test = X_test.detach().numpy()
y_test = y_test.detach().numpy()
w_test = w_test.detach().numpy()

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Plot the evaluation performance
if model.do_eval: 
    print("Plotting model evaluation")
    x_axis = range(0, epochs)

    # Loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Logloss')
    ax.legend()
    plt.savefig(f"{cwd}/plots/training/training_loss_curve_bdt.png")

    # Confusion matrix (row norm)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PhotonGun (0)", "LLP (1)"])
    disp.plot(values_format='.2f')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{cwd}/plots/training/confusion_matrix_rownorm_bdt.png")

    # Confusion matrix (column norm)
    cm = confusion_matrix(y_test, y_pred, normalize='pred')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PhotonGun (0)", "LLP (1)"])
    disp.plot(values_format='.2f')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{cwd}/plots/training/confusion_matrix_colnorm_bdt.png")

    # Confusion matrix (no norm)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PhotonGun (0)", "LLP (1)"])
    disp.plot(values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{cwd}/plots/training/confusion_matrix_bdt.png")

    # ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    # Get the FPR and TPR for the y_pred 
    fpr_pred, tpr_pred, thresholds_pred = roc_curve(y_test, y_pred)
    fpr_05 = fpr_pred[thresholds_pred == 1]
    tpr_05 = tpr_pred[thresholds_pred == 1]
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'BDT ROC curve (area = {roc_auc:.2f})')
    ax.plot(fpr_05, tpr_05, 'ro', label='Threshold = 0.5')
    ax.plot(np.logspace(-5, 0, 100), np.logspace(-5, 0, 100), 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.savefig(f"{cwd}/plots/training/roc_curve_bdt.png")

end = time.time()
print(f"Code executed in {end - start:.2f} seconds")