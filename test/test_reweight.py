# Test reweighting dataset from preprocessed output
import os 
import sys
import numpy as np
import pandas as pd
import awkward as ak
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

# Plot the ROC curve
from sklearn.metrics import roc_curve, auc

# Imports from project
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
cwd = os.getcwd()
from utils.preprocess import Preprocessor

preprocessor_photon_gun = Preprocessor([f"{cwd}/ntuples/photon_gun.root"], cache_dir=f"{cwd}/cache/preprocess/photon_gun", use_existing_cache=True, batch_size=10000, class_label=0)
preprocessor_llp = Preprocessor([f"{cwd}/ntuples/llp_ctau_1000.root"], cache_dir=f"{cwd}/cache/preprocess/llp_ctau_1000", use_existing_cache=True, batch_size=10000, class_label=1)

preprocessor_photon_gun.cache_files()
preprocessor_llp.cache_files()

data_dict_photon_gun = preprocessor_photon_gun.get_data_dict()
data_dict_llp = preprocessor_llp.get_data_dict()

# Kinematic reweighting
gen_photons_photon_gun = ak.flatten(data_dict_photon_gun["gen_photons"], axis=1)
gen_photons_llp = ak.flatten(data_dict_llp["gen_photons"], axis=1)
cl3d_photon_gun = ak.flatten(data_dict_photon_gun["cl3d_best"], axis=1)
cl3d_llp = ak.flatten(data_dict_llp["cl3d_best"], axis=1)
class_label_photon_gun = ak.flatten(data_dict_photon_gun["class_label"], axis=1)
class_label_llp = ak.flatten(data_dict_llp["class_label"], axis=1)
weights_photon_gun = ak.flatten(data_dict_photon_gun["weights"], axis=1)
weights_llp = ak.flatten(data_dict_llp["weights"], axis=1)

# Filter out nones
cl3d_photon_gun_mask = ~ak.is_none(cl3d_photon_gun.pt)
cl3d_llp_mask = ~ak.is_none(cl3d_llp.pt)
gen_photons_photon_gun = gen_photons_photon_gun[cl3d_photon_gun_mask]
gen_photons_llp = gen_photons_llp[cl3d_llp_mask]
cl3d_photon_gun = cl3d_photon_gun[cl3d_photon_gun_mask]
cl3d_llp = cl3d_llp[cl3d_llp_mask]
class_label_photon_gun = class_label_photon_gun[cl3d_photon_gun_mask]
class_label_llp = class_label_llp[cl3d_llp_mask]
weights_photon_gun = weights_photon_gun[cl3d_photon_gun_mask]
weights_llp = weights_llp[cl3d_llp_mask]

# Consider only clusters with a large gen particle trajectory offset (alpha > 5 deg)
alpha_cut = gen_photons_llp["alpha"] > (5.0 * np.pi / 180.0)
gen_photons_llp = gen_photons_llp[alpha_cut]
cl3d_llp = cl3d_llp[alpha_cut]
class_label_llp = class_label_llp[alpha_cut]
weights_llp = weights_llp[alpha_cut]

# Renormalize
weights_photon_gun = weights_photon_gun / ak.sum(weights_photon_gun)
weights_llp = weights_llp / ak.sum(weights_llp)

## Reweight LLP to photon gun (BDT)
kinematics_photon_gun = ak.to_dataframe(
    ak.zip(
        {
            "energy": cl3d_photon_gun.energy, 
            "eta": cl3d_photon_gun.eta,
            #"phi": cl3d_photon_gun.phi,
        }
    )
)
kinematics_llp = ak.to_dataframe(
    ak.zip(
        {
            "energy": cl3d_llp.energy, 
            "eta": cl3d_llp.eta,
            #"phi": cl3d_llp.phi,
        }
    )
)
#model = XGBClassifier(n_estimators=200, max_depth=3)
model = XGBClassifier()
model.set_params(eval_metric=["logloss"])
model.fit(
    pd.concat([kinematics_photon_gun, kinematics_llp]), 
    pd.concat([ak.to_dataframe(class_label_photon_gun), ak.to_dataframe(class_label_llp)]),
    sample_weight = pd.concat([ak.to_dataframe(weights_photon_gun), ak.to_dataframe(weights_llp)]) * (len(weights_photon_gun) + len(weights_llp)),
)
prob_llp = model.predict_proba(kinematics_llp)
weights_llp_bdt = np.clip(prob_llp[:, 0], 1e-6, 1-1e-6)/np.clip(prob_llp[:, 1], 1e-6, 1-1e-6)
weights_llp_bdt = weights_llp_bdt / np.sum(weights_llp_bdt)

## Reweight LLP to photon gun (Histogram)
counts_photon_gun, bins_e, bins_eta = np.histogram2d(cl3d_photon_gun.energy, cl3d_photon_gun.eta, bins=[30, 30], range=[[0, 1000], [-3.2, 3.2]], weights=weights_photon_gun)
counts_llp, _, _ = np.histogram2d(cl3d_llp.energy, cl3d_llp.eta, bins=[30, 30], range=[[0, 1000], [-3.2, 3.2]], weights=weights_llp)
weights_llp_histogram = ak.zeros_like(weights_llp)
for i in range(len(bins_e) - 1):
    x_low = bins_e[i]
    x_high = bins_e[i+1]
    mask_photon_gun_i = np.logical_and(cl3d_photon_gun.energy >= x_low, cl3d_photon_gun.energy < x_high) 
    mask_llp_i = np.logical_and(cl3d_llp.energy >= x_low, cl3d_llp.energy < x_high) 

    for j in range(len(bins_eta) - 1):
        y_low = bins_eta[j]
        y_high = bins_eta[j+1]
        mask_photon_gun_j = np.logical_and(cl3d_photon_gun.eta >= y_low, cl3d_photon_gun.eta < y_high) 
        mask_llp_j = np.logical_and(cl3d_llp.eta >= y_low, cl3d_llp.eta < y_high) 

        mask_photon_gun = np.logical_and(mask_photon_gun_i, mask_photon_gun_j)
        mask_llp = np.logical_and(mask_llp_i, mask_llp_j)

        if (len(cl3d_llp[mask_llp]) > 0) and (counts_llp[i][j] > 0):
            weights_llp_histogram = weights_llp_histogram + (mask_llp * counts_photon_gun[i][j]/counts_llp[i][j])
weights_llp_histogram = weights_llp_histogram / np.sum(weights_llp_histogram)

## Plot the cluster variables
#plotvars = ["id", "pt", "energy", "eta", "phi", "hoe", "meanz", "rho_roverz_z", "rho_roverz_z_eweight", "rho_phi_z", "rho_phi_z_eweight", "rho_roverz_phi", "rho_roverz_phi_eweight"]
plotvars = ["pt", "energy", "eta", "phi", "hoe", "meanz", "clusters_n", "showerlength", "coreshowerlength", "firstlayer", "maxlayer", "seetot", "seemax", "spptot", "sppmax", "szz", "srrtot", "srrmax", "srrmean", "varrr", "varzz", "varee", "varpp", "emaxe", "layer10", "layer50", "layer90", "first1layers", "first3layers", "first5layers", "emax1layers", "emax3layers", "emax5layers", "ntc67", "ntc90", "rho_roverz_z", "rho_roverz_z_eweight", "rho_phi_z", "rho_phi_z_eweight", "rho_roverz_phi", "rho_roverz_phi_eweight"]
for field in cl3d_photon_gun.fields:
    if field not in plotvars: continue

    print(f"Plotting cluster {field}")
    # Bin edges are estimated using an unweighted sample
    #_, bins = np.histogram(cl3d_photon_gun[field], bins="auto")
    _, bins = np.histogram(cl3d_llp[field], bins="auto")
    bins = np.linspace(bins[0], bins[-1], 30)
    counts_photon_gun, _ = np.histogram(cl3d_photon_gun[field], bins=bins, weights=weights_photon_gun)
    counts_llp, _ = np.histogram(cl3d_llp[field], bins=bins, weights=weights_llp)
    counts_llp_rwgt_bdt, _ = np.histogram(cl3d_llp[field], bins=bins, weights=weights_llp_bdt)
    counts_llp_rwgt_histogram, _ = np.histogram(cl3d_llp[field], bins=bins, weights=weights_llp_histogram)

    fig, ax = plt.subplots()
    hep.histplot(counts_photon_gun, bins=bins, ax=ax, histtype="step", color="skyblue", label="Photon gun", flow=None)
    hep.histplot(counts_llp, bins=bins, ax=ax, histtype="step", color="firebrick", label="LLP", flow=None)
    hep.histplot(counts_llp_rwgt_bdt, bins=bins, ax=ax, histtype="step", color="firebrick", ls="--", label="LLP (BDT to photon gun)", flow=None)
    hep.histplot(counts_llp_rwgt_histogram, bins=bins, ax=ax, histtype="step", color="firebrick", ls=":", label="LLP (Histo to photon gun)", flow=None)
    ax.set_xlabel(f"Cluster {field}")
    ax.set_ylabel("Normalized counts")
    # Log scale variables
    if field in ["energy", "hoe", "meanz"]: 
        ax.set_yscale("log")
    ax.legend()
    plt.savefig(f"{cwd}/plots/test_reweight/{field}_photon_gun_vs_llp.png")
    plt.close()

    # ROC curve
    x = np.concatenate([ak.to_numpy(cl3d_photon_gun[field]), ak.to_numpy(cl3d_llp[field])])
    y = np.concatenate([ak.to_numpy(class_label_photon_gun), ak.to_numpy(class_label_llp)])
    w = np.concatenate([ak.to_numpy(weights_photon_gun), ak.to_numpy(weights_llp)])
    w_bdt = np.concatenate([ak.to_numpy(weights_photon_gun), ak.to_numpy(weights_llp_bdt)])
    w_histogram = np.concatenate([ak.to_numpy(weights_photon_gun), ak.to_numpy(weights_llp_histogram)])
    fpr, tpr, thresholds = roc_curve(y, x, sample_weight=w)
    fpr_bdt, tpr_bdt, thresholds_bdt = roc_curve(y, x, sample_weight=w_bdt)
    fpr_histogram, tpr_histogram, thresholds_histogram = roc_curve(y, x, sample_weight=w_histogram)
    roc_auc = auc(fpr, tpr)
    roc_auc_bdt = auc(fpr_bdt, tpr_bdt)
    roc_auc_histogram = auc(fpr_histogram, tpr_histogram)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot(fpr_bdt, tpr_bdt, ls="--", label=f'ROC curve BDT rgwt (area = {roc_auc_bdt:.2f})')
    ax.plot(fpr_histogram, tpr_histogram, ls=":", label=f'ROC curve Histogram rgwt (area = {roc_auc_histogram:.2f})')
    ax.plot(np.logspace(-5, 0, 100), np.logspace(-5, 0, 100), 'k--')
    ax.set_xlim(1e-5, 1)
    ax.set_xscale('log')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"ROC curve for {field}")
    ax.legend()
    plt.savefig(f"{cwd}/plots/test_reweight/roc_curve_{field}.png")
