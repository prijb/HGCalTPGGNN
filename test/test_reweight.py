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

# Renormalize
weights_photon_gun = weights_photon_gun / ak.sum(weights_photon_gun)
weights_llp = weights_llp / ak.sum(weights_llp)

# Reweight LLP to photon gun 
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

model = XGBClassifier(n_estimators=200, max_depth=3)
model.set_params(eval_metric=["logloss"])
model.fit(
    pd.concat([kinematics_photon_gun, kinematics_llp]), 
    pd.concat([ak.to_dataframe(class_label_photon_gun), ak.to_dataframe(class_label_llp)]),
    sample_weight = pd.concat([ak.to_dataframe(weights_photon_gun), ak.to_dataframe(weights_llp)]) * (len(weights_photon_gun) + len(weights_llp)),
)
prob_llp = model.predict_proba(kinematics_llp)
weights_llp_bdt = np.clip(prob_llp[:, 0], 1e-6, 1-1e-6)/np.clip(prob_llp[:, 1], 1e-6, 1-1e-6)
weights_llp_bdt = weights_llp_bdt / np.sum(weights_llp_bdt)

fig, ax = plt.subplots()
ax.hist(cl3d_photon_gun.energy, bins=50, range=(0, 1000), weights=weights_photon_gun, histtype="step", color="skyblue", label="Photon gun")
ax.hist(cl3d_llp.energy, bins=50, range=(0, 1000), weights=weights_llp, ls="--", histtype="step", color="firebrick", label="LLP")
ax.hist(cl3d_llp.energy, bins=50, range=(0, 1000), weights=weights_llp_bdt, histtype="step", color="firebrick", label="LLP (rwgt to photon gun)")
ax.set_xlabel("Cluster energy")
ax.set_ylabel("Normalized counts")
#ax.set_yscale("log")
ax.set_title("Energy histogram")
ax.legend()
plt.savefig(f"{cwd}/plots/test_reweight/energy_histogram_photon_gun_vs_llp.png")

fig, ax = plt.subplots()
ax.hist(cl3d_photon_gun.eta, bins=30, range=(-3.2, 3.2), weights=weights_photon_gun, histtype="step", color="skyblue", label="Photon gun")
ax.hist(cl3d_llp.eta, bins=30, range=(-3.2, 3.2), weights=weights_llp, ls="--", histtype="step", color="firebrick", label="LLP")
ax.hist(cl3d_llp.eta, bins=30, range=(-3.2, 3.2), weights=weights_llp_bdt, histtype="step", color="firebrick", label="LLP (rwgt to photon gun)")
ax.set_xlabel("Cluster eta")
ax.set_ylabel("Normalized counts")
#ax.set_yscale("log")
ax.set_title("Eta histogram")
ax.legend()
plt.savefig(f"{cwd}/plots/test_reweight/eta_histogram_photon_gun_vs_llp.png")
