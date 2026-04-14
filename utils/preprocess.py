# Using Awkward's cartesian product to match TCs and clusters. Data is cached and retrieved as awkward arrays with dataframe formation done later
import yaml
import uproot   
import awkward as ak
import vector
vector.register_awkward()
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

# Correlation function
def correlation(x, y, weights=None):
    if weights is None:
        w = ak.ones_like(x)
    else:
        w = weights

    W = ak.sum(w, axis=-1)
    SX  = ak.sum(w * x, axis=-1)
    SY  = ak.sum(w * y, axis=-1)
    SXY = ak.sum(w * x * y, axis=-1)
    SXX = ak.sum(w * x * x, axis=-1)
    SYY = ak.sum(w * y * y, axis=-1)
    
    N = W * SXY - SX * SY

    var_x = W * SXX - SX * SX
    var_y = W * SYY - SY * SY
    bad_var = (var_x <= 0) | (var_y <= 0)
    D = np.sqrt(ak.where(var_x > 0, var_x, 0)) * np.sqrt(ak.where(var_y > 0, var_y, 0))
    bad_den = D == 0

    corr = np.abs(N / D)
    bad = bad_var | bad_den | ~np.isfinite(corr)
    corr = ak.where(bad, -999, corr)    

    return corr

def phi_phasewrap(phi):
    """
    Used for example when phi is deltaphi = phi1 - phi2
    """
    return (phi + np.pi) % (2 * np.pi) - np.pi

def eta_to_theta(eta):
    return 2*np.arctan(np.exp(-eta))
def angle(vec1,vec2):
    cos = np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    return np.arccos(cos)



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
            "vtx":["x","y","z"],
            "gen": ["pt", "eta", "phi", "energy", "charge", "pdgid", "status", "daughters"],
            "genpart": ["pt", "eta", "phi", "energy", "pid", "reachedEE", "ovx", "ovy", "ovz", "dvx", "dvy", "dvz", "mother", "exphi", "exeta", "exx", "exy"],
            "tc": ["n", "id", "subdet", "zside", "layer", "waferu", "waferv", "cellu", "cellv", "pt", "energy", "eta", "phi", "x", "y", "z", "cluster_id", "multicluster_id", "multicluster_pt"],
            "cl3d": ["n", "id", "pt", "energy", "eta", "phi", "hoe", "bdteg", "meanz", "clusters_n", "showerlength", "coreshowerlength", "firstlayer", "maxlayer", "seetot", "seemax", "spptot", "sppmax", "szz", "srrtot", "srrmax", "srrmean", "varrr", "varzz", "varee", "varpp", "emaxe", "layer10", "layer50", "layer90", "first1layers", "first3layers", "first5layers", "emax1layers", "emax3layers", "emax5layers", "ntc67", "ntc90"],
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
            events = t.arrays(var_list, library="ak", entry_start=i_batch*self.batch_size, entry_stop=(i_batch+1)*self.batch_size, how="zip")
            total_events = len(events)

            # Basic filters
            mask = events["cl3d_n"] > 0
            events = events[mask]

            # Match TCs to clusters
            tc = events["tc"]
            cl3d = ak.Array(events["cl3d"], with_name="Momentum4D")
            tc_cl3d_pairs = ak.cartesian({"cl3d": cl3d, "tc": tc}, axis=1, nested=True)
            matched_tc_cl3d_pairs = tc_cl3d_pairs[tc_cl3d_pairs.tc.multicluster_id == tc_cl3d_pairs.cl3d.id]
            tcs_per_cl3d = matched_tc_cl3d_pairs.tc
            tcs_energy_order = ak.argsort(tcs_per_cl3d.energy, axis=-1, ascending=False)
            tcs_per_cl3d = tcs_per_cl3d[tcs_energy_order]
            cl3d = ak.with_field(cl3d, tcs_per_cl3d, "tcs")
            
            # Get the correlation
            cl3d_tcs_z = cl3d.tcs.z
            cl3d_tcs_r = np.sqrt(cl3d.tcs.x**2 + cl3d.tcs.y**2)
            cl3d_tcs_phi = cl3d.tcs.phi
            cl3d_rho_roverz_z = correlation(cl3d_tcs_r/cl3d_tcs_z, cl3d_tcs_z)
            cl3d_rho_roverz_z_eweight = correlation(cl3d_tcs_r/cl3d_tcs_z, cl3d_tcs_z, cl3d.tcs.energy)
            cl3d_rho_phi_z = correlation(cl3d_tcs_phi, cl3d_tcs_z)
            cl3d_rho_phi_z_eweight = correlation(cl3d_tcs_phi, cl3d_tcs_z, cl3d.tcs.energy)
            cl3d_rho_roverz_phi = correlation(cl3d_tcs_r/cl3d_tcs_z, cl3d_tcs_phi)
            cl3d_rho_roverz_phi_eweight = correlation(cl3d_tcs_r/cl3d_tcs_z, cl3d_tcs_phi, cl3d.tcs.energy)
            cl3d = ak.with_field(cl3d, cl3d_rho_roverz_z, "rho_roverz_z")
            cl3d = ak.with_field(cl3d, cl3d_rho_roverz_z_eweight, "rho_roverz_z_eweight")
            cl3d = ak.with_field(cl3d, cl3d_rho_phi_z, "rho_phi_z")
            cl3d = ak.with_field(cl3d, cl3d_rho_phi_z_eweight, "rho_phi_z_eweight")
            cl3d = ak.with_field(cl3d, cl3d_rho_roverz_phi, "rho_roverz_phi")
            cl3d = ak.with_field(cl3d, cl3d_rho_roverz_phi_eweight, "rho_roverz_phi_eweight")

            # Do gen-matching/filtering
            genpart = ak.Array(events["genpart"], with_name="Momentum4D")
            gen = ak.Array(events["gen"], with_name="Momentum4D")
            genpart_photons = genpart[genpart.pid==22]

            if self.class_label == 0:
                print(f"\nUsing photon gun gen-filtering")
                gen_photons_selected = genpart_photons[ genpart_photons.mother == -1 ] 
                gen_photons_selected = gen_photons_selected[ ( abs(gen_photons_selected.exeta) > 1.5 )&( abs(gen_photons_selected.exeta) < 3.0 ) ]

            elif self.class_label == 1:
                print("\nUsing LLP gen-filtering")
                gen_daughters = events["gen_daughters"]
                bsm_higgs_decays = gen_daughters[gen.pdgid==25]
                gen_photons = gen[ak.flatten(bsm_higgs_decays,axis=2)]
                gen_photons = gen_photons[abs(gen_photons.pdgid)==22]

                pairs = ak.cartesian( {'gen':gen_photons, 'genpart':genpart_photons} )
                dR_mask = pairs.gen.deltaR(pairs.genpart)==0
                gen_photons_from_BSM = pairs.genpart[dR_mask]
                ax = ak.cartesian({'dvx':gen_photons_from_BSM.ovx, 'pvx':events["vtx_x"]})
                ay = ak.cartesian({'dvy':gen_photons_from_BSM.ovy, 'pvy':events["vtx_y"]})
                az = ak.cartesian({'dvz':gen_photons_from_BSM.ovz, 'pvz':events["vtx_z"]})
                decay_length = ((ax.dvx-ax.pvx)**2+(ay.dvy-ay.pvy)**2+(az.dvz-az.pvz)**2)**0.5
                gen_photons_from_BSM['decay_length'] = decay_length
                gen_photons_reachedEE = gen_photons_from_BSM[( abs(gen_photons_from_BSM.exeta) > 1.5 )&( abs(gen_photons_from_BSM.exeta) < 3.0 ) ]
                #gen_photons_reachedEE = gen_photons_from_BSM[gen_photons_from_BSM.reachedEE == 2]

                gen_photons_selected = gen_photons_reachedEE

            # Get the trajectory offset angle of gen-photons (at first-layer)
            neg_mask = gen_photons_selected.exeta/np.abs(gen_photons_selected.exeta)
            px = gen_photons_selected.pt*np.cos(gen_photons_selected.phi)
            py = gen_photons_selected.pt*np.sin(gen_photons_selected.phi)
            pz = gen_photons_selected.pt*np.sinh(gen_photons_selected.eta)
            pos_x = gen_photons_selected['exx'] 
            pos_y = gen_photons_selected['exy']
            pos_z = np.ones_like(gen_photons_selected.exeta)*322.0
            pos_z = pos_z * neg_mask
            mag_gen = np.sqrt(px**2 + py**2 + pz**2)
            mag_axis = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
            dot_num = px*pos_x + py*pos_y + pz*pos_z 
            dot_angle = np.arccos((dot_num/(mag_gen*mag_axis)))
            alpha = np.abs(dot_angle)
            gen_photons_selected["alpha"] = alpha

            # Apply cuts to gen photons and clusters
            cl3d = cl3d[cl3d.pt > 10]
            gen_photons_selected = gen_photons_selected[gen_photons_selected.pt > 10]

            # Match clusters to gen photons
            pairs = ak.cartesian({"gp":gen_photons_selected, "cl3d":cl3d}, axis=1, nested=True)
            pairs_arg = ak.argcartesian({"gp":gen_photons_selected, "cl3d":cl3d}, axis=1, nested=True)
            dR = ((pairs.gp.exeta-pairs.cl3d.eta)**2 + phi_phasewrap((pairs.gp.exphi-pairs.cl3d.phi))**2)**0.5
            order = ak.argsort(dR, ascending=True, axis=2)
            dR = dR[order]
            pairs_dR_ordered = pairs[order]
            pairs_arg_dR_ordered = pairs_arg[order]
            cut_dR = dR<0.2

            # Assign each gen photon with a matched cluster
            pairs_dR_ordered = pairs_dR_ordered[cut_dR]
            pairs_arg_dR_ordered = pairs_arg_dR_ordered[cut_dR] 
            gp_i, cl3d_i = ak.unzip(pairs_dR_ordered)
            # Calculate the trajectory offset at the cluster position
            neg_mask = gp_i.exeta/np.abs(gp_i.exeta)
            px = gp_i.pt*np.cos(gp_i.phi)
            py = gp_i.pt*np.sin(gp_i.phi)
            pz = gp_i.pt*np.sinh(gp_i.eta)
            cl3d_i_rho = np.abs(np.tan(eta_to_theta(cl3d_i.eta))*cl3d_i.meanz)
            cl3d_i_x = cl3d_i_rho*np.cos(cl3d_i.phi)
            cl3d_i_y = cl3d_i_rho*np.sin(cl3d_i.phi)
            cl3d_i_z = cl3d_i.meanz*neg_mask
            mag_gen = np.sqrt(px**2 + py**2 + pz**2)
            mag_axis = np.sqrt(cl3d_i_x**2 + cl3d_i_y**2 + cl3d_i_z**2)
            axis_phi = cl3d_i.phi
            axis_theta = np.arctan2(cl3d_i_rho, cl3d_i_z)
            llp_phi = np.arctan2(py,px)
            llp_theta = np.arctan2(np.sqrt(py**2 + px**2),pz)
            dot_num = px*cl3d_i_x + py*cl3d_i_y + pz*cl3d_i_z 
            dot_angle = np.arccos((dot_num/(mag_gen*mag_axis)))
            phi_offset = np.abs(phi_phasewrap(llp_phi - axis_phi))
            theta_offset = np.abs(llp_theta - axis_theta)
            cl3d_i = ak.with_field(cl3d_i, dot_angle, "alpha")
            cl3d_i = ak.with_field(cl3d_i, phi_offset, "phi_offset")
            cl3d_i = ak.with_field(cl3d_i, theta_offset, "theta_offset")

            gen_photons_selected["cl3d"] = cl3d_i
            gen_photons_selected["cl3d_best"] = ak.firsts(gen_photons_selected.cl3d, axis=2)
            
            # Save skim (not using dataframes yet)
            gen_photons_skim = gen_photons_selected[[field for field in gen_photons_selected.fields if "cl3d" not in field]]
            cl3d_skim = gen_photons_selected["cl3d"]
            cl3d_best_skim = gen_photons_selected["cl3d_best"]
            data_dict["gen_photons"] = gen_photons_skim
            data_dict["cl3d"] = cl3d_skim
            data_dict["cl3d_best"] = cl3d_best_skim
            data_dict["class_label"] = ak.ones_like(gen_photons_skim.pt) * self.class_label
            output_path = output_path.replace(".pkl", f"_{i_batch}.pkl")
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
                #self.cache_file_from_yaml(input_path, output_path, config_path="config/config.yaml")

    # Load the concatenated data from the cache
    def get_data_dict(self):
        gen_photons = ak.ArrayBuilder()
        cl3d = ak.ArrayBuilder()
        cl3d_best = ak.ArrayBuilder()
        class_label = ak.ArrayBuilder()

        print(f"Loading data from {self.cache_dir}")
        for i, file in enumerate(tqdm(os.listdir(self.cache_dir), total=len(os.listdir(self.cache_dir)))):
            
            cache_file = os.path.join(self.cache_dir, file)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            
            gen_photons_i = data["gen_photons"]
            cl3d_i = data["cl3d"]
            cl3d_best_i = data["cl3d_best"]
            class_label_i = data["class_label"]

            gen_photons.append(gen_photons_i)
            cl3d.append(cl3d_i)
            cl3d_best.append(cl3d_best_i)
            class_label.append(class_label_i)

        gen_photons = gen_photons.snapshot()
        cl3d = cl3d.snapshot()
        cl3d_best = cl3d_best.snapshot()
        class_label = class_label.snapshot()
        
        data_dict = dict()
        data_dict["gen_photons"] = ak.concatenate(gen_photons)
        data_dict["cl3d"] = ak.concatenate(cl3d)
        data_dict["cl3d_best"] = ak.concatenate(cl3d_best)
        data_dict["class_label"] = ak.concatenate(class_label)
        data_dict["weights"] = ak.ones_like(data_dict["class_label"])

        return data_dict    

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    cwd = os.getcwd()

    preprocessor_photon_gun = Preprocessor([f"{cwd}/ntuples/photon_gun.root"], cache_dir=f"{cwd}/cache_test/photon_gun", use_existing_cache=False, batch_size=10000, class_label=0)
    preprocessor_llp = Preprocessor([f"{cwd}/ntuples/llp_ctau_1000.root"], cache_dir=f"{cwd}/cache_test/llp_ctau_1000", use_existing_cache=False, batch_size=10000, class_label=1)
    
    preprocessor_photon_gun.cache_files()
    preprocessor_llp.cache_files()

    data_dict_photon_gun = preprocessor_photon_gun.get_data_dict()
    data_dict_llp = preprocessor_llp.get_data_dict()

    gen_photons_photon_gun = ak.flatten(data_dict_photon_gun["gen_photons"], axis=1)
    gen_photons_llp = ak.flatten(data_dict_llp["gen_photons"], axis=1)
    cl3d_photon_gun = ak.flatten(data_dict_photon_gun["cl3d_best"], axis=1)
    cl3d_llp = ak.flatten(data_dict_llp["cl3d_best"], axis=1)
    class_label_photon_gun = ak.flatten(data_dict_photon_gun["class_label"], axis=1)
    class_label_llp = ak.flatten(data_dict_llp["class_label"], axis=1)
    weights_photon_gun = ak.flatten(data_dict_photon_gun["weights"], axis=1)
    weights_llp = ak.flatten(data_dict_llp["weights"], axis=1)

    print(cl3d_photon_gun.fields)

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
