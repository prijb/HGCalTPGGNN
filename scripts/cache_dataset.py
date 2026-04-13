# Cache a full dataset
import os
import sys

import torch 

# Timing
import time
start = time.time()

# Add the project path
cwd = os.getcwd()
condor_parent_dir = f"{cwd}/condor_submission/cache"
os.makedirs(f"{condor_parent_dir}", exist_ok=True)
os.makedirs(f"{condor_parent_dir}/logs", exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import Preprocessor

# Dataset directories
photon_gun_dir = "/vols/cms/pb4918/StoreNTuple/HGCalTPG/DoublePhoton"
photon_gun_cache_dir = f"{cwd}/cache/preprocess/photon_gun"
llp_ctau_1000_dir = "/vols/cms/pb4918/StoreNTuple/HGCalTPG/LLPCtau1000NoPU"
llp_ctau_1000_cache_dir = f"{cwd}/cache/preprocess/llp_ctau_1000"
os.makedirs(f"{photon_gun_cache_dir}", exist_ok=True)
os.makedirs(f"{llp_ctau_1000_cache_dir}", exist_ok=True)

# File list is all files ending in .root
photon_gun_files = [f"{photon_gun_dir}/{f}" for f in os.listdir(photon_gun_dir) if f.endswith(".root")]
llp_ctau_1000_files = [f"{llp_ctau_1000_dir}/{f}" for f in os.listdir(llp_ctau_1000_dir) if f.endswith(".root")]

print(f"Photon gun {len(photon_gun_files)} files")
print(f"LLP ctau 1000 {len(llp_ctau_1000_files)} files")

print("\nCache directories created")
print("\nSetting up cache job")

with open(f"{condor_parent_dir}/cache_args.txt", "w") as f:
    for file in photon_gun_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {photon_gun_cache_dir}/file_{filenum}.pkl 0\n")
    for file in llp_ctau_1000_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {llp_ctau_1000_cache_dir}/file_{filenum}.pkl 1\n")
print(f"Cache job arguments written to {condor_parent_dir}/cache_args.txt")

# Write the wrapper file
wrapper_file_content = f"""#!/bin/bash
cd {cwd}

# CMS loads
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/grid.cern.ch/alma9-ui-current/etc/profile.d/setup-alma9-test.sh

# Env activation
eval "$(/vols/cms/pb4918/miniforge3/bin/conda shell.bash hook)"
conda activate ml_env
echo "ml env activated"

# Proxy export
export X509_USER_PROXY={cwd}/proxy/cms.proxy

# Job
python3 scripts/cache_file.py $1 $2 $3
"""
with open(f"{condor_parent_dir}/cache_wrapper.sh", "w") as f:
    f.write(wrapper_file_content)
os.system(f"chmod +x {condor_parent_dir}/cache_wrapper.sh")
print("\nCache job wrapper file created")

# Create the HTCondor job submission file
submit_file_content = f"""
universe = vanilla
executable = {condor_parent_dir}/cache_wrapper.sh
arguments = $(infile) $(outfile) $(class_label)
output = {condor_parent_dir}/logs/job_$(CLUSTER)_$(PROCESS).out
error = {condor_parent_dir}/logs/job_$(CLUSTER)_$(PROCESS).err
log = {condor_parent_dir}/logs/job_$(CLUSTER)_$(PROCESS).log
request_cpus = 1
request_memory = 4GB
use_x509userproxy = true
+MaxRuntime = 7199
queue infile, outfile, class_label from {condor_parent_dir}/cache_args.txt
"""
with open(f"{condor_parent_dir}/cache.sub", "w") as f:
    f.write(submit_file_content)
print("\nCache job submission file created")

# Delete existing log files
os.system(f"rm {condor_parent_dir}/logs/*")

# Run the condor job
os.system(f"condor_submit {condor_parent_dir}/cache.sub")