# This code caches the files in an ntuple directory into training, validation, and test sets
import os
import sys

import torch 

# Timing
import time
start = time.time()

# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make plotdir if it doesn't exist

from utils.preprocess import Preprocessor

# TO DO: Read from a yaml
photon_gun_dir = "/vols/cms/pb4918/StoreNTuple/HGCalTPG/DoublePhoton"
photon_gun_cache_dir = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/photon_gun"
llp_ctau_1000_dir = "/vols/cms/pb4918/StoreNTuple/HGCalTPG/LLPCtau1000NoPU"
llp_ctau_1000_cache_dir = "/vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/cache_new/llp_ctau_1000"

# File list is all files ending in .root
photon_gun_files = [f"{photon_gun_dir}/{f}" for f in os.listdir(photon_gun_dir) if f.endswith(".root")]
llp_ctau_1000_files = [f"{llp_ctau_1000_dir}/{f}" for f in os.listdir(llp_ctau_1000_dir) if f.endswith(".root")]

# Train-valid-test split
train_frac = 0.6
valid_frac = 0.2
test_frac = 0.2

photon_gun_train_files = photon_gun_files[:int(len(photon_gun_files)*train_frac)]
photon_gun_valid_files = photon_gun_files[int(len(photon_gun_files)*train_frac):int(len(photon_gun_files)*(train_frac+valid_frac))]
photon_gun_test_files = photon_gun_files[int(len(photon_gun_files)*(train_frac+valid_frac)):]

llp_ctau_1000_train_files = llp_ctau_1000_files[:int(len(llp_ctau_1000_files)*train_frac)]
llp_ctau_1000_valid_files = llp_ctau_1000_files[int(len(llp_ctau_1000_files)*train_frac):int(len(llp_ctau_1000_files)*(train_frac+valid_frac))]
llp_ctau_1000_test_files = llp_ctau_1000_files[int(len(llp_ctau_1000_files)*(train_frac+valid_frac)):]

print(f"Photon gun {len(photon_gun_files)} files split into {len(photon_gun_train_files)} train, {len(photon_gun_valid_files)} valid, {len(photon_gun_test_files)} test")
print(f"LLP ctau 1000 {len(llp_ctau_1000_files)} files split into {len(llp_ctau_1000_train_files)} train, {len(llp_ctau_1000_valid_files)} valid, {len(llp_ctau_1000_test_files)} test")

# Make the cache directories 
os.makedirs(f"{photon_gun_cache_dir}/train", exist_ok=True)
os.makedirs(f"{photon_gun_cache_dir}/validation", exist_ok=True)
os.makedirs(f"{photon_gun_cache_dir}/test", exist_ok=True)
os.makedirs(f"{llp_ctau_1000_cache_dir}/train", exist_ok=True)
os.makedirs(f"{llp_ctau_1000_cache_dir}/validation", exist_ok=True)
os.makedirs(f"{llp_ctau_1000_cache_dir}/test", exist_ok=True)

print("\nCache directories created")

print("\nSetting up cache job")
# Write a list of arguments for condor jobs 
# Args: INPUT_PATH OUTPUT_PATH CLASS_LABEL
with open("submission/cache_file_args.txt", "w") as f:
    for file in photon_gun_train_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {photon_gun_cache_dir}/train/file_{filenum}.pkl 0\n")
    for file in photon_gun_valid_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {photon_gun_cache_dir}/validation/file_{filenum}.pkl 0\n")
    for file in photon_gun_test_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {photon_gun_cache_dir}/test/file_{filenum}.pkl 0\n")
    for file in llp_ctau_1000_train_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {llp_ctau_1000_cache_dir}/train/file_{filenum}.pkl 1\n")
    for file in llp_ctau_1000_valid_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {llp_ctau_1000_cache_dir}/validation/file_{filenum}.pkl 1\n")
    for file in llp_ctau_1000_test_files:
        filenum = (file.split("_")[-1]).split(".")[0]
        f.write(f"{file} {llp_ctau_1000_cache_dir}/test/file_{filenum}.pkl 1\n")

print("\nCache job arguments written to submission/cache_file_args.txt")

# Write the wrapper file
wrapper_file_content = f"""#!/bin/bash
cd /vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN
eval "$(/vols/cms/pb4918/miniforge3/bin/conda shell.bash hook)"
conda activate ml_env
echo "ml env activated"
python3 cache_file.py $1 $2 $3
"""

with open("submission/cache_files_job_wrapper.sh", "w") as f:
    f.write(wrapper_file_content)
os.system("chmod +x submission/cache_files_job_wrapper.sh")

print("\nCache job wrapper file created")

# Create the HTCondor job submission file
submit_file_content = f"""
universe = vanilla
executable = /vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/submission/cache_files_job_wrapper.sh
arguments = $(infile) $(outfile) $(class_label)
output = /vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/submission/cache_logs/outputfile.$(CLUSTER)_$(PROCESS)
error = /vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/submission/cache_logs/errorfile.$(CLUSTER)_$(PROCESS)
log = /vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/submission/cache_logs/cache_file.$(CLUSTER)_$(PROCESS).log
request_cpus = 1
request_memory = 4GB
+MaxRuntime = 3600
queue infile, outfile, class_label from /vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN/submission/cache_file_args.txt
"""

with open("submission/cache_files_job.sub", "w") as f:
    f.write(submit_file_content)

print("\nCache job submission file created")

# Delete existing log files
os.system("rm submission/cache_logs/*")

# Run the condor job
os.system("condor_submit submission/cache_files_job.sub")
