#!/bin/bash
# Make the proxy directory if it doesn't exist
PROXY_DIR_NAME="proxy"
if [ ! -d "${PROXY_DIR_NAME}" ]; then
    mkdir ${PROXY_DIR_NAME}
fi

# Make some other directories (inputs, logs, outputs)
if [ ! -d "cache" ]; then
    mkdir cache
fi
if [ ! -d "condor_submission" ]; then
    mkdir condor_submission
fi
if [ ! -d "outputs" ]; then
    mkdir outputs
fi
if [ ! -d "plots" ]; then
    mkdir plots
fi

# CMS loads
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/grid.cern.ch/alma9-ui-current/etc/profile.d/setup-alma9-test.sh

# Miniforge load
eval "$(/vols/cms/pb4918/miniforge3/bin/conda shell.bash hook)"
conda activate ml_env

# Proxy export
if [ -z ${X509_USER_PROXY+x} ]; then
    echo "Setting up proxy"
#    voms-proxy-init --rfc --voms cms --valid 192:00
    voms-proxy-init --rfc --voms cms --valid 192:00 --out ${PROXY_DIR_NAME}/cms.proxy
else
    echo "Proxy already set up"
fi
export X509_USER_PROXY=${PROXY_DIR_NAME}/cms.proxy
