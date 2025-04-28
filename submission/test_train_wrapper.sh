#!/bin/bash
cd /vols/cms/pb4918/HGCalTPG/Apr25/HGCalTPGGNN
eval "$(/vols/cms/pb4918/miniforge3/bin/conda shell.bash hook)"
conda activate ml_env
echo "ml env activated"
python3 test_train_bdt_split.py