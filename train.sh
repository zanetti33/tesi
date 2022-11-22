#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate radiography_retrieval
python3 train.py -ws 2500 -wd 1e-4 -st cosine -e 5 -en ./train_no_label_16x16_patch --patch_size 16 
