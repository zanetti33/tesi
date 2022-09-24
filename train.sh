#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate radiography_retrieval
python3 train.py -lr 2e-4 -e 5 35000