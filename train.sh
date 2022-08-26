#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate radiography_retrieval
python3 train.py