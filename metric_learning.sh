#!/bin/bash
#source /opt/conda/etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate radiography_retrieval

python metric_learning.py "$@"