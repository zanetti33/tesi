#!/bin/bash
#source /opt/conda/etc/profile.d/conda.sh

#eval "$(conda shell.bash hook)"
#conda activate metric_learning

python /home/zanetti/test_retrieval.py --pretrained_model /home/zanetti/test_multisim_alpha_1_beta_60/best_models/trunk_25.pth --dataloader_workers 8 --task img2txt