#!/bin/bash
#source /opt/conda/etc/profile.d/conda.sh

#eval "$(conda shell.bash hook)"
#conda activate metric_learning

python /home/zanetti/metric_learning.py -en test_multisim_alpha_1_beta_60 --pretrained_model /home/zanetti/test_check_label/checkpoint-92000 --dataloader_workers 8 --num_epochs 25 --train_batch_size 30 --multsim_alpha 1 --multsim_beta 60