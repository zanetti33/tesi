#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate radiography_retrieval
python3 find_best.py -k 50 -n 9000 -bs 20 ./test_check_label/checkpoint-92000/
