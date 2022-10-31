#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate radiography_retrieval
python3 train.py -wr 0.2 -ws 8000 -cd ./test_trainer/checkpoint-76500 -e 5 1000000
