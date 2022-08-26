#!/bin/bash

docker run --rm --gpus '"device='$CUDA_VISIBLE_DEVICES'"' -v $HOME:$HOME radiography_retrieval /home/zanetti/train.sh