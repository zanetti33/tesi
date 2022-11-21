#!/bin/bash

docker run --rm --gpus '"device='$CUDA_VISIBLE_DEVICES'"' -v $HOME:$HOME -v /datasets/MIMIC-CXR:/datasets/MIMIC-CXR radiography_retrieval /home/zanetti/search.sh