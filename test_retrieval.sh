#!/bin/bash

docker run --rm  --shm-size=16gb --gpus '"device='$CUDA_VISIBLE_DEVICES'"' -v $HOME:$HOME -v /datasets/MIMIC-CXR:/datasets/MIMIC-CXR metric_learning:latest /home/zanetti/retrieval.sh