#!/bin/bash

docker run --rm --gpus '"device='$CUDA_VISIBLE_DEVICES'"' -v $HOME:$HOME -v /datasets/MIMIC-CXR:/datasets/MIMIC-CXR metric_learning:latest /home/zanetti/train.sh