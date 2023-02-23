#!/bin/bash

docker run --rm -p 37337:37337 --shm-size=16gb --gpus '"device='$CUDA_VISIBLE_DEVICES'"' -v $HOME:$HOME -v /datasets/MIMIC-CXR:/datasets/MIMIC-CXR web_app:latest /home/zanetti/web_app_args.sh 