#!/bin/bash

streamlit run  --server.port 37337 --server.fileWatcherType none /home/zanetti/web_app.py -- --pretrained_model /home/zanetti/test_triplet_loss_hard/best_models/trunk_25.pth --dataloader_workers 8