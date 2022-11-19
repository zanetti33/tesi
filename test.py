from codecs import ignore_errors
from distutils.command.config import config
from importlib.resources import path
from operator import index
import random
from transformers import ViltProcessor, ViltForMaskedLM
from transformers import PreTrainedTokenizer
import requests
from PIL import Image
import re
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from dataset import MimicCxrDatasetBasic
import numpy as np
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass
from typing import List, Union, Dict
from transformers import DataCollatorForWholeWordMask, DataCollatorForLanguageModeling, ViltConfig, ViltFeatureExtractor, BertTokenizerFast
from model import ViltForMaskedLMAndITM, ViltForImageTextMatching
from data_collator import ViltDataCollatorForPretraining
from transformers.utils import logging
from trainer import MemoryEfficientTrainer
import argparse

#assegno a vari parametri i valori dati come argomento
argparser = argparse.ArgumentParser()
argparser.add_argument('N', type=int, help="defines the number of record in the training dataset (max 36896)")
argparser.add_argument('-bs','--batch_size', type=int, default=20, help="defines the batch size for train and eval")
argparser.add_argument('-e','--epochs', type=int, default=10, help="defines the number of epochs")
argparser.add_argument('-lr','--learning_rate', type=float, default=2e-5, help="defines the learning rate")
argparser.add_argument('-wd','--weight_decay', type=float, default=0.0, help="defines the weight decay")
argparser.add_argument('-b1','--adam_beta1', type=float, default=0.9, help="defines the hyperparameter beta 1")
argparser.add_argument('-b2','--adam_beta2', type=float, default=0.999, help="defines the hyperparameter beta 1")
argparser.add_argument('-st','--lr_scheduler_type', choices=['linear','cosine','cosine_with_restarts','polynomial','constant','constant_with_warmup'], default='linear', help="defines the learning rate scheduling type")
argparser.add_argument('-wr','--warmup_ratio', type=float, default=0.0, help="defines the warmup ration of the training")
argparser.add_argument('-ws','--warmup_steps', type=int, default=0, help="defines the number of warmup steps of the training")
argparser.add_argument('-s','--seed', type=int, default=42, help="defines the seed used for picking data and training the model")
args = argparser.parse_args()

#inizializzo valori di default
dataset_path = "/datasets/MIMIC-CXR"
random.seed(args.seed)
test_size = 10

#creo i dataset di train, test e validation
train_dataset = MimicCxrDatasetBasic(dataset_path)

config = ViltConfig(max_position_embeddings=512)
fe = ViltFeatureExtractor(resample=3, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], size_divisor=32)


image = train_dataset[10][0]
print(image.shape)
image = fe.resize(image, (384, 416))
image = np.asarray(image)
print(image.shape)
k = 0
for i in range(0, 416, 120):
    for j in range(0, 384, 120):
        Image.fromarray(image[i:i+32,j:j+32,:]).save("test" + str(k) + ".jpg")
        k+=1