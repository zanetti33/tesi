from codecs import ignore_errors
from distutils.command.config import config
from importlib.resources import path
from operator import index
import random
from transformers import ViltProcessor, DataCollatorForLanguageModeling, ViltConfig, ViltFeatureExtractor, BertTokenizerFast, TrainingArguments
import requests
from PIL import Image
import re
import torch
import pandas as pd
import numpy as np
import os
from datasets import load_metric
import numpy as np
from dataclasses import dataclass
from typing import List, Union, Dict
from model import ViltForMaskedLMAndITM
from data_collator import ViltDataCollator
from transformers.utils import logging
from trainer import MemoryEfficientTrainer
import wandb
import argparse
logger = logging.get_logger(__name__)

#inizializzo wandb
wandb.login(key="9ad1cd077e95967a0961345a858b3028d18c80f5")
wandb.init(project="radiography", entity="tesi-zanetti")

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
argparser.add_argument('-en','--experiment_name', default="train_checkpoint", help="defines the directory where the training checkpoint are saved")
argparser.add_argument('-cd','--checkpoint_dir', default=None, help="defines the directory with the checkpoint from which the training starts")

args = argparser.parse_args()

#inizializzo valori di default
dataset_path = "/datasets/MIMIC-CXR/files"
random.seed(args.seed)
test_size = 1000
max_position_embeddings = 512

#config modello
config = ViltConfig(max_position_embeddings=max_position_embeddings)

#modello pre addestrato
processor = ViltProcessor(ViltFeatureExtractor(resample=3, image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5],size_divisor=32),BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length=max_position_embeddings))
tokenizer = processor.tokenizer
model = ViltForMaskedLMAndITM.from_pretrained("dandelin/vilt-b32-mlm", config=config, ignore_mismatched_sizes=True)

#passaggio del modello su gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()
model = model.to(device)

#leggo il dataframe originale con i vari id (id numerico, immagine, radiografia, paziente)
dataframe = pd.read_csv("/datasets/MIMIC-CXR/random_sampled.csv", index_col=0).sort_index()
print(dataframe.size)
train_dataframe = dataframe[dataframe["split"] == "train"].iloc[:args.N]
val_dataframe = dataframe[dataframe["split"]  == "validate"]
test_dataframe = dataframe[dataframe["split"]  == "test"].iloc[:test_size]

#leggo le label assegnate ai vari record
label_dataframe = pd.read_csv("/datasets/MIMIC-CXR/mimic-cxr-2.0.0-negbio.csv")
train_label_dataframe = label_dataframe.loc[label_dataframe['study_id'].isin(train_dataframe['study_id'])]
val_label_dataframe = label_dataframe.loc[label_dataframe['study_id'].isin(val_dataframe['study_id'])]
test_label_dataframe = label_dataframe.loc[label_dataframe['study_id'].isin(test_dataframe['study_id'])]
print(label_dataframe.head())

#classe dataset
class MimicCxrDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataframe:pd.DataFrame, labels_dataframe:pd.DataFrame):
        self.path = path
        self.dataframe = dataframe
        self.error = True
        self.labels_dataframe = labels_dataframe

    def __getitem__(self, idx):
        texts = []
        images = []
        labels = []
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        elif isinstance(idx, int):
            idx = [idx]
        else:
            print("Error, type not expected: " + type(idx))
        for i in idx:
            if i >= len(self.dataframe):
                text_index = i % len(self.dataframe)
                text_row = self.dataframe.iloc[text_index]
                labels_to_check = self.labels_dataframe.loc[self.labels_dataframe['study_id'] == text_row.study_id] == 1.0
                searching = True
                while searching:
                    randomIdx = random.randint(0, len(self.dataframe) - 1)
                    image_row = self.dataframe.iloc[randomIdx]
                    image_labels = self.labels_dataframe.loc[self.labels_dataframe["study_id"] == image_row.study_id]
                    print(image_labels[labels_to_check])
                    searching = (len(labels_to_check.index) > 0 and (image_labels[labels_to_check].squeeze() == 1.0).any(axis=None)) or text_row.study_id == image_row.study_id
                    if searching:
                        self.error = False
            else:
                text_row = self.dataframe.iloc[i]
                image_row = self.dataframe.iloc[i]
            txt_path = os.path.join(dataset_path, f"p{str(text_row.subject_id)[:2]}", f"p{text_row.subject_id}",f"s{text_row.study_id}.txt")
            image_path = os.path.join(dataset_path, f"p{str(image_row.subject_id)[:2]}", f"p{image_row.subject_id}", f"s{image_row.study_id}", f"{image_row.dicom_id}.jpg")
            with open(txt_path, "r") as f:
                text = f.read()
                texts.append(text)
            image = Image.open(image_path)
            image = np.array(image)
            image = np.stack((image,) * 3, axis=-1)
            images.append(image)
            labels.append(int(i < len(self.dataframe)))
        return {"texts": texts, "images": images, "next_sentence_labels": labels}

    def __len__(self):
        return 2 * len(self.dataframe)

    def checkerror(self):
        return self.error

#creo i dataset di train, test e validation
train_dataset = MimicCxrDataset(dataset_path, train_dataframe, train_label_dataframe)
test_dataset = MimicCxrDataset(dataset_path, test_dataframe, test_label_dataframe)
val_dataset = MimicCxrDataset(dataset_path, val_dataframe, val_label_dataframe)

#custom data collator
mlm_data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability=0.15)
my_data_collator = ViltDataCollator(processor, mlm_data_collator)

#metriche del training
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    (mlm_predictions,itm_predictions,mlm_loss,itm_loss), (mlm_labels, itm_labels) = eval_pred
    #mlm_predictions = np.argmax(mlm_logits, axis=-1)
    #itm_predictions = np.argmax(itm_logits, axis=-1)
    return {"mlm_acc": metric.compute(predictions=mlm_predictions.flatten(), references=mlm_labels.flatten()),
        "itm_acc": metric.compute(predictions=itm_predictions, references=itm_labels),
        "mlm_loss": mlm_loss.mean(),
        "itm_loss": itm_loss.mean()}

training_args = TrainingArguments(
    output_dir=args.experiment_name, 
    evaluation_strategy="epoch",
    eval_accumulation_steps=1,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    warmup_steps=args.warmup_steps,
    seed=args.seed,
    data_seed=args.seed,
    logging_steps=100,
    remove_unused_columns=False, 
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    save_total_limit=2,
    report_to="wandb")

#training
trainer = MemoryEfficientTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=my_data_collator
)

trainer.train(resume_from_checkpoint=args.checkpoint_dir)

print("\n\n CHECK FOR ERROR:\n", " THERE WAS 0 MISS (PROBABLY AN ERROR)" if train_dataset.checkerror() else " THERE WERE SOME MISSES (PROBABLY OKAY)")
