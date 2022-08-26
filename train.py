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
from datasets import load_metric
import numpy as np
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass
from typing import List, Union, Dict
from transformers import DataCollatorForWholeWordMask, DataCollatorForLanguageModeling
from model import ViltForMaskedLMAndITM
from data_collator import ViltDataCollator
from transformers.utils import logging
logger = logging.get_logger(__name__)

dataset_path = "./MIMIC-CXR/files"

#modello pre addestrato
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
tokenizer = processor.tokenizer
model = ViltForMaskedLMAndITM.from_pretrained("dandelin/vilt-b32-mlm")

#passaggio del modello su gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

#leggo il dataframe originale con i vari id (id numerico, immagine, radiografia, paziente)
dataframe = pd.read_csv("./MIMIC-CXR/random_sampled.csv", index_col=0).sort_index()
train_dataframe = dataframe[dataframe["split"] == "train"]
test_dataframe = dataframe[dataframe["split"]  == "test"]
val_dataframe = dataframe[dataframe["split"]  == "validate"]

#classe dataset
class MimicCxrDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataframe:pd.DataFrame):
        self.path = path
        self.dataframe = dataframe

    def __getitem__(self, idx):
        text_index = idx % len(self.dataframe)
        if idx > len(self.dataframe):
            randomIdx = random.randint(0, len(self.dataframe) - 1)
            while (randomIdx == idx):
                randomIdx = random.randint(0, len(self.dataframe) - 1)
            image_index = randomIdx
        else:
            image_index = idx
        texts = []
        images = []
        text_row = self.dataframe.iloc[text_index]
        image_row = self.dataframe.iloc[image_index]
        txt_path = os.path.join(dataset_path, f"p{str(text_row.subject_id)[:2]}", f"p{text_row.subject_id}",f"s{text_row.study_id}.txt")
        image_path = os.path.join(dataset_path, f"p{str(image_row.subject_id)[:2]}", f"p{image_row.subject_id}", f"s{image_row.study_id}", f"{image_row.dicom_id}.jpg")
        with open(txt_path, "r") as f:
            text = f.read()
            texts.append(text)
        image = Image.open(image_path)
        image = np.array(image)
        image = np.stack((image,) * 3, axis=-1)
        images.append(image)
        return {"texts": texts, "images": images, "next_sentence_labels": int(idx <= len(self.dataframe))}

    def __len__(self):
        return 2 * len(self.dataframe)

train_dataset = MimicCxrDataset(dataset_path, train_dataframe)
test_dataset = MimicCxrDataset(dataset_path, test_dataframe)
val_dataset = MimicCxrDataset(dataset_path, val_dataframe)

#custom data collator
mlm_data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability=0.15)

my_data_collator = ViltDataCollator(processor, mlm_data_collator)

#metriche del training
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logger.warning(eval_pred)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", remove_unused_columns=False, num_train_epochs=30)

#training
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=my_data_collator
)

trainer.train()