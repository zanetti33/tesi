from distutils.command.config import config
import random
from transformers import (
    ViltProcessor,
    DataCollatorForLanguageModeling,
    ViltConfig,
    ViltFeatureExtractor,
    BertTokenizerFast,
    TrainingArguments,
)
import torch
import pandas as pd
from datasets import load_metric
import numpy as np
from dataset import MimicCxrPretrainingDataset, MimicCxrPretrainingDatasetAnyLabels, MimicCxrPretrainingDatasetRandom
from model import ViltForMaskedLMAndITM
from data_collator import ViltDataCollatorForPretraining
from transformers.utils import logging
from trainer import MemoryEfficientTrainer
import wandb
import argparse

logger = logging.get_logger(__name__)


# assegno a vari parametri i valori dati come argomento
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--wandb_key",
    type=str,
    default="9ad1cd077e95967a0961345a858b3028d18c80f5",
    help="Wandb key for logging",
)
argparser.add_argument("--wandb_project_name", default="radiography", type=str, help="Project name for wandb logs")
argparser.add_argument("--wandb_entity", default="tesi-zanetti", type=str, help="Wandb entity")
#argparser.add_argument("N", type=int, help="defines the number of record in the training dataset (max 36896)")
argparser.add_argument("--neg_selection", choices=["random", "any", "all"], default="random", help="approach used in selecting negative sampling (based on labels)")
argparser.add_argument("-bs", "--batch_size", type=int, default=20, help="defines the batch size for train and eval")
argparser.add_argument("-e", "--epochs", type=int, default=10, help="defines the number of epochs")
argparser.add_argument("-lr", "--learning_rate", type=float, default=2e-5, help="defines the learning rate")
argparser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="defines the weight decay")
argparser.add_argument("-b1", "--adam_beta1", type=float, default=0.9, help="defines the hyperparameter beta 1")
argparser.add_argument("-b2", "--adam_beta2", type=float, default=0.999, help="defines the hyperparameter beta 2")
argparser.add_argument(
    "-st",
    "--lr_scheduler_type",
    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    default="linear",
    help="defines the learning rate scheduling type",
)
argparser.add_argument(
    "-wr", "--warmup_ratio", type=float, default=0.0, help="defines the warmup ration of the training"
)
argparser.add_argument(
    "-ws", "--warmup_steps", type=int, default=0, help="defines the number of warmup steps of the training"
)

argparser.add_argument("--mlm_prob", type=float, default=0.15, help="masked language model probability")

argparser.add_argument(
    "-s", "--seed", type=int, default=42, help="defines the seed used for picking data and training the model"
)
argparser.add_argument(
    "-en",
    "--experiment_name",
    default="train_checkpoint",
    help="defines the directory where the training checkpoint are saved",
)
argparser.add_argument(
    "-cd",
    "--checkpoint_dir",
    default=None,
    help="defines the directory with the checkpoint from which the training starts",
)
argparser.add_argument(
    "--dataset_path",
    type=str,
    default="/datasets/MIMIC-CXR",
    help="Tokenizer used to tokenize texts",
)
argparser.add_argument(
    "--max_position_embeddings",
    default=512,
    type=int,
    help="Maximum number of position embeddings",
)
argparser.add_argument(
    "--patch_size",
    default=32,
    type=int,
    help="Patch size, where each patch is patch_size x patch_size",
)

args = argparser.parse_args()


# inizializzo wandb
wandb.login(key=args.wandb_key)
wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=args.experiment_name)

# inizializzo valori di default
random.seed(args.seed)
torch.manual_seed(args.seed)
# test_size = 1000

# config modello
config = ViltConfig(max_position_embeddings=args.max_position_embeddings, patch_size=args.patch_size)

# modello pre addestrato
processor = ViltProcessor(
    ViltFeatureExtractor(resample=3, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], size_divisor=args.patch_size),
    BertTokenizerFast.from_pretrained("bert-base-uncased", model_max_length=args.max_position_embeddings),
)
tokenizer = processor.tokenizer
model = ViltForMaskedLMAndITM.from_pretrained("dandelin/vilt-b32-mlm", config=config, ignore_mismatched_sizes=True)

# passaggio del modello su gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()
print(device)
model = model.to(device)

# creo i dataset di train, test e validation
if args.neg_selection == "all":
    train_dataset = MimicCxrPretrainingDataset(args.dataset_path, split="train")
    val_dataset = MimicCxrPretrainingDataset(args.dataset_path, split="validate")
elif args.neg_selection == "any":
    train_dataset = MimicCxrPretrainingDatasetAnyLabels(args.dataset_path, split="train")
    val_dataset = MimicCxrPretrainingDatasetAnyLabels(args.dataset_path, split="validate")
else:
    train_dataset = MimicCxrPretrainingDatasetRandom(args.dataset_path, split="train")
    val_dataset = MimicCxrPretrainingDatasetRandom(args.dataset_path, split="validate")


# custom data collator
mlm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_prob)
my_data_collator = ViltDataCollatorForPretraining(processor, mlm_data_collator)

# metriche del training
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    (mlm_predictions, itm_predictions, mlm_loss, itm_loss), (mlm_labels, itm_labels) = eval_pred
    # mlm_predictions = np.argmax(mlm_logits, axis=-1)
    # itm_predictions = np.argmax(itm_logits, axis=-1)
    return {
        "mlm_acc": metric.compute(predictions=mlm_predictions.flatten(), references=mlm_labels.flatten()),
        "itm_acc": metric.compute(predictions=itm_predictions, references=itm_labels),
        "mlm_loss": mlm_loss.mean(),
        "itm_loss": itm_loss.mean(),
    }


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
    report_to="wandb",
)

# training
trainer = MemoryEfficientTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=my_data_collator,
)

trainer.train(resume_from_checkpoint=args.checkpoint_dir)

print(
    "\n\n CHECK FOR ERROR:\n",
    " THERE WAS 0 MISS (PROBABLY AN ERROR)"
    if train_dataset.checkerror()
    else " THERE WERE SOME MISSES (PROBABLY OKAY)",
)
