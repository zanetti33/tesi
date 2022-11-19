from dataset import MimicCxrDatasetBasic
from data_collator import ViltDataCollatorBasic
from model import ViltForMaskedLMAndITM
from transformers import ViltProcessor, ViltConfig, ViltFeatureExtractor, BertTokenizerFast
from torch.utils.data import DataLoader
import torch
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('model_directory', help="defines the directory with the checkpoint from which to take the model weights")
argparser.add_argument('-bs','--eval_batch_size', type=int, default=10, help="batch size in evaluation")
argparser.add_argument('-dw','--dataloader_workers', type=int, default=0, help="number of workers in dataloader")
argparser.add_argument('-n', type=int, default=10, help="number of batch to analise")
argparser.add_argument('-k', type=int, default=20, help="number of the max values to keep")
args = argparser.parse_args()
dataset_path = "/datasets/MIMIC-CXR"
max_position_embeddings = 512

config = ViltConfig(max_position_embeddings=max_position_embeddings)
processor = ViltProcessor(ViltFeatureExtractor(resample=3, image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5],size_divisor=32),BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length=max_position_embeddings))
val_dataset = MimicCxrDatasetBasic(dataset_path, split="train")
data_collator = ViltDataCollatorBasic(processor)

# getting the model weights
if args.model_directory is not None:
    model = ViltForMaskedLMAndITM.from_pretrained(args.model_directory, config=config, ignore_mismatched_sizes=True)
else:
    model = ViltForMaskedLMAndITM.from_pretrained("dandelin/vilt-b32-mlm", config=config, ignore_mismatched_sizes=True)

# passaggio del modello su gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()
model = model.to(device)

dataloader = DataLoader(
        val_dataset, 
        args.eval_batch_size, 
        False, 
        collate_fn=data_collator, 
        num_workers=args.dataloader_workers
    )

x = 0
best_values = torch.Tensor()
best_values_ids = torch.IntTensor()
for batch in dataloader:
    images, texts, indexes = batch
    input_ids = texts["input_ids"].to(device)
    attention_mask = texts["attention_mask"].to(device)
    token_type_ids = texts["token_type_ids"].to(device)
    pixel_values = images["pixel_values"].to(device)
    pixel_mask = images["pixel_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask)
    logits = outputs.itm_logits[:,1].cpu()
    best_values = torch.cat((best_values, logits))
    best_values_ids = torch.cat((best_values_ids, indexes))
    if best_values.size(-1) > args.k:
        best_values, best_ids = torch.topk(best_values, args.k)
        best_values_ids = best_values_ids[best_ids]
    del outputs, logits, input_ids, token_type_ids, attention_mask, pixel_mask, pixel_values, batch, images, texts
    torch.cuda.empty_cache()
    x += 1
    print(x)
    if x == args.n:
        break

best_values_ids = best_values_ids
best_values = best_values
print("Index: Value")
for i in range(args.k):
    print(str(best_values_ids[i].item()) + ":", best_values[i].item())
