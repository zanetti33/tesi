# dependencies
from transformers import ViltProcessor
from PIL import Image
import torch
import numpy as np
import pandas as pd
import cv2
import os
from save_html import get_html
from dataset import MimicCxrDatasetBasic
from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients
from transformers import ViltProcessor, ViltConfig, ViltFeatureExtractor, BertTokenizerFast
from model import ViltForMaskedLMAndITM
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('idx', type=int, default=0, help="defines the index of the record to explain")
argparser.add_argument('-s','--study', type=int, default=50175065, help="the study you want to get processed")
argparser.add_argument('-d','--drop_rate', type=float, default=0.9, help="the drop rate used to remove noise")
argparser.add_argument('-md','--model_dir', default=None, help="defines the directory with the checkpoint to use for explainability")
argparser.add_argument('-rd','--result_dir', default="./explainability_results", help="defines the directory where to put the results")
args = argparser.parse_args()
dataset_path = "/datasets/MIMIC-CXR"
max_position_embeddings = 512

config = ViltConfig(max_position_embeddings=max_position_embeddings)
processor = ViltProcessor(ViltFeatureExtractor(resample=3, image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5],size_divisor=32),BertTokenizerFast.from_pretrained('bert-base-uncased', model_max_length=max_position_embeddings))

# getting the model weights
if args.model_dir is not None:
    model = ViltForMaskedLMAndITM.from_pretrained(args.model_dir, config=config, ignore_mismatched_sizes=True)
else:
    model = ViltForMaskedLMAndITM.from_pretrained("dandelin/vilt-b32-mlm", config=config, ignore_mismatched_sizes=True)

dataset = MimicCxrDatasetBasic(dataset_path, split="train")
# getting images and texts

image, text, _ = dataset[args.idx]
label = "Related"

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
Image.fromarray(image).save(os.path.join(args.result_dir, "radiography.pdf"))
Image.fromarray(image).save(os.path.join(args.result_dir, "radiography.jpg"))
with open(os.path.join(args.result_dir, "report.txt"), "w") as f:
    f.write(text)

# forwards inputs to the model and applies sigmoid function for texts
def forward_fun(input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                 pixel_values=pixel_values, pixel_mask=pixel_mask)
    return torch.sigmoid(output.itm_logits.amax(-1))

# forwards inputs to the model and applies sigmoid function for images
def forward_fun_for_images(pixel_values: torch.Tensor, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_mask: torch.Tensor):
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                 pixel_values=pixel_values, pixel_mask=pixel_mask)
    return torch.sigmoid(output.itm_logits.amax(-1))

# sums and normalizes attributions
def summarize_attributions(attributions, dim=2):
    attributions = attributions.sum(dim=dim).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

# construct reference token ids 
def construct_ref_id(text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([ref_input_ids])

# construct reference pixel values 
def construct_ref_pixel(pixel_mask, ref_token_id, sep_token_id):
    ref = torch.where(pixel_mask>0, ref_token_id, sep_token_id)
    ref = torch.stack((ref,) * 3, axis=1)
    return ref.type(torch.FloatTensor)

# masks input image
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def smooth_and_normalize(x, discard_ratio=0):
    k = np.percentile(x, 100 * discard_ratio)
    x = np.where(x < k, x, k)
    return (x - x.min()) / (x.max() - x.min())

# compute reference token ids and reference pixel values
ref_token_id = processor.tokenizer.pad_token_id
sep_token_id = processor.tokenizer.sep_token_id
cls_token_id = processor.tokenizer.cls_token_id
ref_input_id = construct_ref_id(text, ref_token_id, sep_token_id, cls_token_id)

lig = LayerIntegratedGradients(forward_fun, model.vilt.embeddings.text_embeddings.word_embeddings)
image_lig = LayerIntegratedGradients(forward_fun_for_images, model.vilt.embeddings.patch_embeddings)

# calculate model results
input = processor(image, text, return_tensors="pt")
ref_pixel_values = construct_ref_pixel(input.pixel_mask, ref_token_id, sep_token_id)
logits = model(**input).itm_logits
pred = logits.argmax(-1).item()
# compute attributions
attributions, delta = lig.attribute(
    inputs=input.input_ids,
    baselines=ref_input_id,
    additional_forward_args=(input.token_type_ids, input.attention_mask, input.pixel_values, input.pixel_mask),
    return_convergence_delta=True
)
attributions_sum = summarize_attributions(attributions)
# create visualizator
data_record = viz.VisualizationDataRecord(
    attributions_sum,
    pred,
    "Related" if pred>0 else "Not related",
    label,
    label,
    attributions_sum.sum(),
    processor.tokenizer.convert_ids_to_tokens(input.input_ids[0].tolist()),
    delta,
    )
# compute patch relevance
attributions, delta = image_lig.attribute(
    inputs=input.pixel_values,
    baselines=ref_pixel_values,
    additional_forward_args=(input.input_ids, input.token_type_ids, input.attention_mask, input.pixel_mask),
    return_convergence_delta=True
)
patch_relevance = summarize_attributions(attributions, 1)
# save masked image
mask = cv2.resize(patch_relevance.cpu().numpy(), (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
mask = smooth_and_normalize(mask, args.drop_rate)
Image.fromarray(show_mask_on_image(image, mask)).save(os.path.join(args.result_dir, "radiography_result.pdf"))
Image.fromarray(show_mask_on_image(image, mask)).save(os.path.join(args.result_dir, "radiography_result.jpg"))

html = get_html([data_record])
data = html.data
with open(os.path.join(args.result_dir, "report_result.html"), "w") as f:
    f.write(data)
