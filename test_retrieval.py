import argparse
import random
from typing import List
import torch
from data_collator import ViltDataCollatorForMetricLearning
from dataset import MimicCxrMetricLearningDataset
from model import ViltModelForEmbedding
from transformers import ViltConfig, ViltProcessor, ViltFeatureExtractor, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def data_to_device(data: dict, device: torch.device):
    return {k: v.to(device) for k, v in data.items()}


def rank_at_k(correct_labels: List, sorted_candidate_labels: List[List], Ks: List[int] = [1, 5, 10, 30]):
    found_at_top_k = {k: 0 for k in Ks}
    for label, candidate_labels in zip(correct_labels, sorted_candidate_labels):
        for k in Ks:
            for j in range(int(k)):
                if np.equal(candidate_labels[j], label).any():
                    found_at_top_k[k] = found_at_top_k[k] + 1
                    break
    return {k: count_at_k / len(correct_labels) for k, count_at_k in found_at_top_k.items()}


def main():

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    print(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vilt_processor = ViltProcessor(
        ViltFeatureExtractor(resample=3, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], size_divisor=32),
        AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=args.max_position_embeddings),
    )
    # # config modello
    config = ViltConfig(max_position_embeddings=args.max_position_embeddings)
    model = ViltModelForEmbedding.from_pretrained(args.pretrained_model, config=config, ignore_mismatched_sizes=True)
    model.eval()
    model.to(device)

    # training_dataset = MimicCxrMetricLearningDataset(args.dataset_path, split="train")
    validation_dataset = MimicCxrMetricLearningDataset(args.dataset_path, split="validate")
    data_collator = ViltDataCollatorForMetricLearning(vilt_processor)

    dataloader = DataLoader(
        validation_dataset,
        args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.dataloader_workers,
    )

    # Compute all embeddings for images and descriptions
    reports_embeddings = torch.zeros((len(validation_dataset), config.hidden_size))
    images_embs = torch.zeros((len(validation_dataset), config.hidden_size))
    all_study_ids = torch.zeros((len(validation_dataset)))
    for i, batch in tqdm(enumerate(dataloader)):
        img_inputs, txt_inputs, labels = batch
        with torch.no_grad():
            reports_embeddings[i * args.eval_batch_size : i * args.eval_batch_size + len(labels)] = model(
                **data_to_device(txt_inputs, device)
            )
            images_embs[i * args.eval_batch_size : i * args.eval_batch_size + len(labels)] = model(
                **data_to_device(img_inputs, device)
            )
            all_study_ids[i * args.eval_batch_size : i * args.eval_batch_size + len(labels)] = labels

    # 'txt2img' = description as query; images as documents
    # 'img2txt' = image as query; description as document
    all_queries = reports_embeddings if args.task == "txt2img" else images_embs
    all_documents = images_embs if args.task == "txt2img" else reports_embeddings
    all_indices = set([i for i in range(len(all_documents))])

    # Normalize embeddings
    all_queries = torch.nn.functional.normalize(all_queries, 2, dim=1)
    all_documents = torch.nn.functional.normalize(all_documents, 2, dim=1)

    # Mapping between study_id and indices in the dataset (note that the same study id can occur at at least two indices since the can be twoo images of the same person)
    labels_to_ids = {}
    for i, study_id  in enumerate(all_study_ids):
        labels_to_ids[study_id.cpu().item()] = labels_to_ids.get(study_id.cpu().item(), []) + [i]

    # Extract N random queries
    query_indices = random.sample(range(len(all_queries)), args.num_queries)

    all_query_study_ids = []
    all_sorted_candidated_study_ids = []
    # For each query we generate a candidate set and search its corresponding dataset
    # n.b. we make sure that there is only 1 correct result
    for query_index in tqdm(query_indices):
        query_embedding, query_study_id = (
            all_queries[query_index].unsqueeze(0),
            all_study_ids[query_index].cpu().item(),
        )
        # Get negative candidates making sure that the query is not among them
        query_indices = set(labels_to_ids[query_study_id])
        candidate_indices = all_indices - query_indices
        candidate_indices = random.sample(list(candidate_indices), args.candidate_set_size)
        # The we insert 1 correct target that we want to find
        candidate_indices.append(query_index)
        candidate_embeddings = all_documents[candidate_indices]

        # Compute and sort distances between the query and the candidates embeddings
        distances = torch.cdist(query_embedding, candidate_embeddings).squeeze(0)
        sorted_distance_indices = torch.sort(distances)[1].squeeze(0).long()
        sorted_candidate_indices = np.array(candidate_indices)[sorted_distance_indices]
        sorted_candidate_labels = all_study_ids[sorted_candidate_indices]
        # Store the results so that we can then compute the overall metrics (e.g., precision, recall...)
        all_query_study_ids.append(query_study_id)
        all_sorted_candidated_study_ids.append(sorted_candidate_labels)

    # Compute Rank@K
    print(rank_at_k(all_query_study_ids, all_sorted_candidated_study_ids, args.Ks))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval Test")

    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/datasets/MIMIC-CXR",
        help="Tokenizer used to tokenize texts",
    )
    parser.add_argument(
        "--max_position_embeddings",
        default=512,
        type=int,
        help="Maximum number of position embeddings",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="dandelin/vilt-b32-mlm",
        help="Path to a pretrained model",
    )

    parser.add_argument(
        "--dataloader_workers",
        default=1,
        type=int,
        help="Number of workers to use inside the dataloader",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size used during validation",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="txt2img",
        help="Task to test (txt2img or img2txt)",
    )
    parser.add_argument(
        "--num_queries",
        default=1000,
        type=int,
        help="Numero di query da testare",
    )
    parser.add_argument(
        "--candidate_set_size",
        default=100,
        type=int,
        help="Dimensione del candidate set entro cui generare la query",
    )
    parser.add_argument(
        "--Ks",
        default=[1, 5, 10, 30, 50, 100],
        type=int,
        nargs="+",
        help="Valori di K per il calcolo della Rank@K",
    )

    args = parser.parse_args()
    main()
