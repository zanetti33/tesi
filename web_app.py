import streamlit as st
import numpy as np
import argparse
import random
import time
from typing import List
import torch
from data_collator import ViltDataCollatorForMetricLearning
from dataset import MimicCxrMetricLearningDataset
from model import ViltModelForEmbedding
import tokenizers
from transformers import ViltConfig, ViltProcessor, ViltFeatureExtractor, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# PARAMS
@st.cache
def load_args():
    parser = argparse.ArgumentParser(description="Retrieval Web App")
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--results_per_page",
        default=10,
        type=int,
        help="Number of items per page",
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
    return args


# MODEL PREPARATION
def data_to_device(data: dict, device: torch.device):
    return {k: v.to(device) for k, v in data.items()}

@st.cache(allow_output_mutation=True)
def load_data(args):
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    vilt_processor = ViltProcessor(
        ViltFeatureExtractor(resample=3, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], size_divisor=32),
        AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=args.max_position_embeddings),
    )
    # # config modello
    config = ViltConfig(max_position_embeddings=args.max_position_embeddings)
    model = ViltModelForEmbedding.from_pretrained(args.pretrained_model, config=config, ignore_mismatched_sizes=True)
    model.eval()
    model.to(device)

    # dataset = MimicCxrMetricLearningDataset(args.dataset_path, split="train")
    # train, test, validate
    dataset = MimicCxrMetricLearningDataset(args.dataset_path, split="validate")
    print(len(dataset))
    data_collator = ViltDataCollatorForMetricLearning(vilt_processor)

    dataloader = DataLoader(
        dataset,
        args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.dataloader_workers,
    )

    # Compute all embeddings for images and descriptions
    reports_embeddings = torch.zeros((len(dataset), config.hidden_size))
    images_embs = torch.zeros((len(dataset), config.hidden_size))
    all_study_ids = torch.zeros((len(dataset)))
    for i, batch in enumerate(dataloader):
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
    return model, vilt_processor, device, all_queries, all_documents, all_indices, all_study_ids, labels_to_ids, dataset



# WEB PAGE
@st.cache
def execute_query(query_index):
    # For this id we generate a candidate set and search its corresponding dataset
    query_embedding, query_study_id = (
        all_queries[query_index].unsqueeze(0),
        all_study_ids[query_index].cpu().item(),
    )
    # Get negative candidates making sure that the query is not among them
    query_indices = set(labels_to_ids[query_study_id])
    candidate_indices = all_indices - query_indices
    candidate_indices = random.sample(list(candidate_indices), args.candidate_set_size - 1)
    # Then we insert 1 correct target that we want to find
    candidate_indices.append(query_index)
    candidate_embeddings = all_documents[candidate_indices]

    # Compute and sort distances between the query and the candidates embeddings
    distances = torch.cdist(query_embedding, candidate_embeddings).squeeze(0)
    sorted_distance_indices = torch.sort(distances)[1].squeeze(0).long()
    return np.array(candidate_indices)[sorted_distance_indices]



@st.cache(hash_funcs={tokenizers.Tokenizer: hash})
def free_text_query(text):
    text_encoding = vilt_processor.tokenizer(
            text, padding="max_length", truncation=True, return_tensors="pt"
        )
    with torch.no_grad():
            query_embedding = model(
                **data_to_device(dict(text_encoding), device)
            )
    # Compute and sort distances between the query and the all embeddings
    query_embedding = torch.nn.functional.normalize(query_embedding, 2).cpu()
    starting_time = time.time()
    distances = torch.cdist(query_embedding, all_documents[list(all_indices)]).squeeze(0)
    elapsed_time = time.time() - starting_time
    sorted_distance_indices = torch.sort(distances)[1].squeeze(0).long()
    return sorted_distance_indices.tolist()[:100], elapsed_time

st.set_page_config(
    page_title='Retrieval di radiografie toraciche',
    layout='wide',
)

st.title('Retrieval di radiografie toraciche')

data_load_state = st.text('Caricamento argomenti...')
args = load_args()
data_load_state.text('Caricamento argomenti...fatto!')

data_load_state = st.text('Caricamento del modello e dei dati...')
model, vilt_processor, device, all_queries, all_documents, all_indices, all_study_ids, labels_to_ids, dataset = load_data(args)
data_load_state.text('Caricamento del modello e dei dati...fatto!')

free_search_tab, example_tab = st.tabs(['Ricerca con testo libero', 'Ricerca su esempi'])

st.markdown(
    """
    <style>
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            outline: 5px solid green;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Description
example_tab.text(" Questo applicativo mostra le capacità di retrieval del nostro modello in ambito biomedico. \n Selezionando un dato referto, l'algoritmo classificherà 100 radiografie (99 casuali più quella corrispondende al referto) in ordine di similarità. \n Fatto questo, la radiografia corretta verrà poi evidenziata in verde, in modo da visualizzare se è stata classificata correttamente (o se almeno risulta tra i primi risultati).")
# Get the element to query on
col, _ = example_tab.columns([1,4])
query_index = col.selectbox('Scegli un esempio', all_indices)
# Start the query
if (example_tab.button('Cerca', 'query')):
    sorted_candidate_indices = execute_query(query_index)
    query_image, query_text, _ = dataset[query_index]
    # display query
    txt_col, _, img_col, _ = example_tab.columns([2,1,2,1])
    txt_col.markdown('### Referto (input del modello)')
    txt_col.write(query_text)
    img_col.markdown('### Radiografia corretta')
    img_col.image(query_image)
    # display results
    example_tab.markdown('# ')
    example_tab.markdown('# ')
    example_tab.markdown('# ')
    example_tab.markdown('##### Elenco dei risultati ordinati per similarità')
    tab_num = len(sorted_candidate_indices) // args.results_per_page
    tab_names = []
    for i in range(tab_num):
        tab_names.append(str(i+1))
    tabs = example_tab.tabs(tab_names)
    img_cols = []
    col1, col2, col3, col4, col5 = tabs[0].columns(5)
    img_cols.append(col5)
    img_cols.append(col4)
    img_cols.append(col3)
    img_cols.append(col2)
    img_cols.append(col1)
    desc_cols = []
    col1, col2, col3, col4, col5 = tabs[0].columns(5)
    desc_cols.append(col5)
    desc_cols.append(col4)
    desc_cols.append(col3)
    desc_cols.append(col2)
    desc_cols.append(col1)
    for i, idx in enumerate(sorted_candidate_indices):
        if len(img_cols) == 0:
            col1, col2, col3, col4, col5 = tabs[(i+1) // args.results_per_page].columns(5)
            img_cols.append(col5)
            img_cols.append(col4)
            img_cols.append(col3)
            img_cols.append(col2)
            img_cols.append(col1)
            col1, col2, col3, col4, col5 = tabs[(i+1) // args.results_per_page].columns(5)
            desc_cols.append(col5)
            desc_cols.append(col4)
            desc_cols.append(col3)
            desc_cols.append(col2)
            desc_cols.append(col1)
        image, text, _ = dataset[idx]
        if idx == query_index:
            container = img_cols.pop().container()
            container.image(image)
        else:
            img_cols.pop().image(image)
        desc_space = desc_cols.pop()
        desc_space.write(str(idx))
        expander = desc_space.expander('Vedi referto')
        expander.write(text)

# Description
free_search_tab.text(" Questo applicativo mostra le capacità di retrieval su testo libero del nostro modello in ambito biomedico. \n Inserendo qualsiasi testo l'algoritmo classificherà tutte le radiografie del dataset in ordine di similarità. ")
# Get the element to query on
text_query = free_search_tab.text_area('Ricerca')
# Start the query
if (free_search_tab.button('Cerca', 'free text')):
    sorted_candidate_indices, elapsed_time = free_text_query(text_query)
    # display results
    free_search_tab.markdown('# ')
    free_search_tab.markdown('# ')
    free_search_tab.markdown('# ')
    free_search_tab.markdown(f'##### Elenco dei risultati ordinati per similarità (calcolati in {elapsed_time:.3f}s)')
    tab_num = len(sorted_candidate_indices) // args.results_per_page
    tab_names = []
    for i in range(tab_num):
        tab_names.append(str(i+1))
    tabs = free_search_tab.tabs(tab_names)
    img_cols = []
    col1, col2, col3, col4, col5 = tabs[0].columns(5)
    img_cols.append(col5)
    img_cols.append(col4)
    img_cols.append(col3)
    img_cols.append(col2)
    img_cols.append(col1)
    desc_cols = []
    col1, col2, col3, col4, col5 = tabs[0].columns(5)
    desc_cols.append(col5)
    desc_cols.append(col4)
    desc_cols.append(col3)
    desc_cols.append(col2)
    desc_cols.append(col1)
    for i, idx in enumerate(sorted_candidate_indices):
        if len(img_cols) == 0:
            col1, col2, col3, col4, col5 = tabs[(i+1) // args.results_per_page].columns(5)
            img_cols.append(col5)
            img_cols.append(col4)
            img_cols.append(col3)
            img_cols.append(col2)
            img_cols.append(col1)
            col1, col2, col3, col4, col5 = tabs[(i+1) // args.results_per_page].columns(5)
            desc_cols.append(col5)
            desc_cols.append(col4)
            desc_cols.append(col3)
            desc_cols.append(col2)
            desc_cols.append(col1)
        image, text, _ = dataset[idx]
        img_cols.pop().image(image)
        desc_space = desc_cols.pop()
        desc_space.write(str(idx))
        expander = desc_space.expander('Vedi referto')
        expander.write(text)
