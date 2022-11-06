from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
)
import torch
from dataclasses import dataclass
from typing import List, Union, Dict
from transformers import ViltProcessor


@dataclass
class ViltDataCollatorForPretraining:
    processor: ViltProcessor
    mlm_datacollator: DataCollatorForLanguageModeling

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples, list):
            texts = []
            images = []
            for e in examples:
                texts.extend(e["texts"])
                images.extend(e["images"])
        else:
            texts = examples["texts"]
            images = examples["images"]
        encoding = self.processor.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")
        masked = self.mlm_datacollator.torch_mask_tokens(torch.Tensor(encoding["input_ids"]).long())
        encoding["input_ids"] = masked[0]
        encoding["labels"] = masked[1]
        encoding_feature_extractor = self.processor.feature_extractor(images, return_tensors="pt")
        encoding.update(encoding_feature_extractor)
        if isinstance(examples, list):
            next_sentence_labels = []
            for e in examples:
                next_sentence_labels.append(e["next_sentence_labels"])
            labels_encoding = {"next_sentence_labels": torch.LongTensor(next_sentence_labels)}
        else:
            labels_encoding = {"next_sentence_labels": torch.stack(examples["next_sentence_labels"], dim=1)}
        encoding.update(labels_encoding)
        return encoding


class ViltDataCollatorForMetricLearning:
    def __init__(self, preprocessor: ViltProcessor) -> None:
        self.processor = preprocessor

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        text_encoding = self.processor.tokenizer(
            [e[1] for e in examples], padding="max_length", truncation=True, return_tensors="pt"
        )
        image_encoding = self.processor.feature_extractor([e[0] for e in examples], return_tensors="pt")
        labels = torch.Tensor([e[2] for e in examples])
        # we return dictionaries so that we can easily use them through the pytorch-metric-learning APIs
        # (BatchFeatures and BatchEncoding classes are huggingface specific)
        return dict(image_encoding), dict(text_encoding), labels
