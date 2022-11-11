import torch
import pandas as pd
import random
import os
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
import numpy as np


class MimicCxrDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, path: str, split: str = "train"):
        self.path = path
        self.error = True

        self.file_path = os.path.join(self.path, "files")

        self.dataframe = pd.read_csv(os.path.join(self.path, "random_sampled.csv"), index_col=0).sort_index()
        self.dataframe = self.dataframe[self.dataframe["split"] == split]

        # leggo le label assegnate ai vari record
        self.labels_dataframe = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-negbio.csv"))
        self.labels_dataframe = self.labels_dataframe.loc[
            self.labels_dataframe["study_id"].isin(self.dataframe["study_id"])
        ]

    def _get_text(self, idx) -> str:
        self.__check_valid_index(idx)
        text_row = self.dataframe.iloc[idx]
        txt_path = os.path.join(
            self.file_path,
            f"p{str(text_row.subject_id)[:2]}",
            f"p{text_row.subject_id}",
            f"s{text_row.study_id}.txt",
        )
        with open(txt_path, "r") as f:
            text = f.read()
        return text

    def _get_image(self, idx) -> np.array:
        self.__check_valid_index(idx)
        image_row = self.dataframe.iloc[idx]
        image_path = os.path.join(
            self.file_path,
            f"p{str(image_row.subject_id)[:2]}",
            f"p{image_row.subject_id}",
            f"s{image_row.study_id}",
            f"{image_row.dicom_id}.jpg",
        )
        image = Image.open(image_path)
        image = np.array(image)
        image = np.stack((image,) * 3, axis=-1)
        return image

    def get_labels(self, idx) -> np.array:
        study_id = self.get_study_id(idx)
        return self.labels_dataframe.loc[self.labels_dataframe.study_id == study_id].iloc[0, 2:].values

    def get_study_id(self, idx) -> int:
        return self.dataframe.iloc[idx].study_id

    def __check_valid_index(self, idx) -> None:
        assert idx < len(self.dataframe), f"Index out of range ({idx} > {len(self.dataframe)-1})"

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class MimicCxrPretrainingDataset(MimicCxrDataset):
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
                labels_to_check = (
                    self.labels_dataframe.loc[self.labels_dataframe["study_id"] == text_row.study_id] == 1.0
                )
                searching = True
                while searching:
                    image_index = random.randint(0, len(self.dataframe) - 1)
                    image_row = self.dataframe.iloc[image_index]
                    image_labels = self.labels_dataframe.loc[self.labels_dataframe["study_id"] == image_row.study_id]
                    searching = (
                        len(labels_to_check.index) > 0
                        and (image_labels[labels_to_check].squeeze() == 1.0).any(axis=None)
                    ) or text_row.study_id == image_row.study_id
                    if searching:
                        self.error = False
            else:
                text_index = i
                image_index = i
                # text_row = self.dataframe.iloc[i]
                # image_row = self.dataframe.iloc[i]

            texts.append(self._get_text(text_index))
            images.append(self._get_image(image_index))
            labels.append(int(i < len(self.dataframe)))
        return {"texts": texts, "images": images, "next_sentence_labels": labels}

    def __len__(self):
        return 2 * len(self.dataframe)

    def checkerror(self):
        return self.error


class MimicCxrMetricLearningDataset(MimicCxrDataset):
    def __getitem__(self, idx):
        text = self._get_text(idx)
        image = self._get_image(idx)
        study_id = self.get_study_id(idx)
        return image, text, study_id

    def __len__(self):
        return len(self.dataframe)
