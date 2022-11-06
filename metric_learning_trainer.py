from pytorch_metric_learning.trainers.twostream_metric_loss import TwoStreamMetricLoss
import torch
from pytorch_metric_learning.testers import GlobalTwoStreamEmbeddingSpaceTester
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import tqdm
from pytorch_metric_learning.utils import accuracy_calculator
from transformers.feature_extraction_utils import BatchFeature


class TwoStreamMetricLearningTrainer(TwoStreamMetricLoss):
    def calculate_loss(self, curr_batch):
        (img, txt), labels = curr_batch
        embeddings = (
            self.compute_embeddings(img),
            self.compute_embeddings(txt),
        )
        all_labels = torch.cat([labels, labels], dim=0)
        all_embeddings = torch.cat(embeddings, dim=0)
        indices_tuple = self.maybe_mine_embeddings(all_embeddings, all_labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(all_embeddings, all_labels, indices_tuple)

    def maybe_get_metric_loss(self, embeddings, labels, indices_tuple):
        if self.loss_weights.get("metric_loss", 0) > 0:
            return self.loss_funcs["metric_loss"](embeddings, labels, indices_tuple)
        return 0

    def maybe_mine_embeddings(self, embeddings, labels):
        # for both get_all_triplets_indices and mining_funcs
        # we need to clone labels and pass them as ref_labels
        # to ensure triplets are generated between anchors and posnegs
        if "tuple_miner" in self.mining_funcs:
            return self.mining_funcs["tuple_miner"](embeddings, labels)
        else:
            return lmu.get_all_triplets_indices(labels, labels.clone())

    def compute_embeddings(self, data, **kwargs):
        trunk_output = self.get_trunk_output(data, **kwargs)
        embeddings = self.get_final_embeddings(trunk_output, **kwargs)
        return embeddings

    def get_final_embeddings(self, base_output, **kwargs):
        return self.models["embedder"](base_output, **kwargs)

    def get_trunk_output(self, data):
        if isinstance(data, dict) or isinstance(data, BatchFeature):
            for k, v in data.items():
                data[k] = c_f.to_device(v, device=self.data_device, dtype=self.dtype)
            return self.models["trunk"](**data)
        else:
            data = c_f.to_device(data, device=self.data_device, dtype=self.dtype)
            return self.models["trunk"](data)


class TwoStreamMetricLearningTester(GlobalTwoStreamEmbeddingSpaceTester):
    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                img, txt, label = self.data_and_label_getter(data)
                label = c_f.process_label(label, self.label_hierarchy_level, self.label_mapper)
                a = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
                pns = self.get_embeddings_for_eval(trunk_model, embedder_model, txt)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(len(dataloader.dataset), label.size(1))
                    all_anchors = torch.zeros(len(dataloader.dataset), pns.size(1))
                    all_posnegs = torch.zeros(len(dataloader.dataset), pns.size(1))

                e = s + pns.size(0)
                all_anchors[s:e] = a
                all_posnegs[s:e] = pns
                labels[s:e] = label
                s = e
        return all_anchors, all_posnegs, labels

    def get_embeddings_for_eval(self, trunk_model, embedder_model, input):
        if isinstance(input, dict):
            for k, v in input.items():
                input[k] = c_f.to_device(v, device=self.data_device, dtype=self.dtype)
            trunk_output = trunk_model(**input)
        else:
            input = c_f.to_device(input, device=self.data_device, dtype=self.dtype)
            trunk_output = trunk_model(input)

        if self.use_trunk_output:
            return trunk_output
        return embedder_model(trunk_output)
