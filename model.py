import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple
from transformers.utils import ModelOutput
from dataclasses import dataclass
from transformers import ViltPreTrainedModel, ViltModel
from transformers.models.vilt.modeling_vilt import ViltMLMHead
from transformers.utils import logging
import wandb
import modeling_vilt

logger = logging.get_logger(__name__)


@dataclass
class MaskedLMAndITMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    itm_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    itm_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ITMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ViltForMaskedLMAndITM(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)
        self.mlm_score = ViltMLMHead(config)
        self.itm_score = nn.Linear(config.hidden_size, 2)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        head_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        labels=None,
        next_sentence_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[1]

        sequence_output, pooled_output = outputs[:2]
        # split up final hidden states into text and image features
        text_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        text_features, _ = (sequence_output[:, :text_seq_len], sequence_output[:, text_seq_len:])

        mlm_logits = self.mlm_score(text_features)
        itm_logits = self.itm_score(pooler_output)

        total_loss = None
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # workaround to log in wandb
            wandb.log({"train/mlm_loss": masked_lm_loss})
            total_loss = masked_lm_loss

        itm_loss = None
        if next_sentence_labels is not None:
            loss_fct = CrossEntropyLoss()
            itm_loss = loss_fct(itm_logits.view(-1, 2), next_sentence_labels.view(-1))
            # workaround to log in wandb
            wandb.log({"train/itm_loss": itm_loss})
            total_loss = total_loss + itm_loss if total_loss is not None else itm_loss

        if not return_dict:
            output = (mlm_logits, itm_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MaskedLMAndITMOutput(
            loss=total_loss,
            mlm_loss=masked_lm_loss,
            itm_loss=itm_loss,
            mlm_logits=mlm_logits,
            itm_logits=itm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ViltForImageTextMatching(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)
        self.itm_score = nn.Linear(config.hidden_size, 2)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        head_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        next_sentence_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.itm_score(pooler_output)

        loss = None
        if next_sentence_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), next_sentence_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ITMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ViltModelForEmbedding(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vilt = modeling_vilt.CustomViltModel(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        head_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs.pooler_output
