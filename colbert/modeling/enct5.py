"""Code from https://github.com/facebookresearch/tart/blob/main/TART/src/modeling_enc_t5.py and https://github.com/facebookresearch/tart/blob/main/TART/src/tokenization_enc_t5.py
Copyright from the original code: # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Used under the Attribution-NonCommercial 4.0 International license kindly provided by Meta.
Changes are made to EncT5Model to remove the sequence classification aspect.
No changes are made to EncT5Tokenizer."""

from typing import Any, Dict, List, Optional

from transformers import T5Tokenizer


import copy

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'



class EncT5ForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config, dropout=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    # def parallelize(self, device_map=None):
    #     self.device_map = (
    #         get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
    #         if device_map is None
    #         else device_map
    #     )
    #     assert_device_map(self.device_map, len(self.encoder.block))
    #     self.encoder.parallelize(self.device_map)
    #     self.classifier = self.classifier.to(self.encoder.first_device)
    #     self.model_parallel = True

    # def deparallelize(self):
    #     self.encoder.deparallelize()
    #     self.encoder = self.encoder.to("cpu")
    #     self.model_parallel = False
    #     self.device_map = None
    #     torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_output = hidden_states[:, 0, :]  # Take bos token (equiv. to <s>)

        pooled_output = self.dropout(pooled_output)
        return pooled_output

class EncT5Model(T5PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.dropout = nn.Dropout(0.1)
        self.encoder = T5Stack(encoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        print(hidden_states)
        pooled_output = hidden_states[:, 0, :]  # Take bos token (equiv. to <s>)

        pooled_output = self.dropout(pooled_output)
        
        return pooled_output

class EncT5Tokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=sp_model_kwargs,
            **kwargs,
        )

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            return (
                [self.bos_token_id]
                + token_ids_0
                + [self.eos_token_id]
                + token_ids_1
                + [self.eos_token_id]
            )
