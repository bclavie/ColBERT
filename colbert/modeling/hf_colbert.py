import importlib
from unicodedata import name
import torch.nn as nn
import transformers
from transformers import (
    BertPreTrainedModel,
    BertModel,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from colbert.utils.utils import torch_load_dnn

import torch
# EncT5
from colbert.modeling.enct5 import EncT5Model, EncT5Tokenizer


class XLMRobertaPreTrainedModel(RobertaPreTrainedModel):
    """
    This class overrides [`RobertaModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    config_class = XLMRobertaConfig


base_class_mapping = {
    "roberta-base": RobertaPreTrainedModel,
    "google/electra-base-discriminator": ElectraPreTrainedModel,
    "xlm-roberta-base": XLMRobertaPreTrainedModel,
    "xlm-roberta-large": XLMRobertaPreTrainedModel,
    "bert-base-uncased": BertPreTrainedModel,
    "bert-large-uncased": BertPreTrainedModel,
    "microsoft/mdeberta-v3-base": DebertaV2PreTrainedModel,
    "bert-base-multilingual-uncased": BertPreTrainedModel,
}

model_object_mapping = {
    "roberta-base": RobertaModel,
    "google/electra-base-discriminator": ElectraModel,
    "xlm-roberta-base": XLMRobertaModel,
    "xlm-roberta-large": XLMRobertaModel,
    "bert-base-uncased": BertModel,
    "bert-large-uncased": BertModel,
    "microsoft/mdeberta-v3-base": DebertaV2Model,
    "bert-base-multilingual-uncased": BertModel,
}


transformers_module = dir(transformers)


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



class CrossAttention(nn.Module):
    def __init__(self, query_dim, instruction_dim):
        super(CrossAttention, self).__init__()
        self.dims_match = True
        if instruction_dim != query_dim:
            self.dims_match = False
            self.instruction_projection = nn.Linear(instruction_dim, query_dim)
        self.attention = nn.MultiheadAttention(query_dim, num_heads=2)

    def forward(self, query_tokens, instruction_embedding):
        if not self.dims_match:
            instruction_embedding = self.instruction_projection(instruction_embedding)
        
        # Adjusting for different sequence lengths
        batch_size, seq_len, _ = query_tokens.size()

        print('INSTR EMB')
        instruction_embedding.size()
        _, instr_seq_len, _ = instruction_embedding.size()
        
        if seq_len != instr_seq_len:
            # Pad query_tokens to match instruction_embedding's seq_len
            if seq_len < instr_seq_len:
                padding = query_tokens.new_zeros(batch_size, instr_seq_len - seq_len, query_tokens.size(2))
                query_tokens = torch.cat([query_tokens, padding], dim=1)
        
        # print(query_tokens)
        # print('BEFORE/AFTER')
        fused_query_representation, _ = self.attention(
            query=query_tokens, key=instruction_embedding, value=instruction_embedding
        )
        
        # Truncate back to original seq_len (32)
        fused_query_representation = fused_query_representation[:, :32, :]
        
        # print(fused_query_representation)
        return fused_query_representation


def find_class_names(model_type, class_type):
    model_type = model_type.replace("-", "").lower()
    for item in transformers_module:
        if model_type + class_type == item.lower():
            return item

    return None


def class_factory(name_or_path, **colbert_kwargs):
    loadedConfig = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)

    if getattr(loadedConfig, "auto_map", None) is None:
        model_type = loadedConfig.model_type
        pretrained_class = find_class_names(model_type, "pretrainedmodel")
        model_class = find_class_names(model_type, "model")

        if pretrained_class is not None:
            pretrained_class_object = getattr(transformers, pretrained_class)
        elif model_type == "xlm-roberta":
            pretrained_class_object = XLMRobertaPreTrainedModel
        elif base_class_mapping.get(name_or_path) is not None:
            pretrained_class_object = base_class_mapping.get(name_or_path)
        else:
            raise ValueError(
                "Could not find correct pretrained class for the model type {model_type} in transformers library"
            )

        if model_class != None:
            model_class_object = getattr(transformers, model_class)
        elif model_object_mapping.get(name_or_path) is not None:
            model_class_object = model_object_mapping.get(name_or_path)
        else:
            raise ValueError(
                "Could not find correct model class for the model type {model_type} in transformers library"
            )
    else:
        assert (
            "AutoModel" in loadedConfig.auto_map
        ), "The custom model should have AutoModel class in the config.automap"
        model_class = loadedConfig.auto_map["AutoModel"]
        assert model_class.endswith("Model")
        pretrained_class = model_class.replace("Model", "PreTrainedModel")
        model_class_object = get_class_from_dynamic_module(model_class, name_or_path)
        pretrained_class_object = get_class_from_dynamic_module(
            pretrained_class, name_or_path
        )

    class HF_ColBERT(pretrained_class_object):
        """
        Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.
        This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
        """

        _keys_to_ignore_on_load_unexpected = [r"cls"]

        def __init__(
            self,
            config,
            colbert_config,
            instruction_model=None,
            freeze_existing_layers=False,
        ):
            super().__init__(config)
            self.config = config
            self.dim = colbert_config.dim
            self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)
            setattr(self, self.base_model_prefix, model_class_object(config))

            print(instruction_model)
            if instruction_model is not None:
                if not hasattr(self, "instruction_encoder"):
                    # self.instruction_encoder = EncT5Model.from_pretrained(
                    #     instruction_model
                    # )
                    # self.instruction_dropout = nn.Dropout(0.1)
                    self.instruction_encoder = AutoModel.from_pretrained("google/gemma-2b-it")
                    self.cross_attention = CrossAttention(
                        config.hidden_size, self.instruction_encoder.config.hidden_size
                    )

                    # Freeze the original linear layer
                    for param in self.linear.parameters():
                        param.requires_grad = False

                    # Create a new linear layer for instructions
                    self.instruction_linear = nn.Linear(
                        config.hidden_size, self.linear.out_features
                    )
                    self.instruction_linear.weight.data.copy_(self.linear.weight.data)
                else:
                    print(
                        "You cannot pass an instruction_model if your model already has a built-in one!"
                    )

            if freeze_existing_layers:
                for param in self.parameters():
                    param.requires_grad = False

                if instruction_model is not None:
                    for param in self.instruction_encoder.parameters():
                        param.requires_grad = True
                    for param in self.cross_attention.parameters():
                        param.requires_grad = True
                    try:
                        for param in self.instruction_linear.parameters():
                            param.requires_grad = True
                    except Exception:
                        pass

            self.init_weights()

        @property
        def LM(self):
            base_model_prefix = getattr(self, "base_model_prefix")
            return getattr(self, base_model_prefix)

        @classmethod
        def from_pretrained(
            cls,
            name_or_path,
            colbert_config,
            instruction_model=None,
            freeze_existing_layers=False,
        ):
            if name_or_path.endswith(".dnn"):
                dnn = torch_load_dnn(name_or_path)
                base = dnn.get("arguments", {}).get("model", "bert-base-uncased")

                obj = super().from_pretrained(
                    base,
                    state_dict=dnn["model_state_dict"],
                    colbert_config=colbert_config,
                    freeze_existing_layers=freeze_existing_layers,
                )
                obj.base = base

                return obj

            obj = super().from_pretrained(
                name_or_path,
                colbert_config=colbert_config,
                freeze_existing_layers=freeze_existing_layers,
                instruction_model=instruction_model,
            )
            obj.base = name_or_path

            if instruction_model is not None:
                if not hasattr(obj, "instruction_linear"):
                    # Create the instruction linear layer if it doesn't exist
                    obj.instruction_linear = nn.Linear(
                        obj.config.hidden_size, obj.linear.out_features
                    )
                    obj.instruction_linear.weight.data.copy_(obj.linear.weight.data)
                    for param in obj.instruction_linear.parameters():
                        param.requires_grad = True
            return obj

        @staticmethod
        def raw_tokenizer_from_pretrained(name_or_path):
            if name_or_path.endswith(".dnn"):
                dnn = torch_load_dnn(name_or_path)
                base = dnn.get("arguments", {}).get("model", "bert-base-uncased")

                obj = AutoTokenizer.from_pretrained(base)
                obj.base = base

                return obj

            obj = AutoTokenizer.from_pretrained(name_or_path)
            obj.base = name_or_path

            return obj

        @staticmethod
        def instruction_tokenizer_from_pretrained(name_or_path):
            # obj = EncT5Tokenizer.from_pretrained(name_or_path)
            obj = AutoTokenizer.from_pretrained("google/gemma-2b-it")
            obj.base = name_or_path

            return obj

    return HF_ColBERT
