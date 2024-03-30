from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT
from colbert.parameters import DEVICE


# from colbert.modeling.enct5 import EncT5Model, EncT5Tokenizer, EncT5ForSequenceClassification

import torch
import string

import os
import pathlib
from torch.utils.cpp_extension import load

import torch.nn as nn

# TEST_T5_TOKENIZER = EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")
from transformers import AutoModel, AutoTokenizer

import torch
from transformers import T5EncoderModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from typing import List, Optional, Union
import torch
from transformers import PreTrainedModel, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5Model


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")


def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# print("Loading")
# encoder = AutoModel.from_pretrained("Qwen/Qwen1.5-1.8B-Chat").to('cuda')
# tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
# print("loaded")

class ColBERT(BaseColBERT):
    """
    This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name="bert-base-uncased", colbert_config=None, **colbert_kwargs):
        super().__init__(name, colbert_config, **colbert_kwargs)
        self.use_gpu = colbert_config.total_visible_gpus > 0

        ColBERT.try_load_torch_extensions(self.use_gpu)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [
                    symbol,
                    self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0],
                ]
            }
        self.pad_token = self.raw_tokenizer.pad_token_id

    def train(self, mode=True, only_train_instructor=False):
        super().train(mode=mode)
        if only_train_instructor:
            # print(self)
            for param in self.parameters():
                param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = False
            # Assuming instruction related layers are defined as self.instruction_encoder and self.cross_attention
            # This part might need to be adjusted based on the actual implementation of instruction related layers in ColBERT
            for param in self.model.instruction_encoder.parameters():
                param.requires_grad = True
            for param in self.model.cross_attention.parameters():
                param.requires_grad = True
            try:
                for param in self.model.instruction_linear.parameters():
                    param.requires_grad = True
            except AttributeError:
                print('skip isntruction linear')
                pass

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(
            f"Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def forward(self, Q, D):
        # print(Q)
        Q = self.query(*Q)
        # instruction=instruction)
        D, D_mask = self.doc(*D, keep_dims="return_mask")

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        # TODO: Organize the code below! Quite messy.
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(
            0, 1
        )  # query-major unsqueeze

        scores = colbert_score_reduce(
            scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config
        )

        nway = self.colbert_config.nway
        all_except_self_negatives = [
            list(range(qidx * D.size(0), qidx * D.size(0) + nway * qidx + 1))
            + list(
                range(
                    qidx * D.size(0) + nway * (qidx + 1), qidx * D.size(0) + D.size(0)
                )
            )
            for qidx in range(Q.size(0))
        ]

       
        scores = scores[flatten(all_except_self_negatives)]

        scores = scores.view(Q.size(0), -1)  # D.size(0) - self.colbert_config.nway + 1)

        labels = torch.arange(0, Q.size(0), device=scores.device) * (
            self.colbert_config.nway
        )

        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask, instruction_ids=None, instruction_masks=None):
        input_ids, attention_mask = (
            input_ids.to(self.device),
            attention_mask.to(self.device),
        )
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]

        if instruction_ids is not None:
            # print("INSTRUCTIONS")
            # print(instruction)
            # q = "What is the population of Tokyo?"
            # in_answer = "retrieve a passage that answers this question from Wikipedia"

            # p_1 = "The population of Japan's capital, Tokyo, dropped by about 48,600 people to just under 14 million at the start of 2022, the first decline since 1996, the metropolitan government reported Monday."
            # p_2 = "Tokyo, officially the Tokyo Metropolis (東京都, Tōkyō-to), is the capital and largest city of Japan."

            # 1. TART-full can identify more relevant paragraph. 
            # features = tokenizer(['{0} [SEP] {1}'.format(in_answer, q)] * 12, padding=True, truncation=True, return_tensors="pt").to('cuda')          # print(self.model.instruction_encoder.encoder(**instruction))
            # print('DEBERTA ON SEPARATE TOKENIZER')
            # print(encoder(**features))
            # print("DEBERTA MODEL WEIGHTS")
            # for name, param in TEST_T5.named_parameters():
                # if torch.isnan(param.data).any():
                    # print(f"NaN detected in {name}")
            # print("SEE ABOVE")
            # print(instruction_ids)
            # print('____')
            # print(attention_mask)

            # NO QWEN
            full_instruction_embedding = self.model.instruction_encoder(instruction_ids, attention_mask=instruction_masks).last_hidden_state
            
            # we got bidirectionality at home
            half_length = full_instruction_embedding.size(1) // 2
            instruction_embedding = full_instruction_embedding[:, half_length + 1:, :]

            # QWEN
            # instruction_embedding = last_token_pool(instruction_embedding, instruction_masks)


            og_shape = Q.shape
            # print(instruction_embedding.shape)
            Q = self.model.cross_attention(Q, instruction_embedding)
            # print('SHAPE')
            assert Q.shape == og_shape
            Q = self.model.instruction_linear(Q)
        else:
            Q = self.linear(Q)  
        mask = (
            torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device)
            .unsqueeze(2)
            .float()
        )
        Q = Q * mask
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, "return_mask"]

        input_ids, attention_mask = (
            input_ids.to(self.device),
            attention_mask.to(self.device),
        )
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = (
            torch.tensor(
                self.mask(input_ids, skiplist=self.skiplist), device=self.device
            )
            .unsqueeze(2)
            .float()
        )
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == "return_mask":
            return D, mask.bool()

        return D

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'
        if self.colbert_config.similarity == "l2":
            assert self.colbert_config.interaction == "colbert"
            return (
                (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1)) ** 2).sum(-1))
                .max(-1)
                .values.sum(-1)
            )
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [
            [(x not in skiplist) and (x != self.pad_token) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer


# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction in ["colbert", "flipr"], config.interaction

    if config.interaction == "flipr":
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, : config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen :].topk(K2, dim=-1).values.sum(1)

        return A + B

    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
    Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
    If Q.size(0) is 1, the matrix will be compared with all passages.
    Otherwise, each query matrix will be compared against the *aligned* passage.

    EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
    Works with a single query only.
    """

    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    if use_gpu or config.interaction == "flipr":
        scores_padded, scores_mask = StridedTensor(
            scores, D_lengths, use_gpu=use_gpu
        ).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        return ColBERT.segmented_maxsim(scores, D_lengths)
