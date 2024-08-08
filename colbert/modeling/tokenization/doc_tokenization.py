import torch

# from transformers import BertTokenizerFast

import numpy as np
from colbert.modeling.hf_colbert import class_factory
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length
from colbert.parameters import DEVICE
import math


class DocTokenizer():
    def __init__(self, config: ColBERTConfig):
        HF_ColBERT = class_factory(config.checkpoint)
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)

        self.config = config
        self.doc_maxlen = config.doc_maxlen

        self.D_marker_token, self.D_marker_token_id = self.config.doc_token, self.tok.convert_tokens_to_ids(self.config.doc_token_id)
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

        self.gist_token = "[unused0]"
        self.gist_token_id = self.tok.get_vocab()[self.gist_token]
        self.gist_freq = config.gist_freq

    def insert_gist_tokens(self, token_ids, max_seq_len):
        gist_frac = 1 / self.gist_freq
        gist_num = max(1, math.ceil(min(max_seq_len, len(token_ids)) * gist_frac))
        return [self.gist_token] * gist_num + token_ids

    def gist_windows(self, seq_len, num_windows, window_size: int=8):
        if num_windows == 1:
            return [list(range(seq_len))]
        # First GIST is 0-centered
        start = - window_size // 2

        # Last GIST will be centered at end of sequence
        stepsize = seq_len / (num_windows - 1)
        windows = []
        idxs = set()
        for i in range(num_windows):
            b = round(max(0, start + stepsize * i))
            e = round(min(seq_len, start + stepsize * i + window_size))
            windows.append(list(range(b, e)))
            for idx in range(b, e):
                idxs.add(idx)

        idxs = sorted(list(idxs))
        assert idxs == list(range(seq_len))
        return windows

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False).to(DEVICE) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        batch_size = len(batch_text)

        # [CLS], [D], [SEP]
        num_special = 3
        non_special_max_len = self.doc_maxlen - num_special

        batch_tokens = [
            self.tok.tokenize(x, add_special_tokens=False) for x in batch_text
        ]

        if self.gist_freq > 1:
            # 3 special, non-gistable tokens
            # GIST budget is based on min(len(tokens) - 1, doc_maxlen - 3) 
            batch_tokens = [self.insert_gist_tokens(
                tokens, non_special_max_len
            ) for tokens in batch_tokens]

        batch_tokens = [
            ['[CLS]', '.'] + t[:min(len(t), non_special_max_len)] + ['[SEP]'] for t in batch_tokens
        ]

        ids = [self.tok.convert_tokens_to_ids(t) for t in batch_tokens]

        seq_lens = [len(t) for t in ids]
        max_seq_len = max(seq_lens)

        for i in range(len(ids)):
            n = len(ids[i])
            pad_n = max_seq_len - n
            ids[i][1] = self.D_marker_token_id
            ids[i] += [self.tok.pad_token_id] * pad_n

        ids = torch.tensor(ids, dtype=torch.int64).to(DEVICE)

        if self.gist_freq > 1:
            num_global = num_special - 1  # [CLS] and D_marker_token_id (exclude [SEP] which is at the end)

            mask = torch.ones((batch_size, max_seq_len, max_seq_len), dtype=torch.int64).to(DEVICE)

            for i in range(batch_size):
                # GIST idxs
                gist_idxs = torch.where(ids[i] == self.gist_token_id)[0]
                num_gist = len(gist_idxs)
                assert num_gist > 0, "No GIST tokens found in the input"
                token_seq_len = seq_lens[i] - num_special - num_gist
                gist_windows = self.gist_windows(token_seq_len, num_gist)
                first_tok_idx = num_global + num_gist  # SEP is at end (so its 2)

                assert ids[i, first_tok_idx] != self.gist_token_id
                assert ids[i, first_tok_idx - 1] == self.gist_token_id

                for gist_idx, window in zip(gist_idxs, gist_windows):
                    mask[i, gist_idx, :] = 0
                    mask[i, gist_idx, [w + first_tok_idx for w in window]] = 1

                    # GISTS can see special tokens
                    mask[i, gist_idx, :2] = 1
                    mask[i, gist_idx, seq_lens[i] - 1] = 1

                    # Tokens can't see GISTS
                    mask[i, first_tok_idx:, gist_idx] = 0

                    # Ignore padded tokens
                    mask[i, seq_lens[i]:, :] = 0
                    mask[i, :, seq_lens[i]:] = 0

                    # Mark tokens we are keeping with 2 for mask
                    mask[i, 0, :first_tok_idx] = 2
                    mask[i, 0, seq_lens[i] - 1] = 2
        else:
            mask = torch.ones((batch_size, max_seq_len), dtype=torch.int64).to(DEVICE)
            for i in range(batch_size):
                mask[i, seq_lens[i]:] = 0

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask
