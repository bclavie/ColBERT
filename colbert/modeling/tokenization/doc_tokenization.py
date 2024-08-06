import torch

# from transformers import BertTokenizerFast

import numpy as np
from colbert.modeling.hf_colbert import class_factory
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length
from colbert.parameters import DEVICE

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

    def insert_gist_tokens(self, token_ids):
        # First 2 tokens receive global attention ([CLS, D marker])
        # Don't need to be "GISTed"
        result = []
        for i, token in enumerate(token_ids, 1):
            result.append(token)
            # GIST freq refers to how many tokens in each GIST -- need to add 1 to make the math math
            if i > 1 and i % self.gist_freq == 0:
                result.append(self.gist_token)
        return result

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False).to(DEVICE) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False).to(DEVICE)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        batch_size = len(batch_text)

        # add placehold for the [D] marker
        batch_text = ['. ' + x for x in batch_text]

        batch_tokens = [
            self.tok.tokenize(x, add_special_tokens=False) for x in batch_text
        ]

        if self.gist_freq > 1:
            batch_tokens = [tokens[:1] + self.insert_gist_tokens(tokens[1:]) for tokens in batch_tokens]

        batch_tokens = [
            ['[CLS]'] + t[:min(len(t), self.doc_maxlen - 2)] + ['[SEP]'] for t in batch_tokens
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
            num_global = 2  # [CLS] and D_marker_token_id

            mask = torch.ones((batch_size, max_seq_len, max_seq_len), dtype=torch.int64).to(DEVICE)
            gist_idxs = torch.arange(num_global + self.gist_freq, max_seq_len, self.gist_freq + 1)

            last_gist_idx = num_global - 1
            for idx in gist_idxs:
                # Tokens cannot see GISTs
                mask[:, num_global:, idx] = 0

                # GIST token can see itself
                mask[:, idx, idx] = 1

                # GIST tokens should not see tokens at or before previous GIST
                mask[:, idx, :last_gist_idx + 1] = 0
                # Or after itself
                mask[:, idx, idx + 1:] = 0

                # Update last gist idx to current
                last_gist_idx = idx
            
            # All tokens can see global tokens
            mask[:, :, :num_global] = 1
            # Global tokens can see all tokens
            mask[:, :num_global, :] = 1

            for i in range(batch_size):
                # Ignore padded tokens
                mask[i, seq_lens[i]:, :] = 0
                mask[i, :, seq_lens[i]:] = 0

                # Every token can see [SEP]
                mask[i, :, seq_lens[i] - 1] = 1
                # [SEP] can see every token
                mask[i, seq_lens[i] - 1, :] = 1

            # To avoid passing in 2 masks, let's use the first position to indicate which are tokens we are indexing
            # Set these to 2
            # We are indexing global tokens
            mask[:, 0, :num_global] = 2
            mask[:, 0, gist_idxs] = 2
            for i in range(batch_size):
                # We are keeping [SEP]
                mask[i, 0, seq_lens[i] - 1] = 2
                # Ignore padded tokens
                mask[i, 0, seq_lens[i]:] = 0
        else:
            mask = torch.ones((batch_size, max_seq_len), dtype=torch.int64).to(DEVICE)
            for i in range(batch_size):
                mask[i, seq_lens[i]:] = 0

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask
