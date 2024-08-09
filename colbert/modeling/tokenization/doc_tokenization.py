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

        # add placehold for the [D] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(
            batch_text, padding='longest', truncation='longest_first',
            return_tensors='pt', max_length=self.doc_maxlen,
            pad_to_multiple_of=self.gist_freq
        ).to(DEVICE)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        # Add "2" to account for the [CLS] and [SEP] tokens.
        mask = torch.cat([mask, torch.zeros(mask.size(0), 2, device=DEVICE)], dim=1)
        pad_ids = torch.zeros(ids.size(0), 2, dtype=torch.long, device=DEVICE)
        pad_ids.fill_(self.tok.pad_token_id)
        ids = torch.cat([ids, pad_ids], dim=1)

        if bsize:
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask