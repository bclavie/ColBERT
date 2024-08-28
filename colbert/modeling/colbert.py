from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT
from colbert.parameters import DEVICE

import torch
import string

import os
import pathlib
from torch.utils.cpp_extension import load
from fast_pytorch_kmeans import KMeans

class ColBERT(BaseColBERT):
    """
        This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        super().__init__(name, colbert_config)
        self.use_gpu = colbert_config.total_visible_gpus > 0

        ColBERT.try_load_torch_extensions(self.use_gpu)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.pad_token = self.raw_tokenizer.pad_token_id


    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(f"Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        # TODO: Organize the code below! Quite messy.
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze

        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)

        nway = self.colbert_config.nway
        all_except_self_negatives = [list(range(qidx*D.size(0), qidx*D.size(0) + nway*qidx+1)) +
                                     list(range(qidx*D.size(0) + nway * (qidx+1), qidx*D.size(0) + D.size(0)))
                                     for qidx in range(Q.size(0))]

        scores = scores[flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1)  # D.size(0) - self.colbert_config.nway + 1)

        labels = torch.arange(0, Q.size(0), device=scores.device) * (self.colbert_config.nway)

        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        out_mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        bert_mask = attention_mask

        D = self.bert(input_ids, attention_mask=bert_mask)[0]

        D = D * out_mask

        # Separate out Special Tokens
        if self.colbert_config.gist_freq != 0:
            special = D[:, :2]
            special_mask = out_mask[:, :2]
            D = D[:, 2:]
            out_mask = out_mask[:, 2:]
            if self.colbert_config.hierarchical_gist_in_training:
                # Store original shapes and lengths
                original_shape = D.shape
                original_lengths = out_mask.sum(dim=1).squeeze(-1)

                # Initialize lists for pooled embeddings and lengths
                D_pooled = []
                pooled_lengths = []

                # Process each document in the batch separately
                for doc_idx in range(D.shape[0]):
                    doc_embeddings = D[doc_idx]
                    doc_mask = out_mask[doc_idx].squeeze(-1)
                    
                    # Filter out padding tokens
                    valid_embeddings = doc_embeddings[doc_mask.bool()]

                    # Calculate number of clusters for this document
                    n_clusters = max(len(valid_embeddings) // self.colbert_config.gist_freq, 1)
                    
                    # Perform KMeans clustering on this document
                    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
                    labels = kmeans.fit_predict(valid_embeddings)
                    
                    # Create pooled embeddings for this document
                    pooled = []
                    for i in range(n_clusters):
                        cluster_embeddings = valid_embeddings[labels == i]
                        if len(cluster_embeddings) > 0:
                            pooled.append(cluster_embeddings.mean(dim=0))
                        else:
                            pooled.append(torch.zeros_like(valid_embeddings[0]))
                    
                    pooled = torch.stack(pooled)
                    D_pooled.append(pooled)
                    pooled_lengths.append(n_clusters)

                # Pad D_pooled to match the maximum number of clusters
                max_length = max(pooled_lengths)
                D_padded = torch.zeros(original_shape[0], max_length, D.size(-1), device=D.device)
                
                for i, pooled in enumerate(D_pooled):
                    D_padded[i, :pooled.size(0)] = pooled

                # Update D and out_mask
                D = D_padded
                out_mask = (D.sum(dim=-1) != 0).float().unsqueeze(-1)
            else:
                weights = self.attn_weight(D)
                D = D.view(D.size(0), -1, self.colbert_config.gist_freq, D.size(-1))

                out_mask = out_mask.view(out_mask.size(0), -1, self.colbert_config.gist_freq)
                num_gists = out_mask.sum(-1)

                weights = weights.view(weights.size(0), -1, self.colbert_config.gist_freq)
                weights.masked_fill_(~out_mask.bool(), -1e4)
                weights = torch.nn.functional.softmax(weights, dim=-1)

                # Weighted average of D over last dimension using weights
                D = D * weights.unsqueeze(-1)
                D = D.sum(-2)

                D = torch.cat([special, D], dim=1)
                out_mask = torch.cat([special_mask, (num_gists > 0).float().unsqueeze(-1)], dim=1)

        D = self.linear(D)
        D = D * out_mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[out_mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, out_mask.bool()
        return D

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'
        if self.colbert_config.similarity == 'l2':
            assert self.colbert_config.interaction == 'colbert'
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer

# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction in ['colbert', 'flipr'], config.interaction

    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

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
        scores_padded, scores_mask = StridedTensor(scores, D_lengths, use_gpu=use_gpu).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        return ColBERT.segmented_maxsim(scores, D_lengths)
