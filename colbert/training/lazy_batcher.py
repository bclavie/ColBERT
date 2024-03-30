import os
import ujson

from functools import partial
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar
from colbert.modeling.tokenization import (
    QueryTokenizer,
    DocTokenizer,
    tensorize_triples,
)
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples

# from colbert.utils.runs import Run


class LazyBatcher:
    def __init__(
        self,
        config: ColBERTConfig,
        triples,
        queries,
        collection,
        rank=0,
        nranks=1,
        has_instructions: bool = False,
        instructions_separator: str = "[__ENDINSTR__]",
    ):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config, instruction_model = "google/gemma-2b-it" if has_instructions else None)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(
            tensorize_triples, self.query_tokenizer, self.doc_tokenizer
        )
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

        self.instructions_separator = instructions_separator
        self.has_instructions = has_instructions

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = (
            self.position,
            min(self.position + self.bsize, len(self.triples)),
        )
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            query, *pids = self.triples[position]
            pids = pids[: self.nway]

            query = self.queries[query]

            try:
                pids, scores = zipstar(pids)
            except:
                scores = []

            passages = [self.collection[pid] for pid in pids]

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)

        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        return self.collate(all_queries, all_passages, all_scores)

    def collate(self, queries, passages, scores):
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize
        # print('QUERIES AT COLLATE TIME')
        # print(queries)

        return self.tensorize_triples(
            queries, passages, scores, self.bsize // self.accumsteps, self.nway, has_instructions=self.has_instructions, instruction_token=self.instructions_separator
        )

    # def skip_to_batch(self, batch_idx, intended_batch_size):
    #     Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
    #     self.position = intended_batch_size * batch_idx
