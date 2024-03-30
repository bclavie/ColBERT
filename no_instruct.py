from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=2, experiment="test")):

        config = ColBERTConfig(
            bsize=64,
            nway=4,
            use_ib_negatives=True,
            lr=2e-5,
            warmup=100,
        )
        trainer = Trainer(
            # triples="triples.train.colbert.jsonl",
            triples='shuftriplets.jsonl',
            queries="queries.train.colbert.tsv",
            collection="corpus.train.colbert.tsv",
            config=config,
            instruction_model=None,
            # instruction_model="google/gemma-2b-it",
        )

        checkpoint_path = trainer.train('colbert-ir/colbertv2.0')

        print(f"Saved checkpoint to {checkpoint_path}...")

        