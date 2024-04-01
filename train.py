from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="test", )):

        config = ColBERTConfig(
            bsize=16,
            nway=4,
            use_ib_negatives=True,
            lr=1e-5,
            warmup=100,
        )
        trainer = Trainer(
            # triples="triples.train.colbert.jsonl",
            triples='shuftriplets.jsonl',
            # triples="tinytriples.jsonl",
            queries="queries.train.colbert.tsv",
            collection="corpus.train.colbert.tsv",
            config=config,
        )

        checkpoint_path = trainer.train("bclavie/pile-t5-xl-encoder")

        print(f"Saved checkpoint to {checkpoint_path}...")