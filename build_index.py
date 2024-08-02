from colbert.infra import ColBERTConfig
from colbert import Indexer
import srsly
import pandas as pd
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline", help="Path to the experiment file")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    args = parser.parse_args()

    experiments = pd.read_csv("./experiments.csv")
    experiment2path = dict(zip(experiments["name"], experiments["path"]))
    model = experiment2path[args.experiment]

    docs = list(srsly.read_json(os.path.join(args.data_dir, 'scifact_int2doc.json')).values())
    print("Doc sample:")
    print(docs[0])
    print(f"Indexing {len(docs)} documents", flush=True)
    model_name = model.split('experiments/')[1].split('/')[0]
    config = ColBERTConfig(
            nbits=2,
            nranks=1,
            root=".ablation_v1_en/",
            avoid_fork_if_possible=True,
            overwrite=True,
            kmeans_niters=20,
            bsize=32,
            index_bsize=32,
            doc_maxlen=400,
        )
    indexer = Indexer(checkpoint=model, config=config)
    indexer.index(name=f"SCIFACT_500k_{args.experiment}", collection=docs, overwrite=True)
