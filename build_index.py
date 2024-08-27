from colbert.infra import ColBERTConfig
from colbert import Indexer
import srsly
import pandas as pd
import argparse
import os
import datasets


def get_data(data_dir, dataset):
    if dataset == "scifact":
        docs = list(srsly.read_json(os.path.join(data_dir, 'scifact_int2doc.json')).values())
        int2docid = {int(k): v for k, v in srsly.read_json(os.path.join(data_dir, 'scifact_int2docid.json')).items()}
    elif dataset == "trec-covid":
        dataset_name = 'beir/trec-covid/test'
        docs = list(srsly.read_json(os.path.join(data_dir, 'trec-covid_int2doc.json')).values())
        int2docid = {int(k): v for k, v in srsly.read_json(os.path.join(data_dir, 'trec-covid_int2docid.json')).items()}
    elif dataset == "litsearch":
        dataset_name = 'princeton-nlp/LitSearch'
        ds = datasets.load_dataset(dataset_name, 'corpus_clean')
        int2docid = {}
        docs = []
        for i, entry in enumerate(ds['full']):
            int2docid = {i: entry['corpusid']}
            docs += [entry['title'] + ' ' + entry['abstract']]
    return docs, int2docid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline", help="Path to the experiment file")
    parser.add_argument("--data_dir", type=str, default="./gist_data", help="Path to the data directory")
    parser.add_argument("--datasets", nargs='+', default=["trec-covid", "litsearch", "scifact"], help="Name of datasets to test")
    args = parser.parse_args()

    experiments = pd.read_csv("./experiments.csv")
    experiment2path = dict(zip(experiments["name"], experiments["path"]))
    model = os.path.expanduser(experiment2path[args.experiment])

    model_name = model.split('experiments/')[1].split('/')[0]
    config = ColBERTConfig(
            nbits=2,
            nranks=1,
            root=".ablation_v1_en/",
            avoid_fork_if_possible=True,
            overwrite=True,
            kmeans_niters=4,
            bsize=1024,
            index_bsize=1024,
            doc_maxlen=500,
        )
    indexer = Indexer(checkpoint=model, config=config)

    for dataset in args.datasets:
        docs, _ = get_data(args.data_dir, dataset)
        print("Doc sample:")
        print(docs[0])
        print(f"Indexing {len(docs)} documents", flush=True)
        indexer.index(name=f"{dataset}_{args.experiment}", collection=docs, overwrite=True)
