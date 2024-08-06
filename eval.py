import ir_datasets
from ranx import Qrels, evaluate
import srsly
from collections import defaultdict
from colbert.infra import ColBERTConfig
from colbert import Searcher
from tqdm import tqdm
import os
from ranx import Run as ranx_run
import pandas as pd
import datasets
import argparse


def load_data(data_dir, dataset, **kwargs):
    if dataset == "scifact":
        dataset = ir_datasets.load("beir/scifact/test")
        all_queries = {}
        for q in dataset.queries_iter():
            all_queries[q.query_id] = q.text

        queries = {id: query for id, query in all_queries.items()}

        print(f"Found {len(queries)} queries")
        print(queries)

        qrels_dict = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            if qrel.query_id in queries:
                qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

        print("Loading documents...")
        int2docid = {int(k): v for k, v in srsly.read_json(os.path.join(data_dir, 'scifact_int2docid.json')).items()}

        return int2docid, queries, qrels_dict
    elif dataset == "litsearch":
        print("Loading dataset...")
        dataset_name = 'princeton-nlp/LitSearch'
        ds = datasets.load_dataset(dataset_name, 'query')
        
        print("Loading queries...")

        specific_qrels_dict = defaultdict(dict)
        broad_qrels_dict = defaultdict(dict)
        all_queries = []
        specific_queries = {}
        broad_queries = {}
        for i, entry in enumerate(ds['full']):
            qid = str(i)
            if entry['specificity'] == 0:
                broad_queries[qid] = entry['query']
                q_type = 'broad'
            else:
                specific_queries[qid] = entry['query']
                q_type = 'specific'
            for doc_id in entry['corpusids']:
                if q_type == 'specific':
                    specific_qrels_dict[qid][str(doc_id)] = 1
                else:
                    broad_qrels_dict[qid][str(doc_id)] = 1
        int2docid = {}
        for i, entry in enumerate(datasets.load_dataset(dataset_name, 'corpus_clean')['full']):
            int2docid[i] = str(entry['corpusid'])

        subset = kwargs.get("subset", "broad")
        if subset == "specific":
            return int2docid, specific_queries, specific_qrels_dict
        elif subset == "broad":
            return int2docid, broad_queries, broad_qrels_dict
        else:
            raise ValueError(f"Subset {subset} not found")
    else:
        raise ValueError(f"Dataset {dataset} not found")


def run(args, **kwargs):
    print("Loading queries...")
    int2docid, queries, qrels_dict = load_data(args.data_dir, args.dataset, **kwargs)

    qrels = Qrels(qrels_dict)
    config = ColBERTConfig(
        nbits=2,
        nranks=1,
        root=".experiment/",
        avoid_fork_if_possible=True,
        ncells=8,
        ndocs=4096,
        centroid_score_threshold=0.3,
    )
    searcher_path = f"{args.dataset}_{args.experiment}"
    searcher = Searcher(index=searcher_path, config=config)

    run_dict = defaultdict(dict)
    for qid, query in tqdm(queries.items()):
        result = searcher.search(query, k=20 if args.dataset == "litsearch" else 10)
        for i, r in enumerate(result[0]):
            run_dict[qid][int2docid[r]] = result[2][i]

    print(len(run_dict.keys()))
    run = ranx_run(run_dict)

    if dataset == "litsearch":
        metrics = {
            'recall@5': evaluate(qrels, run, "recall@5"),
            'recall@10': evaluate(qrels, run, "recall@10"),
            'recall@20': evaluate(qrels, run, "recall@20"),
        }
    else:
        mrr10 = evaluate(qrels, run, "mrr@10")
        map10 = evaluate(qrels, run, "map@10")
        ndcg10 = evaluate(qrels, run, "ndcg@10")
        recall50 = evaluate(qrels, run, "recall@50")
        recall100 = evaluate(qrels, run, "recall@100")
        metrics = {
            'mrr@10': mrr10,
            'map@10': map10,
            'ndcg@10': ndcg10,
            'recall@50': recall50,
            'recall@100': recall100
        }

    print(args.dataset, kwargs)
    print(metrics)

    srsly.write_json(f"./results_{args.dataset}_{args.experiment}.json", metrics)

    obj = {"dataset": args.dataset, "experiment": args.experiment}
    obj.update(kwargs)
    obj.update(metrics)
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline", help="Path to the experiment file")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--datasets", nargs='+', default=["litsearch", "scifact"], help="Name of datasets to test")

    args = parser.parse_args()

    stats = []
    for dataset in args.datasets:
        args.dataset = dataset
        if dataset == "litsearch":
            stats.append(run(args, **{"subset": "broad"}))
            stats.append(run(args, **{"subset": "specific"}))
        else:
            stats.append(run(args))

    stats = pd.DataFrame(stats)

    print(stats.head(len(stats)))

    out_fn = f"./results_{args.experiment}.csv"
    print(f"Saving to {out_fn}..")
    stats.to_csv(out_fn, index=False)
