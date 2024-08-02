import ir_datasets
from ranx import Qrels, evaluate
import srsly
import ir_datasets
from collections import defaultdict
from colbert.infra import ColBERTConfig
from colbert import Searcher
from tqdm import tqdm
import os
from ranx import Run as ranx_run
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline", help="Path to the experiment file")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    args = parser.parse_args()

    print("Loading documents...")
    int2docid = {int(k): v for k, v in srsly.read_json(os.path.join(args.data_dir, 'scifact_int2docid.json')).items()}

    print("Loading queries...")

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

    assert len(qrels_dict) == len(queries)

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
    searcher_path = f"SCIFACT_500k_{args.experiment}"
    searcher = Searcher(index=searcher_path, config=config)

    run_dict = defaultdict(dict)
    for qid, query in tqdm(queries.items()):
        result = searcher.search(query, k=10)
        for i, r in enumerate(result[0]):
            run_dict[qid][int2docid[r]] = result[2][i]

    print(len(run_dict.keys()))
    run = ranx_run(run_dict)
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

    all_results= metrics
    print(all_results)

    srsly.write_json(f"./results_scifact_{args.experiment}.json", all_results)
