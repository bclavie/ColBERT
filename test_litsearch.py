import datasets
from ranx import Qrels, evaluate
import srsly
from collections import defaultdict
from colbert.infra import ColBERTConfig
from colbert import Searcher
from tqdm import tqdm
from ranx import Run as ranx_run

if __name__ == "__main__":
    dataset_name = 'princeton-nlp/LitSearch'

    models = [
        # 'M3_DIM96_Schedulefree'
        # 'M3_GTE_SMALL.out',
        # 'M3_GTE_SMALL_STEP1_3M200K_MMARCO_LR2e-5.out',
        'ABLATION_M3_GTE_SMALL_STEP1_3M200K_MMARCO_LR5e-5'
    ]

    config = ColBERTConfig(
        nbits=2,
        nranks=1,
        root=".ablation_v1_en/",
        avoid_fork_if_possible=True,
        ncells=8,
        ndocs=4096,
        centroid_score_threshold=0.3,
        query_maxlen=32,
    )

    all_results = {model: {} for model in models}

    print(f"Processing dataset: {dataset_name}")
    
    print("Loading dataset...")
    ds = datasets.load_dataset(dataset_name, 'query')
    
    print("Loading queries...")

    specific_qrels_dict = defaultdict(dict)
    broad_qrels_dict = defaultdict(dict)
    all_queries = []
    specific_queries = []
    broad_queries = []
    for i, entry in enumerate(ds['full']):
        if entry['specificity'] == 0:
            broad_queries.append(entry['query'])
            q_type = 'broad'
        else:
            specific_queries.append(entry['query'])
            q_type = 'specific'
        for doc_id in entry['corpusids']:
            if q_type == 'specific':
                specific_qrels_dict[str(i)][str(doc_id)] = 1
            else:
                broad_qrels_dict[str(i)][str(doc_id)] = 1
    specific_qrels = Qrels(specific_qrels_dict)
    broad_qrels = Qrels(broad_qrels_dict)

    int2docid = {}
    for i, entry in enumerate(datasets.load_dataset(dataset_name, 'corpus_clean')['full']):
        int2docid[i] = str(entry['corpusid'])

    for model in models:
        print(f"Evaluating model: {model}")
        searcher_path = f"{dataset_name}_V1_{model}"
        searcher = Searcher(index=searcher_path, config=config)

        run_dict = defaultdict(dict)
        for queries, qrels in [(specific_queries, specific_qrels), (broad_queries, broad_qrels)]:
            for qid, query in tqdm(queries.items()):
                result = searcher.search(query, k=100)
            for i, r in enumerate(result[0]):
                run_dict[str(qid)][str(int2docid[r])] = result[2][i]

            run = ranx_run(run_dict)
            metrics = {
                'recall@5': evaluate(qrels, run, "recall@5"),
                'recall@10': evaluate(qrels, run, "recall@10"),
                'recall@20': evaluate(qrels, run, "recall@20"),
            }

            all_results[model][dataset_name][q_type] = metrics
            print(f"===== {searcher_path} =====")
            print(metrics)

    srsly.write_json("./results_v1_litsearch_specific_broad.json", all_results)