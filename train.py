from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer
import srsly

import os
import argparse
from pathlib import Path


def train(args):
    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=4, experiment=args.experiment)):
        queries = args.data + "/queries.tsv"
        collection = args.data + "/collection.tsv"

        total_triplets = 0
        for x in srsly.read_jsonl(args.triplets):
            total_triplets += 1

        max_num_steps = total_triplets // args.bsize

        config = ColBERTConfig(
            bsize=args.bsize, lr=args.lr, warmup=args.warmup, doc_maxlen=args.doc_maxlen,
            dim=128, nway=args.nway, accumsteps=args.accumsteps,  use_ib_negatives=args.use_ib_negatives,
            schedule_free=args.schedule_free, kldiv_loss=args.kldiv_loss, marginmse_loss=args.marginmse_loss,
            kldiv_weight=args.kldiv_weight, marginmse_weight=args.marginmse_weight,
            normalise_training_scores=args.normalise_training_scores, normalization_method=args.normalization_method,
            maxsteps=max_num_steps, schedule_free_wd=args.schedule_free_wd, cap_padding=args.cap_padding,
            gist_freq=args.gist_freq
        )

        print(config)

        trainer = Trainer(triples=args.triplets, queries=queries, collection=collection, config=config)

        trainer.train(checkpoint=args.base_model)  # or start from scratch, like `bert-base-uncased`


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ColBERT model')
    parser.add_argument('--triplets', type=str, default=os.path.expanduser("~/320k_triplets_normalized.jsonl"), help='Path to triplets file')
    parser.add_argument('--data', type=str, default=str(Path(__file__).parent / 'data'), help='Path to data')
    parser.add_argument('--experiment', type=str, default="gist_4", help='Experiment name')
    parser.add_argument('--base_model', type=str, default='bert-base-uncased', help='Base model')  # bclavie/JaColBERT
    parser.add_argument('--bsize', type=int, default=16, help='Batch size')  # 64
    parser.add_argument('--lr', type=float, default=1e-05, help='Learning rate')
    parser.add_argument('--warmup', type=int, default=2500, help='Warmup steps')  # 500
    parser.add_argument('--doc_maxlen', type=int, default=400, help='Maximum document length')  # 300
    parser.add_argument('--use_ib_negatives', type=lambda x: x.lower() == 'true', default=False, help='Use in-batch negatives')
    parser.add_argument('--nway', type=int, default=16, help='Number of ways for training')  # 32
    parser.add_argument('--accumsteps', type=int, default=4, help='Gradient accumulation steps')  # 1
    parser.add_argument('--schedule_free', type=lambda x: x.lower() == 'true', default=False, help='Use schedule free training')
    parser.add_argument('--schedule_free_wd', type=float, default=0.0, help='Weight decay for schedule free training')
    parser.add_argument('--kldiv_loss', type=lambda x: x.lower() == 'true', default=True, help='Use KL divergence loss')
    parser.add_argument('--marginmse_loss', type=lambda x: x.lower() == 'true', default=False, help='Use margin MSE loss')
    parser.add_argument('--kldiv_weight', type=float, default=1.0, help='Weight for KL divergence loss')
    parser.add_argument('--marginmse_weight', type=float, default=0.05, help='Weight for margin MSE loss')
    parser.add_argument('--normalise_training_scores', type=lambda x: x.lower() == 'true', default=True, help='Normalize training scores')  # False
    parser.add_argument('--normalization_method', type=str, default='minmax', choices=['minmax', 'querylen'], help='Normalization method')
    parser.add_argument('--cap_padding', type=int, default=0, help='Cap padding')
    parser.add_argument('--gist_freq', type=int, default=4, help='Number of tokens in between consecutive GIST tokens. 0 for no GIST.')
    args = parser.parse_args()
    
    train(args)
