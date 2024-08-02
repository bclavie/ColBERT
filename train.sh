#!/bin/bash

set -e

TRIPLETS_PATH=$HOME/320k_triplets_normalized.jsonl

python train.py --data="./data" \
--triplets=$TRIPLETS_PATH \
--base_model="bert-base-uncased" \
--bsize=32 \
--lr=1e-05 \
--warmup=2500 \
--doc_maxlen=300 \
--use_ib_negatives=false \
--nway=16 \
--accumsteps=2 \
--schedule_free=false \
--kldiv_loss=true \
--marginmse_loss=false \
--kldiv_weight=1.0 \
--marginmse_weight=0.05 \
--normalise_training_scores=true \
--normalization_method='minmax' \
--experiment="baseline"