#!/bin/bash

set -e

TRIPLETS_PATH=/home/azureuser/GistColBERT/gist_data/100k_triplets_normalized.jsonl

python train.py --data="/home/azureuser/GistColBERT/gist_data/" \
--triplets=$TRIPLETS_PATH \
--base_model="answerdotai/answerai-colbert-small-v1" \
--bsize=64 \
--lr=1e-05 \
--warmup=100 \
--doc_maxlen=280 \
--use_ib_negatives=false \
--nway=32 \
--accumsteps=1 \
--schedule_free=false \
--kldiv_loss=true \
--marginmse_loss=false \
--kldiv_weight=1.0 \
--marginmse_weight=0.05 \
--normalise_training_scores=true \
--normalization_method='minmax' \
--gist_freq=0 \
--experiment="baseline"