#!/bin/bash

source config.sh 
cd ..

# custom config

DATASET=$1
SEED=$2

SUB=base
NCTX=4  # number of context tokens


DIR=outputs/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "The results already exist in ${DIR}"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file yaml/datasets/${DATASET}.yaml \
    --config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} 
fi  