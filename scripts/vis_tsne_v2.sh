#!/bin/bash

source config.sh 
cd ..

# custom config

DATASET=$1
SEED=$2

# LOADEP=30
SUB=all 

DIR=outputs/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/${DATASET}/seed${SEED}

python visual/vis_tsne.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file yaml/datasets/${DATASET}.yaml \
--config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir outputs/domain_generalization/patternnetv2/${TRAINER}/${CFG}_shots${SHOTS}/seed${SEED} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
