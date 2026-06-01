#!/bin/bash

# custom config
source config.sh 
cd ..

DATASET=${1:-$DATASET}
SEED=${2:-$SEED}
SHOTS=${3:-$SHOTS}
TRAINER=${4:-$TRAINER}
CFG=${5:-$CFG}
SUB=${6:-$SUB}

DIR=outputs/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/${DATASET}/seed${SEED}

python visual/vis_gradcam.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file yaml/datasets/${DATASET}.yaml \
--config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir outputs/domain_generalization/patternnetv2/${TRAINER}/${CFG}_shots${SHOTS}/seed${SEED} \
--num-images 3 \
--target-layer "conv1" \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
