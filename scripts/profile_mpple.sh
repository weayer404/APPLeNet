#!/bin/bash

source config.sh
cd ..

DATASET=${1:-$DATASET}
SEED=${2:-$SEED}
SHOTS=${3:-$SHOTS}
TRAINER=${4:-$TRAINER}
CFG=${5:-$CFG}
SUB=${6:-$SUB}
DOMAIN=${7:-0}

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=outputs/base2new/train_base/${COMMON_DIR}

python profile_mpple.py \
  --root ${DATA} \
  --seed ${SEED} \
  --trainer ${TRAINER} \
  --dataset-config-file yaml/datasets/${DATASET}.yaml \
  --config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
  --domain ${DOMAIN} \
  --model-dir ${MODEL_DIR} \
  --batch-size 1 \
  --warmup 50 \
  --eval-only \
  --repeat 200 \
  DATASET.NUM_SHOTS ${SHOTS} \
  DATASET.SUBSAMPLE_CLASSES ${SUB}