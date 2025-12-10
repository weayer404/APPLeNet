#!/bin/bash

source config.sh 
cd ..

# custom config

DATASET=$1
SEED=$2

# LOADEP=30
SUB=all 

# --load-epoch ${LOADEP} \


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

MODEL_DIR=outputs/base2new/train_base/${COMMON_DIR}
DIR=outputs/base2new/test_${SUB}/${COMMON_DIR}
# if [ -d "$DIR" ]; then
#     echo "The results already exist in ${DIR}"
# else
    python visual/vis_gradcam.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file yaml/datasets/${DATASET}.yaml \
    --config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --num-images 38 \
    --target-layer "conv1" \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} 
# fi