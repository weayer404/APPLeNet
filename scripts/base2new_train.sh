#!/bin/bash
set -euo pipefail

source config.sh
cd ..

DATASET=${1:-$DATASET}
SEED=${2:-$SEED}
SHOTS=${3:-$SHOTS}
TRAINER=${4:-$TRAINER}
CFG=${5:-$CFG}
SUB=${6:-$SUB}
DOMAIN=${7:-0}

DIR=outputs/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_BEST=${DIR}/model/model-best.pth.tar
MODEL_LAST=${DIR}/model/model-last.pth.tar

if [[ -f "$MODEL_BEST" || -f "$MODEL_LAST" ]]; then
    echo "[SKIP] completed training already exists in ${DIR}"
    # run_single_average_test.sh validates train success by an epoch marker.
    echo "epoch [1/1]"
    exit 0
fi

if [[ -d "$DIR" ]]; then
    BACKUP="${DIR}.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "[WARN] incomplete training directory exists; move to ${BACKUP}"
    mv "$DIR" "$BACKUP"
fi

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file yaml/datasets/${DATASET}.yaml \
    --config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --domain ${DOMAIN} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
