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
DOMAIN=${7:-1}

DIR=outputs/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/${DATASET}/seed${SEED}
LOG_DONE=${DIR}/log.txt
MODEL_DIR=outputs/domain_generalization/patternnetv2/${TRAINER}/${CFG}_shots${SHOTS}/seed${SEED}

is_done_log() {
    local f="$1"
    [[ -f "$f" ]] || return 1
    tail -n 200 "$f" | grep -q '=> result' || return 1
    tail -n 200 "$f" | grep -qE 'accuracy:[[:space:]]*[0-9]+(\.[0-9]+)?%' || return 1
    tail -n 200 "$f" | grep -qE 'kappa:[[:space:]]*[0-9]+(\.[0-9]+)?%?' || return 1
    tail -n 200 "$f" | grep -qE 'macro_f1:[[:space:]]*[0-9]+(\.[0-9]+)?%' || return 1
}

if is_done_log "$LOG_DONE"; then
    echo "[SKIP] completed test already exists in ${DIR}; replay metrics"
    tail -n 200 "$LOG_DONE"
    exit 0
fi

if [[ -d "$DIR" ]]; then
    BACKUP="${DIR}.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "[WARN] incomplete test directory exists; move to ${BACKUP}"
    mv "$DIR" "$BACKUP"
fi

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file yaml/datasets/${DATASET}.yaml \
    --config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    --domain ${DOMAIN} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
