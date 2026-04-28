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

python visual/vis_patternnet_heatmaps.py \
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
--only-cm \         # 只想生成这种分类热力图，不想生成 Grad-CAM
--cm-all-classes \  # 想生成全部 38 类
--cm-class-names "river,bridge,runway,parking_lot,storage_tank,harbor" \   # 指定分类热力图类别
# --class-names "river,bridge,runway,parking_lot,storage_tank" \           # 指定注意力热力图类别
DATASET.NUM_SHOTS ${SHOTS} \ 
DATASET.SUBSAMPLE_CLASSES ${SUB}