#!/bin/bash
# local_config.sh 示例文件
# 复制为 local_config.sh 后按本机情况修改，并把 local_config.sh 加入 .gitignore。
# 这个文件会在 config.sh 中被自动 source，因此可以使用 Bash 数组。

# ===== 本机路径 =====
DATA=~/code/Data
OUTPUT_BASE=../outputs
GPU_ID=0

# ===== 实验方案 =====
TRAINER=MPPLe
CFG=vit_b16_c8
SUB=all

# 跑哪些 shot；数组写法适合 local_config.sh。
ALL_SHOTS=(1 4 16)
SHOT_TRIES=3
START_RUN=1
END_RUN=3

# 跑哪些任务；不写则使用 run_single_average_test.sh 里的默认清单。
# 可选 token 示例：
# "base2new_patternnet" "base2new_mlrsnet" "base2new_resisc45" "base2new_rsicd"
# "crossdata_patternnet" "crossdata_mlrsnet" "crossdata_resisc45" "crossdata_rsicd"
# "domaingen_patternnetv2" "domaingen_mlrsnetv2" "domaingen_resisc45v2" "domaingen_rsicdv2"
RUN_ORDER=(
  "base2new_patternnet" "base2new_mlrsnet" "base2new_resisc45" "base2new_rsicd"
  "domaingen_patternnetv2" "domaingen_mlrsnetv2" "domaingen_resisc45v2" "domaingen_rsicdv2"
)

# ===== 调度与显存 =====
# MAX_TASK_NUM=2
GPU_USER_LIMIT_MB=16000
GPU_KEEP_FREE_MB=1000
INIT_RESERVE_MB=12000
PEAK_MARGIN_MB=500
MEM_PROFILE_DIM=dataset_shot
ALLOW_PARALLEL_UNPROFILED_SAME_KEY=0

# ===== 日志与恢复 =====
SPLIT_CSV_BY_SHOT=0
RECOVER_EXISTING_TRAIN_CSV=1
RECOVER_REQUIRE_FULL_EPOCH=1
MEM_SAMPLE_BY_SESSION=1
MEM_DEBUG=0
