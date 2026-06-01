#!/bin/bash

# config.sh
# 公共默认配置：可以提交到 Git。
# 本机差异配置请写到同目录的 .env/local_config.sh。
# 若不存在，则读取 Git 同步的 .env/local_config_default.sh。

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${CONFIG_DIR}/.env/local_config.sh" ]]; then
  LOCAL_CFG="${CONFIG_DIR}/.env/local_config.sh"
else
  LOCAL_CFG="${CONFIG_DIR}/.env/local_config_default.sh"
fi

# ====== 公共默认值 ======
DATA=${DATA:-~/code/Data}
TRAINER=${TRAINER:-MPPLe}
CFG=${CFG:-vit_b16_c8}
SUB=${SUB:-all}
NCTX=${NCTX:-8}
SHOTS=${SHOTS:-16}
SHOT_TRIES=${SHOT_TRIES:-3}

if [[ -n "${ALL_SHOTS_STR:-}" ]]; then
  read -r -a ALL_SHOTS <<< "$ALL_SHOTS_STR"
elif ! declare -p ALL_SHOTS >/dev/null 2>&1; then
  ALL_SHOTS=(1 4 16 32 64)
fi

START_RUN=${START_RUN:-1}
END_RUN=${END_RUN:-3}
OUTPUT_BASE=${OUTPUT_BASE:-../outputs}

SEP=${SEP:-$'\t'}
ENABLE_VIS=${ENABLE_VIS:-0}
LOG_INTERVAL=${LOG_INTERVAL:-5}

CPU_THREADS=$( (nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2) | tr -dc '0-9' )
DEFAULT_MAX_TASK_NUM=$(( CPU_THREADS > 1 ? CPU_THREADS - 1 : 1 ))
MAX_TASK_NUM=${MAX_TASK_NUM:-$DEFAULT_MAX_TASK_NUM}

SPLIT_CSV_BY_SHOT=${SPLIT_CSV_BY_SHOT:-0}
GPU_USER_LIMIT_MB=${GPU_USER_LIMIT_MB:-24000}
GPU_KEEP_FREE_MB=${GPU_KEEP_FREE_MB:-1000}
INIT_RESERVE_MB=${INIT_RESERVE_MB:-12000}
PEAK_MARGIN_MB=${PEAK_MARGIN_MB:-500}
MEM_PROFILE_DIM=${MEM_PROFILE_DIM:-dataset_shot}
ALLOW_PARALLEL_UNPROFILED_SAME_KEY=${ALLOW_PARALLEL_UNPROFILED_SAME_KEY:-0}
RECOVER_EXISTING_TRAIN_CSV=${RECOVER_EXISTING_TRAIN_CSV:-1}
RECOVER_REQUIRE_FULL_EPOCH=${RECOVER_REQUIRE_FULL_EPOCH:-1}
MEM_SAMPLE_INTERVAL=${MEM_SAMPLE_INTERVAL:-3}
MEM_SAMPLE_BY_SESSION=${MEM_SAMPLE_BY_SESSION:-1}
MEM_DEBUG=${MEM_DEBUG:-0}
GPU_MEM_QUERY_INTERVAL=${GPU_MEM_QUERY_INTERVAL:-8}
MIN_PROFILE_RESERVE_MB=${MIN_PROFILE_RESERVE_MB:-500}
RESERVE_ROUND_MB=${RESERVE_ROUND_MB:-200}
POLL_INTERVAL=${POLL_INTERVAL:-2}
GPU_ID=${GPU_ID:-0}

# ====== 本地覆盖层：放在公共默认值之后，更稳 ======
if [[ -f "$LOCAL_CFG" ]]; then
  echo "[CONFIG] load local config: $LOCAL_CFG"
  # shellcheck source=/dev/null
  source "$LOCAL_CFG"
else
  echo "[CONFIG][WARN] local config not found: $LOCAL_CFG"
fi

# 如果 local_config.sh 用 ALL_SHOTS_STR 覆盖，需要在 source 后再解析一次。
if [[ -n "${ALL_SHOTS_STR:-}" ]]; then
  read -r -a ALL_SHOTS <<< "$ALL_SHOTS_STR"
elif ! declare -p ALL_SHOTS >/dev/null 2>&1; then
  ALL_SHOTS=(1 4 16 32 64)
fi

# ====== 派生路径：必须放在本地覆盖之后 ======
RUNLOG_DIR=${RUNLOG_DIR:-"${OUTPUT_BASE}/runner_logs"}
TASKLOG_DIR=${TASKLOG_DIR:-"${RUNLOG_DIR}/task_logs"}

refresh_output_paths() {
  OUTPUT_LOG_FILE="${RUNLOG_DIR}/log_stat_${TRAINER}_${CFG}_shot${SHOTS}_seedstart${START_RUN}_end${END_RUN}.txt"
  VIS_LOG_FILE="${RUNLOG_DIR}/log_vis_${TRAINER}_${CFG}_shot${SHOTS}_seedstart${START_RUN}_end${END_RUN}.txt"
  RESULT_CSV_FILE="${OUTPUT_BASE}/results_${TRAINER}_${CFG}_shot${SHOTS}_seedstart${START_RUN}_end${END_RUN}.csv"
  AVERAGE_CSV_FILE="${OUTPUT_BASE}/averages_${TRAINER}_${CFG}_shot${SHOTS}_seedstart${START_RUN}_end${END_RUN}.csv"
  LOCK_FILE="${RUNLOG_DIR}/.lock_${TRAINER}_${CFG}_shots${SHOTS}_seedstart${START_RUN}_end${END_RUN}.lock"
  SCHED_LOG="${RUNLOG_DIR}/scheduler.log"
  MEM_PROFILE_FILE="${RUNLOG_DIR}/mem_profile_${TRAINER}_${CFG}.tsv"
}

refresh_output_paths