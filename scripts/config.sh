#!/bin/bash

# config.sh
# ====== 全局信息 ======
DATA=${DATA:-~/code/Data}
TRAINER=${TRAINER:-AppleNet}
CFG=${CFG:-vit_b16_c8}         # vit_b16_c4 vit_b32_c4 vit_b16_c16 vit_b32_c16
SUB=${SUB:-all}  
SHOTS=${SHOTS:-1}               # 1 2 4 8 16
SHOT_TRIES=${SHOT_TRIES:-5}
ALL_SHOTS=(1 16 32 64) # 可选的 shots 列表, 请和 train.py 中保持一致
START_RUN=${START_RUN:-1}
END_RUN=${END_RUN:-3}
OUTPUT_BASE=${OUTPUT_BASE:-../outputs}
RUNLOG_DIR="${OUTPUT_BASE}/runner_logs"
TASKLOG_DIR="${RUNLOG_DIR}/task_logs"
SEP=$'\t'                       # Tab 分隔符
ENABLE_VIS=${ENABLE_VIS:-0}   # 1=跑可视化，0=不跑
LOG_INTERVAL=${LOG_INTERVAL:-5} # 每隔几秒刷新一次日志
# 最多同时跑多少任务：默认=CPU线程数-1（至少为1）
CPU_THREADS=$( (nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2) | tr -dc '0-9' )
MAX_TASK_NUM=$(( CPU_THREADS > 1 ? CPU_THREADS - 1 : 1 ))
SPLIT_CSV_BY_SHOT=${SPLIT_CSV_BY_SHOT:-0} # 0/1/2 所有 shots 写到同一对/每个 shot 单独一对/两种都写
GPU_USER_LIMIT_MB=${GPU_USER_LIMIT_MB:-23500} # 脚本最多占用多少显存
GPU_KEEP_FREE_MB=${GPU_KEEP_FREE_MB:-0} # 额外余量，再有其他人使用显卡的时候设置
INIT_RESERVE_MB=${INIT_RESERVE_MB:-4500} # 首次运行的保守显存预算
PEAK_MARGIN_MB=${PEAK_MARGIN_MB:-1000} # 画像峰值预测的显存余量
MEM_PROFILE_DIM=${MEM_PROFILE_DIM:-none} # 画像显存相关系的粒度可选：none | seed | shot | seed_shot  
MEM_SAMPLE_INTERVAL=${MEM_SAMPLE_INTERVAL:-3} # 峰值采样间隔
GPU_MEM_QUERY_INTERVAL=${GPU_MEM_QUERY_INTERVAL:-8} # 显存查询缓存间隔
MIN_PROFILE_RESERVE_MB=${MIN_PROFILE_RESERVE_MB:-500} # 画像出来后允许的最小预算
RESERVE_ROUND_MB=${RESERVE_ROUND_MB:-200} # 预算向上取整粒度：2738 -> 2800
POLL_INTERVAL=${POLL_INTERVAL:-2} # 轮询周期（秒）
GPU_ID=${GPU_ID:-0} # 默认使用的 GPU ID
# ====== 全局信息 ======

# ====== 输出文件 ======
refresh_output_paths() {
  OUTPUT_LOG_FILE="${RUNLOG_DIR}/log_stat_${TRAINER}_${CFG}_shot${SHOTS}_seedstart${START_RUN}_end${END_RUN}.txt"
  VIS_LOG_FILE="${RUNLOG_DIR}/log_vis_${TRAINER}_${CFG}_shot${SHOTS}_seedstart${START_RUN}_end${END_RUN}.txt"
  RESULT_CSV_FILE="${OUTPUT_BASE}/results_${TRAINER}_${CFG}_shot${SHOTS}_seedstart${START_RUN}_end${END_RUN}.csv"
  AVERAGE_CSV_FILE="${OUTPUT_BASE}/averages_${TRAINER}_${CFG}_shot${SHOTS}_seedstart${START_RUN}_end${END_RUN}.csv"
  LOCK_FILE="${RUNLOG_DIR}/.lock_${TRAINER}_${CFG}_shots${SHOTS}_seedstart${START_RUN}_end${END_RUN}.lock"
  SCHED_LOG="${RUNLOG_DIR}/scheduler.log"
  MEM_PROFILE_FILE="${RUNLOG_DIR}/mem_profile_${TRAINER}_${CFG}.tsv"
}

# 初次加载时先生成一次
refresh_output_paths
# ====== 输出文件 ======