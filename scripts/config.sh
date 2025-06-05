#!/bin/bash

# config.sh
DATA=/home/koko/code/APPLeNet/Data
TRAINER=AppleNet
CFG=vit_b16_c4 # vit_b16_c4 vit_b16_c8 vit_b16_c12 vit_b16_c16 vit_b32_c4
SHOTS=16
EPOCH=100
START_RUN=0
END_RUN=4
MAX_TASK_NUM=6
CUDA_MEMORY=2500
OUTPUT_BASE="../outputs"
OUTPUT_LOG_FILE="${OUTPUT_BASE}/log_${CFG}_${START_RUN}_${END_RUN}.txt"
RESULT_CSV_FILE="${OUTPUT_BASE}/results_${CFG}_${START_RUN}_${END_RUN}.csv"
AVERAGE_CSV_FILE="${OUTPUT_BASE}/averages_${CFG}_${START_RUN}_${END_RUN}.csv"

PLUS_AVG_CSV_FILE="${OUTPUT_BASE}/plus_avg_${CFG}_${START_RUN}_${END_RUN}.csv" # 加权平均值

MAX_CSV_FILE="${OUTPUT_BASE}/max_${CFG}_${START_RUN}_${END_RUN}.csv" # 最大值

################################ 运行脚本 ###############################
# nohup bash run_all_average_applenet.sh > ~/files/log_all_applenet.txt 2>&1 &
############################## 查看脚本输出 #############################
# tail -f ~/files/log_all_applenet.txt