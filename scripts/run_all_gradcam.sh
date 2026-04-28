#!/bin/bash

# 全局配置
source config.sh

run_order=( "base2new_patternnet" ) # "domaingen"  "crossdata"
# run_order=("domaingen" "base2new_patternnet" "base2new_mlrsnet" "base2new_resisc45" "base2new_rsicd") # "domaingen"  "crossdata"
run_photo_v1=("base2new_patternnet")
# run_photo_v1=("base2new_patternnet" "base2new_mlrsnet" "base2new_resisc45" "base2new_rsicd")
run_photo_v2=( )
# run_photo_v2=("domaingen" "domaingen" "domaingen" )
vis_models_v1=("patternnet")
# vis_models_v1=("patternnet" "mlrsnet" "resisc45" "rsicd")
vis_models_v2=()
# vis_models_v2=("mlrsnetv2" "resisc45v2" "rsicdv2")

echo "平均值计算完毕，开始生成可视化（t-SNE & Grad-CAM）..."
mkdir -p "$(dirname "$VIS_LOG_FILE")"

# === 新增：对本次所有 run_order 进行可视化 ===
for seed in $(seq $START_RUN $END_RUN); do
    for i in "${!run_photo_v1[@]}"; do
        task_name="${run_photo_v1[$i]}"
        model_name="${vis_models_v1[$i]}"

        echo "为任务 ${task_name} (模型: ${model_name}, seed: ${seed}) 生成可视化..."

        # Grad-CAM 可视化
        bash vis_gradcam.sh "$model_name" "$seed" >>"$VIS_LOG_FILE" 2>&1
    done
done
for seed in $(seq $START_RUN $END_RUN); do
    for i in "${!run_photo_v2[@]}"; do
        task_name="${run_photo_v2[$i]}"
        model_name="${vis_models_v2[$i]}"

        echo "为任务 ${task_name} (模型: ${model_name}, seed: ${seed}) 生成可视化..."

        # Grad-CAM 可视化
        bash vis_gradcam_v2.sh "$model_name" "$seed" >>"$VIS_LOG_FILE" 2>&1
    done
done

echo "所有可视化任务完成，脚本结束。"


