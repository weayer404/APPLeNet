#!/bin/bash

# 全局配置
source config.sh

# 定义输出顺序
output_order=(
    "base2new,base,patternnet"
    "base2new,base,mlrsnet"
    "base2new,base,resisc45"
    "base2new,base,rsicd"
    "base2new,new,patternnet"
    "base2new,new,mlrsnet"
    "base2new,new,resisc45"
    "base2new,new,rsicd"
    "crossdata,source,patternnet"
    "crossdata,target,rsicd"
    "crossdata,target,resisc45"
    "crossdata,target,mlrsnet"
    "domaingen,source,patternnetv2"
    "domaingen,target,rsicdv2"
    "domaingen,target,resisc45v2"
    "domaingen,target,mlrsnetv2"
)

max() {
    # 比较两个数字，返回较大的一个
    if (( $(echo "$1 > $2" | bc -l) )); then
        echo "$1"
    else
        echo "$2"
    fi
}

max_value() {
    # 初始化一个关联数组来保存每个组合的最大准确率对应的行数据
    declare -A max_accuracy_line

    while IFS=',' read -r total_type type model run accuracy error macro_f1; do
        # 确保 run 在指定的区间内
        if [[ $run -ge $START_RUN && $run -le $END_RUN ]]; then
            # 用字符串键（total_type, type, model）进行分组
            key="${total_type},${type},${model}"

            # 检查数据是否为有效数字
            if [[ ! "$accuracy" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ ! "$error" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ ! "$macro_f1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                echo "Skipping invalid line: $total_type,$type,$model,$run,$accuracy,$error,$macro_f1"
                continue
            fi

            # 获取当前组合的最大准确率
            current_max_accuracy="${max_accuracy_line[$key]%%,*}"  # 提取现有最大准确率
            current_max_accuracy=${current_max_accuracy:-0}  # 如果没有值则为0
            max_accuracy=$(max "$accuracy" "$current_max_accuracy")

            # 调试输出，查看每次比较的准确率
            echo "Checking $key: run=$run, accuracy=$accuracy, current_max_accuracy=$current_max_accuracy, max_accuracy=$max_accuracy"

            # 如果当前准确率是最大值，则更新最大准确率对应的行
            if [[ "$max_accuracy" == "$accuracy" ]]; then
                # 更新对应组合的最大准确率的行（包括run）
                max_accuracy_line["$key"]="$accuracy,$error,$macro_f1,$run,$total_type,$type,$model"
            fi
        fi
    done < $RESULT_CSV_FILE
    
    # 将每个组合的最大值行按顺序写入到最大值文件
    for key in "${output_order[@]}"; do
        # 输出对应的最大值行
        echo "${max_accuracy_line[$key]}" >> $MAX_CSV_FILE
    done
}


max_value