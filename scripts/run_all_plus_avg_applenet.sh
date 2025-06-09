#!/bin/bash

# 全局配置
source config.sh

# 输出顺序保持不变
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

plus_avg_value() {
    declare -A entries  # 使用关联数组存储每个组合的所有数据行

    # 读取并收集数据
    while IFS=',' read -r total_type type model run accuracy error macro_f1; do
        if [[ $run -ge $START_RUN && $run -le $END_RUN ]]; then
            # 验证数据有效性
            if [[ ! "$accuracy" =~ ^[0-9.]+$ ]] || [[ ! "$error" =~ ^[0-9.]+$ ]] || [[ ! "$macro_f1" =~ ^[0-9.]+$ ]]; then
                echo "Skipping invalid line: $total_type,$type,$model,$run,$accuracy,$error,$macro_f1"
                continue
            fi
            key="${total_type},${type},${model}"
            entries["$key"]+=$'\n'"$accuracy,$error,$macro_f1,$run"  # 以换行符分隔数据行
        fi
    done < "$RESULT_CSV_FILE"

    # 处理每个组合的数据
    for key in "${output_order[@]}"; do
        IFS=$'\n' read -d '' -ra lines <<< "${entries[$key]}"
        # 过滤空行
        non_empty_lines=()
        for line in "${lines[@]}"; do
            [[ -n "$line" ]] && non_empty_lines+=("$line")
        done
        lines=("${non_empty_lines[@]}")
        
        if [[ ${#lines[@]} -lt 3 ]]; then
            # 数据不足，无法排除两个极值
            avg_acc=0.0000
            avg_err=0.0000
            avg_f1=0.0000
        else
            # 按accuracy排序
            sorted_lines=($(printf "%s\n" "${lines[@]}" | sort -t',' -k1,1n))
            # 排除首尾（最小和最大值）
            remaining_lines=("${sorted_lines[@]:1:$((${#sorted_lines[@]} - 2))}")

            sum_acc=0
            sum_err=0
            sum_f1=0
            count=0
            for line in "${remaining_lines[@]}"; do
                IFS=',' read -r acc err f1 run <<< "$line"
                sum_acc=$(bc <<< "$sum_acc + $acc")
                sum_err=$(bc <<< "$sum_err + $err")
                sum_f1=$(bc <<< "$sum_f1 + $f1")
                ((count++))
            done
            # 计算平均值并格式化为四位小数
            avg_acc=$(printf "%.2f" $(bc -l <<< "scale=6; $sum_acc / $count"))
            avg_err=$(printf "%.2f" $(bc -l <<< "scale=6; $sum_err / $count"))
            avg_f1=$(printf "%.2f" $(bc -l <<< "scale=6; $sum_f1 / $count"))
        fi

        # 输出结果到文件（保持原字段顺序：accuracy,error,macro_f1,run,...）
        IFS=',' read -r t_type ty mo <<< "$key"
        printf "%.2f,%.2f,%.2f,%s,%s,%s\n" "$avg_acc" "$avg_err" "$avg_f1" "$t_type" "$ty" "$mo" >> "$PLUS_AVG_CSV_FILE"
    done
}

plus_avg_value  # 执行修改后的函数