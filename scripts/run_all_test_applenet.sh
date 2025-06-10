#!/bin/bash

source config.sh 

# ===================== 配置区 =====================
> "$RESULT_SUMMARY_FILE"

output_order=(
    "bash base2new_train.sh patternnet"
    "bash base2new_train.sh rsicd"
    "bash base2new_train.sh resisc45"
    "bash base2new_train.sh mlrsnet"
    "bash base2new_test.sh patternnet"
    "bash base2new_test.sh rsicd"
    "bash base2new_test.sh resisc45"
    "bash base2new_test.sh mlrsnet"
    "bash crossdata_train.sh patternnet"
    "bash crossdata_test.sh rsicd"
    "bash crossdata_test.sh resisc45"
    "bash crossdata_test.sh mlrsnet"
    "bash domaingen_train.sh patternnetv2"
    "bash domaingen_test.sh rsicdv2"
    "bash domaingen_test.sh resisc45v2"
    "bash domaingen_test.sh mlrsnetv2"
)

# =============== 提取字段映射函数 ===============
map_task_info() {
    case "$1" in
        *"base2new_train.sh patternnet"*)   mode="base2new"; result="base"; dataset="patternnet" ;;
        *"base2new_train.sh rsicd"*)        mode="base2new"; result="base"; dataset="rsicd" ;;
        *"base2new_train.sh resisc45"*)     mode="base2new"; result="base"; dataset="resisc45" ;;
        *"base2new_train.sh mlrsnet"*)      mode="base2new"; result="base"; dataset="mlrsnet" ;;
        *"base2new_test.sh patternnet"*)    mode="base2new"; result="new"; dataset="patternnet" ;;
        *"base2new_test.sh rsicd"*)         mode="base2new"; result="new"; dataset="rsicd" ;;
        *"base2new_test.sh resisc45"*)      mode="base2new"; result="new"; dataset="resisc45" ;;
        *"base2new_test.sh mlrsnet"*)       mode="base2new"; result="new"; dataset="mlrsnet" ;;
        *"crossdata_train.sh patternnet"*)  mode="crossdata"; result="source"; dataset="patternnet" ;;
        *"crossdata_test.sh rsicd"*)        mode="crossdata"; result="target"; dataset="rsicd" ;;
        *"crossdata_test.sh resisc45"*)     mode="crossdata"; result="target"; dataset="resisc45" ;;
        *"crossdata_test.sh mlrsnet"*)      mode="crossdata"; result="target"; dataset="mlrsnet" ;;
        *"domaingen_train.sh patternnetv2"*) mode="domaingen"; result="source"; dataset="patternnetv2" ;;
        *"domaingen_test.sh rsicdv2"*)      mode="domaingen"; result="target"; dataset="rsicdv2" ;;
        *"domaingen_test.sh resisc45v2"*)   mode="domaingen"; result="target"; dataset="resisc45v2" ;;
        *"domaingen_test.sh mlrsnetv2"*)    mode="domaingen"; result="target"; dataset="mlrsnetv2" ;;
        *) echo "无法识别指令：$1"; exit 1;;
    esac
}

extract_metrics_from_log() {
    local content="$1"
    acc=$(echo "$content" | grep -E "\* accuracy:" | tail -n1 | awk -F': ' '{print $2}' | tr -d '%')
    err=$(echo "$content" | grep -E "\* error:" | tail -n1 | awk -F': ' '{print $2}' | tr -d '%')
    f1=$(echo "$content" | grep -E "\* macro_f1:" | tail -n1 | awk -F': ' '{print $2}' | tr -d '%')
}

# =============== 核心执行函数 ===============
run_and_log() {
    local cmd="$1"
    local seed="$2"
    map_task_info "$cmd"
    local full_cmd="$cmd $seed"

    # ---------- train ----------
    if [[ "$cmd" == *train.sh* ]]; then
        if grep -q "^$mode,$result,$dataset,$seed," "$RESULT_CSV_FILE" 2>/dev/null; then
            echo "[缓存] 训练结果已存在，拷贝到 SUMMARY"
            line=$(awk -F',' -v m="$mode" -v r="$result" -v d="$dataset" -v s="$seed" '($1==m && $2==r && $3==d && $4==s){print; exit}' "$RESULT_CSV_FILE")
            if [ -n "$line" ]; then
                echo "$line" >> "$RESULT_SUMMARY_FILE"
            else
                echo "匹配结果为空：$mode,$result,$dataset,$seed" >> "$RESULT_SUMMARY_FILE"
            fi
            return
        fi


        echo "[训练] 执行：$full_cmd"
        output=$($full_cmd 2>&1)
        extract_metrics_from_log "$output"
        echo "$mode,$result,$dataset,$seed,$acc,$err,$f1" | tee -a "$RESULT_CSV_FILE" >> "$RESULT_SUMMARY_FILE"
        return
    fi

    # ---------- test ----------
    local attempt=1
    local max_attempts=5
    while (( attempt <= max_attempts )); do
        echo "测试尝试 $attempt 次：$full_cmd"
        output=$($full_cmd 2>&1)

        if echo "$output" | grep -q "The results already exist in"; then
            result_dir=$(echo "$output" | grep -oP 'The results already exist in \K.*?/seed[0-9]+')
            result_dir="../$result_dir"
            log_file="$result_dir/log.txt"
            if [ -f "$log_file" ]; then
                content=$(cat "$log_file")
                extract_metrics_from_log "$content"
                echo "$mode,$result,$dataset,$seed,$acc,$err,$f1" >> "$RESULT_SUMMARY_FILE"
                return
            else
                echo "结果存在但 log 不全，删除 $result_dir"
                rm -rf "$result_dir"
                ((attempt++))
                continue
            fi
        fi

        if echo "$output" | grep -q "FileNotFoundError"; then
            model_path=$(echo "$output" | grep -oP 'Model not found at "\K[^"]+')
            seed_dir="$(echo "$model_path" | grep -oP ".*?/seed[0-9]+")"
            echo "模型不存在，删除并训练：$seed_dir"
            seed_dir="../$seed_dir"
            rm -rf "$seed_dir"
            train_cmd=$(echo "$cmd" | sed 's/test/train/')
            run_and_log "$train_cmd" "$seed"
            ((attempt++))
            continue
        fi

        extract_metrics_from_log "$output"
        if [ -n "$acc" ] && [ -n "$err" ] && [ -n "$f1" ]; then
            echo "$mode,$result,$dataset,$seed,$acc,$err,$f1" >> "$RESULT_SUMMARY_FILE"
            return
        fi

        ((attempt++))
    done
    echo "多次失败：$full_cmd"
}

# =============== 主循环 ===============
for cmd in "${output_order[@]}"; do
    for ((seed=START_RUN; seed<=END_RUN; seed++)); do
        run_and_log "$cmd" "$seed"
    done
done

echo "全部任务完成，统计结果见：$RESULT_SUMMARY_FILE"
