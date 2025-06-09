#!/bin/bash

source config.sh 


output_order=(
    "bash base2new_test.sh patternnet"
    "bash base2new_test.sh rsicd"
    "bash base2new_test.sh resisc45"
    "bash base2new_test.sh mlrsnet"
    "bash crossdata_test.sh patternnet"
    "bash crossdata_test.sh rsicd"
    "bash crossdata_test.sh resisc45"
    "bash crossdata_test.sh mlrsnet"
    "bash domaingen_test.sh patternnetv2"
    "bash domaingen_test.sh rsicdv2"
    "bash domaingen_test.sh resisc45v2"
    "bash domaingen_test.sh mlrsnetv2"
)


> "$RESULT_SUMMARY_FILE"

# 判断任务类型和阶段
get_type_info() {
    local cmd=$1
    if [[ $cmd == *"base2new"* ]]; then
        TYPE="base2new"
        if [[ $cmd == *"patternnet" || $cmd == *"rsicd" || $cmd == *"resisc45" || $cmd == *"mlrsnet" ]]; then
            STAGE="base"
        else
            STAGE="new"
        fi
    elif [[ $cmd == *"crossdata"* ]]; then
        TYPE="crossdata"
        if [[ $cmd == *"patternnet"* ]]; then
            STAGE="source"
        else
            STAGE="target"
        fi
    elif [[ $cmd == *"domaingen"* ]]; then
        TYPE="domaingen"
        if [[ $cmd == *"patternnetv2"* ]]; then
            STAGE="source"
        else
            STAGE="target"
        fi
    else
        TYPE="unknown"
        STAGE="unknown"
    fi
}

# 执行并解析输出，直到获取结果或达到最大尝试次数
run_until_success() {
    local cmd="$1"
    local seed="$2"
    local full_cmd="$cmd $seed"
    local attempt=1
    local max_attempts=5

    # 训练方式、结果类型、数据集名映射
    case "$cmd" in
        *"base2new_test.sh patternnet"*)   mode="base2new"; result="new"; dataset="patternnet" ;;
        *"base2new_test.sh rsicd"*)        mode="base2new"; result="new"; dataset="rsicd" ;;
        *"base2new_test.sh resisc45"*)     mode="base2new"; result="new"; dataset="resisc45" ;;
        *"base2new_test.sh mlrsnet"*)      mode="base2new"; result="new"; dataset="mlrsnet" ;;
        *"crossdata_test.sh patternnet"*)  mode="crossdata"; result="source"; dataset="patternnet" ;;
        *"crossdata_test.sh rsicd"*)       mode="crossdata"; result="target"; dataset="rsicd" ;;
        *"crossdata_test.sh resisc45"*)    mode="crossdata"; result="target"; dataset="resisc45" ;;
        *"crossdata_test.sh mlrsnet"*)     mode="crossdata"; result="target"; dataset="mlrsnet" ;;
        *"domaingen_test.sh patternnetv2"*) mode="domaingen"; result="source"; dataset="patternnetv2" ;;
        *"domaingen_test.sh rsicdv2"*)     mode="domaingen"; result="target"; dataset="rsicdv2" ;;
        *"domaingen_test.sh resisc45v2"*)  mode="domaingen"; result="target"; dataset="resisc45v2" ;;
        *"domaingen_test.sh mlrsnetv2"*)   mode="domaingen"; result="target"; dataset="mlrsnetv2" ;;
        *)                                 mode="unknown"; result="unknown"; dataset="$cmd" ;;
    esac

    while (( attempt <= max_attempts )); do
        echo "尝试第 $attempt 次运行：$full_cmd"
        output=$($full_cmd 2>&1)
        echo "$output"

        # 情况1：结果目录已存在
        if echo "$output" | grep -q "The results already exist in"; then
            result_line=$(echo "$output" | grep "The results already exist in")
            result_dir=$(echo "$result_line" | sed -E 's/^.*The results already exist in //')
            result_dir="../$result_dir"  # 路径修正

            log_file="$result_dir/log.txt"
            echo "📁 结果已存在：$result_dir"

            if [ -f "$log_file" ]; then
                echo "📄 检查日志文件：$log_file"

                acc_line=$(tail -n 10 "$log_file" | grep -E "\* accuracy:")
                err_line=$(tail -n 10 "$log_file" | grep -E "\* error:")
                f1_line=$(tail -n 10 "$log_file" | grep -E "\* macro_f1:")

                if [ -n "$acc_line" ] && [ -n "$err_line" ] && [ -n "$f1_line" ]; then
                    accuracy=$(echo "$acc_line" | awk -F': ' '{print $2}' | tr -d '%')
                    error=$(echo "$err_line" | awk -F': ' '{print $2}' | tr -d '%')
                    macro_f1=$(echo "$f1_line" | awk -F': ' '{print $2}' | tr -d '%')

                    echo "$mode,$result,$dataset,$seed,$accuracy,$error,$macro_f1" >> "$RESULT_SUMMARY_FILE"
                    echo "✅ 直接从 log.txt 读取结果：$accuracy, $error, $macro_f1"
                    return
                else
                    echo "⚠️ log.txt 不完整，删除目录重新执行测试"
                    rm -rf "$result_dir"
                    ((attempt++))
                    continue
                fi
            else
                echo "⚠️ log.txt 不存在，删除目录重新执行测试"
                rm -rf "$result_dir"
                ((attempt++))
                continue
            fi
        fi


        # 情况2：模型文件缺失
        if echo "$output" | grep -q "FileNotFoundError"; then
            model_path=$(echo "$output" | grep -oP 'Model not found at "\K[^"]+')
            seed_dir=$(echo "$model_path" | grep -oP '.*?/seed[0-9]+')
            seed_dir="../$seed_dir"

            echo "🧹 删除模型目录：$seed_dir"
            [ -d "$seed_dir" ] && rm -rf "$seed_dir"

            train_cmd=$(echo "$cmd" | sed 's/test/train/')
            full_train_cmd="$train_cmd $seed"
            echo "🚀 重新训练：$full_train_cmd"
            $full_train_cmd

            ((attempt++))
            continue
        fi

        # 情况3：提取测试结果
        if echo "$output" | grep -q "\* accuracy:"; then
            accuracy=$(echo "$output" | grep -E "\* accuracy:" | awk -F': ' '{print $2}' | tr -d '%')
            error=$(echo "$output" | grep -E "\* error:" | awk -F': ' '{print $2}' | tr -d '%')
            macro_f1=$(echo "$output" | grep -E "\* macro_f1:" | awk -F': ' '{print $2}' | tr -d '%')

            if [ -n "$accuracy" ] && [ -n "$error" ] && [ -n "$macro_f1" ]; then
                echo "$mode,$result,$dataset,$seed,$accuracy,$error,$macro_f1" >> "$RESULT_SUMMARY_FILE"
                echo "✅ 成功写入结果：$mode,$result,$dataset,$seed,$accuracy,$error,$macro_f1"
                return
            fi
        fi

        ((attempt++))
    done

    echo "❌ 多次尝试失败：$full_cmd" >> "$RESULT_SUMMARY_FILE"
}

# 主循环
for cmd in "${output_order[@]}"; do
    for ((seed=$START_RUN; seed<=$END_RUN; seed++)); do
        run_until_success "$cmd" "$seed"
    done
done

echo "✅ 所有任务执行完毕，结果已保存至 $RESULT_SUMMARY_FILE"
