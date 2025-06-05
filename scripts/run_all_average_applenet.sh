#!/bin/bash

# 全局配置
source config.sh
# 存储所有 current_task_pid
declare -A child_pids
declare -A task_map

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


monitor_logfile() {
    local task_type=$1
    local phase=$2
    local model=$3
    local run=$4
    local start_run=$5
    local end_run=$6
    local logfile=$7
    local epoch=$8

    # Ensure the directory for OUTPUT_LOG_FILE exists
    mkdir -p "$(dirname $OUTPUT_LOG_FILE)"

    # Create OUTPUT_LOG_FILE if it doesn't exist
    if [[ ! -f $OUTPUT_LOG_FILE ]]; then
        touch $OUTPUT_LOG_FILE
    fi

    # Calculate the seed progress
    local total_runs=$((end_run - start_run + 1))
    local current_index=$((run - start_run + 1))  # Calculate 1-based index for the current run
    local seed_progress="seed${run}[${current_index}/${total_runs}]"

    # Construct the identifier for the specific line
    local identifier="${task_type}--${phase}--${model}"
    local initial_line="${identifier}--${seed_progress}--epoch [1/1]"

    # Ensure the line exists in the OUTPUT_LOG_FILE
    if ! grep -q "^${identifier}" $OUTPUT_LOG_FILE; then
        echo "$initial_line" >> $OUTPUT_LOG_FILE
    fi

    # Function to sort OUTPUT_LOG_FILE based on global output_order
    sort_output_file() {
        local temp_file=$(mktemp)

        for order in "${output_order[@]}"; do
            # Replace commas in order with '--' to match file format
            local formatted_order="${order//,/--}"
            grep "^${formatted_order}" "$OUTPUT_LOG_FILE" >> "$temp_file"
        done

        mv "$temp_file" "$OUTPUT_LOG_FILE"
    }

    # Monitor the logfile for epoch updates
    while true; do
        local latest_epoch = ""
        # Reverse the log file and search for the latest epoch information
        if [[ $epoch == 0 ]]; then
            last_lines=$(tail -n 7 $logfile)
            latest_epoch=$(echo "$last_lines" | grep -oP 'macro_f1: \K[0-9]+\.[0-9]+' | tail -n 1)
            # latest_epoch=$(tail -n 20 "$logfile" 2>&1 | grep -m 1 "\* macro_f1: [0-9]\+\(\.[0-9]\+\)\?%")
        else
            latest_epoch=$(tail -n 20 "$logfile" 2>&1 | grep -m 1 "epoch \[[0-9]\+/[0-9]\+\]")
        fi
        local epoch_info = ""
        if [[ -n "$latest_epoch" ]]; then
            # Extract the epoch information
            if [[ $epoch == 0 ]]; then
                epoch_info="epoch [1/1]"
            else
                epoch_info=$(echo "$latest_epoch" | grep -o "epoch \[[0-9]\+/[0-9]\+\]")
            fi

            # Update the corresponding line in OUTPUT_LOG_FILE
            sed -i "/^${identifier}/c\\${identifier}--${seed_progress}--${epoch_info}" $OUTPUT_LOG_FILE

            # Sort the OUTPUT_LOG_FILE to maintain the desired order
            sort_output_file

            # Check if the epoch is complete
            if [[ "$epoch_info" == "epoch [${epoch}/${epoch}]" ]]; then
                # echo "Task ${identifier} completed with ${epoch_info}. Exiting monitor."
                break
            elif [[ $epoch == 0 ]]; then
                # echo "Task ${identifier} completed with ${epoch_info}. Exiting monitor."
                # sleep 2
                break
            fi
        fi

        # Sleep briefly to reduce CPU usage
        sleep 1
    done
}



log_to_csv() {
    local total_type=$1
    local type=$2
    local model=$3
    local run=$4
    local log_file=$5

    # 从日志文件提取最后几行数据
    if [ ! -f "$log_file" ]; then
        echo "Error: Log file $log_file not found."
        exit 1
    fi

    # 获取最后 7 行日志（根据实际情况可以调整行数）
    last_lines=$(tail -n 7 $log_file)

    # 使用 grep 提取 accuracy, error 和 macro_f1
    accuracy=$(echo "$last_lines" | grep -oP 'accuracy: \K[0-9]+\.[0-9]+' | tail -n 1)
    error=$(echo "$last_lines" | grep -oP 'error: \K[0-9]+\.[0-9]+' | tail -n 1)
    macro_f1=$(echo "$last_lines" | grep -oP 'macro_f1: \K[0-9]+\.[0-9]+' | tail -n 1)

    # 如果没有提取到数据，给出错误提示
    if [ -z "$accuracy" ] || [ -z "$error" ] || [ -z "$macro_f1" ]; then
        echo "Error: Missing accuracy, error, or macro_f1 in the last log entries."
        exit 1
    fi

    # 将数据写入CSV文件
    echo "$total_type,$type,$model,$run,$accuracy,$error,$macro_f1" >> $RESULT_CSV_FILE
}

# 执行 base2new 部分
base2new_patternnet() {
    models=("patternnet")
    seed=$1

    for model in "${models[@]}"; do
        # 训练日志
        common_log_file="${OUTPUT_BASE}/base2new/train_base/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
        bash base2new_train.sh $model $seed > /dev/null 2>&1 &
        base2new_train_pid=$!
        child_pids[$base2new_train_pid]="base2new_train_pid_${model}_SEED${seed}"
        monitor_logfile "base2new" "base" $model $seed $START_RUN $END_RUN $common_log_file $EPOCH &
        base2new_train_pid_log=$!
        child_pids[$base2new_train_pid_log]="base2new_train_pid_log_${model}_SEED${seed}"
        wait $base2new_train_pid
        log_to_csv "base2new" "base" $model $seed $common_log_file

        # 测试日志
        common_log_file="${OUTPUT_BASE}/base2new/test_new/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
        bash base2new_test.sh $model $seed > /dev/null 2>&1 &
        base2new_test_pid=$!
        child_pids[$base2new_test_pid]="base2new_test_pid_${model}_SEED${seed}"
        monitor_logfile "base2new" "new" $model $seed $START_RUN $END_RUN $common_log_file 0 &
        base2new_test_pid_log=$!
        child_pids[$base2new_test_pid_log]="base2new_test_pid_log_${model}_SEED${seed}"
        wait $base2new_test_pid
        log_to_csv "base2new" "new" $model $seed $common_log_file
    done
}
base2new_mlrsnet() {
    models=("mlrsnet")
    seed=$1

    for model in "${models[@]}"; do
        # 训练日志
        common_log_file="${OUTPUT_BASE}/base2new/train_base/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
        bash base2new_train.sh $model $seed > /dev/null 2>&1 &
        base2new_train_pid=$!
        child_pids[$base2new_train_pid]="base2new_train_pid_${model}_SEED${seed}"
        monitor_logfile "base2new" "base" $model $seed $START_RUN $END_RUN $common_log_file $EPOCH &
        base2new_train_pid_log=$!
        child_pids[$base2new_train_pid_log]="base2new_train_pid_log_${model}_SEED${seed}"
        wait $base2new_train_pid
        log_to_csv "base2new" "base" $model $seed $common_log_file

        # 测试日志
        common_log_file="${OUTPUT_BASE}/base2new/test_new/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
        bash base2new_test.sh $model $seed > /dev/null 2>&1 &
        base2new_test_pid=$!
        child_pids[$base2new_test_pid]="base2new_test_pid_${model}_SEED${seed}"
        monitor_logfile "base2new" "new" $model $seed $START_RUN $END_RUN $common_log_file 0 &
        base2new_test_pid_log=$!
        child_pids[$base2new_test_pid_log]="base2new_test_pid_log_${model}_SEED${seed}"
        wait $base2new_test_pid
        log_to_csv "base2new" "new" $model $seed $common_log_file
    done
}
base2new_resisc45() {
    models=("resisc45")
    seed=$1

    for model in "${models[@]}"; do
        # 训练日志
        common_log_file="${OUTPUT_BASE}/base2new/train_base/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
        bash base2new_train.sh $model $seed > /dev/null 2>&1 &
        base2new_train_pid=$!
        child_pids[$base2new_train_pid]="base2new_train_pid_${model}_SEED${seed}"
        monitor_logfile "base2new" "base" $model $seed $START_RUN $END_RUN $common_log_file $EPOCH &
        base2new_train_pid_log=$!
        child_pids[$base2new_train_pid_log]="base2new_train_pid_log_${model}_SEED${seed}"
        wait $base2new_train_pid
        log_to_csv "base2new" "base" $model $seed $common_log_file

        # 测试日志
        common_log_file="${OUTPUT_BASE}/base2new/test_new/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
        bash base2new_test.sh $model $seed > /dev/null 2>&1 &
        base2new_test_pid=$!
        child_pids[$base2new_test_pid]="base2new_test_pid_log_${model}_SEED${seed}"
        monitor_logfile "base2new" "new" $model $seed $START_RUN $END_RUN $common_log_file 0 &
        base2new_test_pid_log=$!
        child_pids[$base2new_test_pid_log]="base2new_test_pid_log_${model}_SEED${seed}"
        wait $base2new_test_pid
        log_to_csv "base2new" "new" $model $seed $common_log_file
    done
}
base2new_rsicd() {
    models=("rsicd")
    seed=$1

    for model in "${models[@]}"; do
        # 训练日志
        common_log_file="${OUTPUT_BASE}/base2new/train_base/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
        bash base2new_train.sh $model $seed > /dev/null 2>&1 &
        base2new_train_pid=$!
        child_pids[$base2new_train_pid]="base2new_train_pid_${model}_SEED${seed}"
        monitor_logfile "base2new" "base" $model $seed $START_RUN $END_RUN $common_log_file $EPOCH &
        base2new_train_pid_log=$!
        child_pids[$base2new_train_pid_log]="base2new_train_pid_log_${model}_SEED${seed}"
        wait $base2new_train_pid
        log_to_csv "base2new" "base" $model $seed $common_log_file

        # 测试日志
        common_log_file="${OUTPUT_BASE}/base2new/test_new/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
        bash base2new_test.sh $model $seed > /dev/null 2>&1 &
        base2new_test_pid=$!
        child_pids[$base2new_test_pid]="base2new_test_pid_log_${model}_SEED${seed}"
        monitor_logfile "base2new" "new" $model $seed $START_RUN $END_RUN $common_log_file 0 &
        base2new_test_pid_log=$!
        child_pids[$base2new_test_pid_log]="base2new_test_pid_log_${model}_SEED${seed}"
        wait $base2new_test_pid
        log_to_csv "base2new" "new" $model $seed $common_log_file
    done
}
# 执行 crossdata 部分
crossdata() {
    source_models=("patternnet")
    target_models=("rsicd" "resisc45" "mlrsnet")
    seed=$1

    for source in "${source_models[@]}"; do
        common_log_file="${OUTPUT_BASE}/crosstransfer/${source}/${TRAINER}/${CFG}_shots${SHOTS}/seed${seed}/log.txt"
        bash crossdata_train.sh $source $seed > /dev/null 2>&1 &
        crossdata_source_pid=$!
        child_pids[$crossdata_source_pid]="crossdata_source_pid_${model}_SEED${seed}"
        monitor_logfile "crossdata" "source" $source $seed $START_RUN $END_RUN $common_log_file $EPOCH &
        crossdata_source_pid_log=$!
        child_pids[$crossdata_source_pid_log]="crossdata_source_pid_log_${model}_SEED${seed}"
        wait $crossdata_source_pid
        log_to_csv "crossdata" "source" $source $seed $common_log_file

        for target in "${target_models[@]}"; do
            common_log_file="${OUTPUT_BASE}/crosstransfer/tests/${TRAINER}/${CFG}_shots${SHOTS}/${target}/seed${seed}/log.txt"
            bash crossdata_test.sh $target $seed > /dev/null 2>&1 &
            crossdata_target_pid=$!
            child_pids[$crossdata_target_pid]="crossdata_target_pid_${model}_SEED${seed}"
            monitor_logfile "crossdata" "target" $target $seed $START_RUN $END_RUN $common_log_file 0 &
            crossdata_target_pid_log=$!
            child_pids[$crossdata_target_pid_log]="crossdata_target_pid_log_${model}_SEED${seed}"
            wait $crossdata_target_pid
            log_to_csv "crossdata" "target" $target $seed $common_log_file
        done
    done
}

# 执行 domaingen 部分
domaingen() {
    source_models=("patternnetv2")
    target_models=("rsicdv2" "resisc45v2" "mlrsnetv2")
    seed=$1

    for source in "${source_models[@]}"; do
        common_log_file="${OUTPUT_BASE}/domain_generalization/${source}/${TRAINER}/${CFG}_shots${SHOTS}/seed${seed}/log.txt"
        bash domaingen_train.sh $source $seed > /dev/null 2>&1 &
        domaingen_source_pid=$!
        child_pids[$domaingen_source_pid]="domaingen_source_pid_${model}_SEED${seed}"
        monitor_logfile "domaingen" "source" $source $seed $START_RUN $END_RUN $common_log_file $EPOCH &
        domaingen_source_pid_log=$!
        child_pids[$domaingen_source_pid_log]="domaingen_source_pid_log_${model}_SEED${seed}"
        wait $domaingen_source_pid
        log_to_csv "domaingen" "source" $source $seed $common_log_file

        for target in "${target_models[@]}"; do
            common_log_file="${OUTPUT_BASE}/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/${target}/seed${seed}/log.txt"
            bash domaingen_test.sh $target $seed > /dev/null 2>&1 &
            domaingen_target_pid=$!
            child_pids[$domaingen_target_pid]="domaingen_target_pid_${model}_SEED${seed}"
            monitor_logfile "domaingen" "target" $target $seed $START_RUN $END_RUN $common_log_file 0 &
            domaingen_target_pid_log=$!
            child_pids[$domaingen_target_pid_log]="domaingen_target_pid_log_${model}_SEED${seed}"
            wait $domaingen_target_pid
            log_to_csv "domaingen" "target" $target $seed $common_log_file
        done
    done
}


# 监测显存函数
check_free_memory() {
    # 使用 nvidia-smi 检测显存，获取最大空闲显存（单位 MB）
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -n 1
}


# 删除输出文件夹函数
clean_output_directory() {
    local task_name=$1
    local seed=$2
    local output_dir=""
    local output_dir_test=""
    local output_dir_source_1""
    local output_dir_source_2=""
    local output_dir_source_3=""

    # 定义任务名称列表
    local task_list=()

    # 根据任务名称确定输出文件夹路径
    case "$task_name" in
        base2new_patternnet)
            task_list=("base2new_patternnet")
            ;;
        base2new_mlrsnet)
            task_list=("base2new_mlrsnet")
            ;;
        base2new_resisc45)
            task_list=("base2new_resisc45")
            ;;
        base2new_rsicd)
            task_list=("base2new_rsicd")
            ;;
        crossdata)
            task_list=("crossdata")
            ;;
        domaingen)
            task_list=("domaingen")
            ;;
        *)
            echo "未知任务名称: $task_name"
            return 1
            ;;
    esac

    # 遍历每个任务名称
    for task_name in "${task_list[@]}"; do
        echo "处理任务: $task_name"

        # 根据任务名称确定输出文件夹路径
        case "$task_name" in
            base2new_patternnet)
                output_dir="${OUTPUT_BASE}/base2new/train_base/patternnet/shots_${SHOTS}/${TRAINER}/${CFG}"
                output_dir_test="${OUTPUT_BASE}/base2new/test_new/patternnet/shots_${SHOTS}/${TRAINER}/${CFG}"
                ;;

            base2new_mlrsnet)
                output_dir="${OUTPUT_BASE}/base2new/train_base/mlrsnet/shots_${SHOTS}/${TRAINER}/${CFG}"
                output_dir_test="${OUTPUT_BASE}/base2new/test_new/mlrsnet/shots_${SHOTS}/${TRAINER}/${CFG}"
                ;;

            base2new_resisc45)
                output_dir="${OUTPUT_BASE}/base2new/train_base/resisc45/shots_${SHOTS}/${TRAINER}/${CFG}"
                output_dir_test="${OUTPUT_BASE}/base2new/test_new/resisc45/shots_${SHOTS}/${TRAINER}/${CFG}"
                ;;

            base2new_rsicd)
                output_dir="${OUTPUT_BASE}/base2new/train_base/rsicd/shots_${SHOTS}/${TRAINER}/${CFG}"
                output_dir_test="${OUTPUT_BASE}/base2new/test_new/rsicd/shots_${SHOTS}/${TRAINER}/${CFG}"
                ;;

            crossdata)
                output_dir="${OUTPUT_BASE}/crosstransfer/patternnet/${TRAINER}/${CFG}_shots${SHOTS}"
                output_dir_source_1="${OUTPUT_BASE}/crosstransfer/tests/${TRAINER}/${CFG}_shots${SHOTS}/mlrsnet"
                output_dir_source_2="${OUTPUT_BASE}/crosstransfer/tests/${TRAINER}/${CFG}_shots${SHOTS}/resisc45"
                output_dir_source_3="${OUTPUT_BASE}/crosstransfer/tests/${TRAINER}/${CFG}_shots${SHOTS}/rsicd"
                ;;

            domaingen)
                output_dir="${OUTPUT_BASE}/domain_generalization/patternnetv2/${TRAINER}/${CFG}_shots${SHOTS}"
                output_dir_source_1="${OUTPUT_BASE}/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/mlrsnetv2"
                output_dir_source_2="${OUTPUT_BASE}/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/resisc45v2"
                output_dir_source_3="${OUTPUT_BASE}/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/rsicdv2"
                ;;

            *)
                echo "未知任务名称: $task_name"
                continue  # 如果是未知任务，跳过当前任务
                ;;
        esac

        # 清理训练文件夹
        if [[ -d $output_dir ]]; then
            seed_folder="${output_dir}/seed${seed}"
            if [[ -d $seed_folder ]]; then
                echo "清理输出目录: $seed_folder"
                rm -rf "$seed_folder"
            else
                echo "目录不存在: $seed_folder"
            fi
        else
            echo "目录不存在: $output_dir"
        fi

        # 清理测试文件夹
        if [[ -n $output_dir_test && -d $output_dir_test ]]; then
            seed_folder="${output_dir_test}/seed${seed}"
            if [[ -d $seed_folder ]]; then
                echo "清理测试目录: $seed_folder"
                rm -rf "$seed_folder"
            else
                echo "目录不存在: $seed_folder"
            fi
        elif [[ -n $output_dir_test ]]; then
            echo "目录不存在: $output_dir_test"
        fi

        # 清理迁移文件夹
        if [[ -n $output_dir_source_1 && -d $output_dir_source_1 ]]; then
            seed_folder="${output_dir_source_1}/seed${seed}"
            if [[ -d $seed_folder ]]; then
                echo "清理迁移目录: $seed_folder"
                rm -rf "$seed_folder"
            else
                echo "目录不存在: $seed_folder"
            fi
        elif [[ -n $output_dir_source_1 ]]; then
            echo "目录不存在: $output_dir_source_1"
        fi
        if [[ -n $output_dir_source_2 && -d $output_dir_source_2 ]]; then
            seed_folder="${output_dir_source_2}/seed${seed}"
            if [[ -d $seed_folder ]]; then
                echo "清理迁移目录: $seed_folder"
                rm -rf "$seed_folder"
            else
                echo "目录不存在: $seed_folder"
            fi
        elif [[ -n $output_dir_source_2 ]]; then
            echo "目录不存在: $output_dir_source_2"
        fi
        if [[ -n $output_dir_source_3 && -d $output_dir_source_3 ]]; then
            seed_folder="${output_dir_source_3}/seed${seed}"
            if [[ -d $seed_folder ]]; then
                echo "清理迁移目录: $seed_folder"
                rm -rf "$seed_folder"
            else
                echo "目录不存在: $seed_folder"
            fi
        elif [[ -n $output_dir_source_3 ]]; then
            echo "目录不存在: $output_dir_source_3"
        fi
    done
}

# 定义等待函数
wait_with_logging() {
    local wait_time=$1  # 总等待时间（秒）
    local interval=5    # 打印信息的时间间隔（秒）
    local elapsed_time=0
    free_memory=$(check_free_memory)
    echo "当前剩余显存: ${free_memory} MB，开始等待 $((wait_time / 60)) 分钟（$wait_time 秒）..."

    while [[ $elapsed_time -lt $wait_time ]]; do
        echo "Current Training Progress (from $OUTPUT_LOG_FILE):"
        tail -n 20 "$OUTPUT_LOG_FILE"  # 显示日志文件的最后 20 行
        sleep $interval
        elapsed_time=$((elapsed_time + interval))
        free_memory=$(check_free_memory)
        echo "已等待 $elapsed_time 秒，剩余 $((wait_time - elapsed_time)) 秒，当前剩余显存: ${free_memory} MB..."

        if [[ $free_memory -ge $CUDA_MEMORY ]]; then
            echo "当前剩余显存: ${free_memory} MB，提前结束，共等待 $elapsed_time 秒..."
            break
        fi
    done

    echo "等待结束，继续执行任务。"
}


# 调度函数
scheduler() {
    # 原始任务列表
    base_tasks=("crossdata" "domaingen" "base2new_patternnet" "base2new_mlrsnet" "base2new_resisc45" "base2new_rsicd")
    # 最大同时运行任务数
    running_pids=()

    # 生成任务组合
    tasks=()
    for seed in $(seq $START_RUN $END_RUN); do
        for task in "${base_tasks[@]}"; do
            tasks+=("${task}_seed${seed}")
        done
    done

    while [[ ${#tasks[@]} -gt 0 ]]; do
        echo "当前待运行任务列表: ${tasks[@]}"
        echo "当前运行中的任务 PID 列表: ${running_pids[@]}"

        # 清理已完成的任务
        new_running_pids=()
        for pid in "${running_pids[@]}"; do
            if ps -p $pid > /dev/null; then
                new_running_pids+=($pid)
            else
                echo "任务 (PID: $pid) 已完成。"
            fi
        done
        running_pids=("${new_running_pids[@]}")

        # 检查是否可以启动新的任务
        if [[ ${#running_pids[@]} -lt $MAX_TASK_NUM && ${#tasks[@]} -gt 0 ]]; then

            # 检查剩余显存
            free_memory=$(check_free_memory)
            echo "当前剩余显存: ${free_memory} MB"

            if [[ $free_memory -ge $CUDA_MEMORY ]]; then
                # 从任务列表中获取第一个任务并移出
                current_task_seed=${tasks[0]}
                tasks=("${tasks[@]:1}")

                # 分离任务名和 seed
                current_task=${current_task_seed%_seed*}
                seed=${current_task_seed#*_seed}

                echo "启动任务: $current_task (seed: $seed)"
                # 调用函数任务，传入 seed 参数，放到后台执行
                $current_task $seed > /dev/null 2>&1 &
                current_task_pid=$!

                # 等待 20 秒，检查任务是否成功启动
                sleep 20
                if ps -p $current_task_pid > /dev/null; then
                    running_pids+=($current_task_pid)
                    task_map[$current_task_pid]="$current_task_seed"
                    echo "任务 $current_task (seed: $seed) 成功启动 (PID: $current_task_pid)。"
                    wait_with_logging 300   
                    continue
                else
                    echo "任务 $current_task (seed: $seed) 启动失败，尝试重新启动..."

                    # 无限循环直到任务启动成功
                    while true; do
                        clean_output_directory "$current_task" $seed

                        echo "重新启动任务: $current_task (seed: $seed)"
                        $current_task $seed > /dev/null 2>&1 &
                        current_task_pid=$!

                        # 等待 10 秒检查任务是否启动成功
                        sleep 10
                        if ps -p $current_task_pid > /dev/null; then
                            running_pids+=($current_task_pid)
                            task_map[$current_task_pid]="$current_task_seed"
                            echo "任务 $current_task (seed: $seed) 成功启动 (PID: $current_task_pid)。"
                            wait_with_logging 300
                            break
                        fi
                    done
                fi
                wait_with_logging 300
            else
                wait_with_logging 300
            fi
        else
            echo "达到最大并行任务数或无任务可运行，等待任务完成..."
            wait_with_logging 300
        fi

    done

    echo "所有任务已完成！开始等待调度"
}

average() {
    # 初始化总和和计数器
    declare -A total_accuracy
    declare -A total_error
    declare -A total_macro_f1
    declare -A count

    # 计算每个组合（total_type, type, model）对应的总和和计数
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

            # 累加每个组合的总和和计数
            total_accuracy["$key"]=$(echo "${total_accuracy[$key]:-0} + $accuracy" | bc)
            total_error["$key"]=$(echo "${total_error[$key]:-0} + $error" | bc)
            total_macro_f1["$key"]=$(echo "${total_macro_f1[$key]:-0} + $macro_f1" | bc)
            count["$key"]=$((count["$key"] + 1))
        fi
    done < $RESULT_CSV_FILE
    
    # 将每个组合的平均值按顺序写入到平均值文件
    for key in "${output_order[@]}"; do
        accuracy_sum=${total_accuracy["$key"]}
        error_sum=${total_error["$key"]}
        macro_f1_sum=${total_macro_f1["$key"]}
        num_entries=${count["$key"]}

        # 计算平均值
        avg_accuracy=$(echo "scale=2; $accuracy_sum / $num_entries" | bc)
        avg_error=$(echo "scale=2; $error_sum / $num_entries" | bc)
        avg_macro_f1=$(echo "scale=2; $macro_f1_sum / $num_entries" | bc)

        # 将结果输出到平均值文件
        echo "$key,$avg_accuracy,$avg_error,$avg_macro_f1" >> $AVERAGE_CSV_FILE
    done
}


# 调用调度器
scheduler

# 等待所有任务完成
while true; do
    all_done=true  # 假设所有任务都完成

    for pid in "${!task_map[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            all_done=false  # 如果某个任务还在运行，更新状态
        else
            unset task_map[$pid]  # 从任务列表中移除已完成任务
        fi
    done

    # 刷新并显示日志文件内容
    clear  # 清屏
    echo "Current Training Progress (from $OUTPUT_LOG_FILE):"
    tail -n 20 "$OUTPUT_LOG_FILE"  # 显示日志文件的最后 20 行

    # 如果所有任务都完成，退出循环
    if $all_done; then
        break
    fi

    sleep 5  # 每 5 秒刷新一次
done


clear  # 清屏
# 输出平均值到 CSV 文件
average

echo "平均值计算完毕，可随时关闭输出日志..."

