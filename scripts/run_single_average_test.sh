#!/usr/bin/env bash
set -euo pipefail

############################
# 0) 基础配置
############################
TRAINER=${TRAINER:-MPPLe}     # MPPLe PromptSRC MaPLe CLIP_Adapter
CFG=${CFG:-vit_b16_c8}     # vit_b16_c4 vit_b32_c4 vit_b16_c16 vit_b32_c16
# SUB=all
# NCTX=4  # number of context tokens
SHOT_TRIES=${SHOT_TRIES:-3}
START_RUN=${START_RUN:-1}
END_RUN=${END_RUN:-3}

# 加载公共 config；config.sh 内部会自动加载 local_config.sh / .env。
CONFIG_FILE=${CONFIG_FILE:-./config.sh}
# shellcheck source=/dev/null
source "$CONFIG_FILE"

if [[ -n "${LOCAL_CONFIG_FILES_LOADED:-}" ]]; then
  echo "[CONFIG] local overrides loaded: ${LOCAL_CONFIG_FILES_LOADED}"
fi

# MAX_TASK_NUM 由 config.sh、local_config.sh、.env 或环境变量控制；如需单任务运行，可用 MAX_TASK_NUM=1 bash run_single_average_test.sh
MAX_TASK_NUM=${MAX_TASK_NUM:-1}

mkdir -p "$OUTPUT_BASE"
mkdir -p "$RUNLOG_DIR"
mkdir -p "$TASKLOG_DIR"

declare -A task_map
declare -a running_pids
declare -a tasks
declare -A mem_est_mb   # key -> 预算（峰值+margin）
declare -A mem_peak_mb  # key -> 峰值
declare -A pid_mem_key  # wrapper pid -> key
declare -A pid_res_mb   # wrapper pid -> reserved mb
declare -A pid_peak_mb  # wrapper pid -> observed peak mb
running_pids=()
tasks=()
TOTAL_TASKS=0
DONE_TASKS=0
FAILED_TASKS=0
GPU_MEM_CACHE_TS=0
GPU_MEM_CACHE_VAL=0
TOTAL_RESERVED_MB=0
LAST_MEM_SAMPLE_TS=0
LAST_WAIT_PID=""
LAST_WAIT_RC=0

# 本次要跑的任务列表
# "base2new_patternnet" "base2new_mlrsnet" "base2new_resisc45" "base2new_rsicd"
# "crossdata_patternnet" "crossdata_mlrsnet" "crossdata_resisc45" "crossdata_rsicd"
# "domaingen_patternnetv2" "domaingen_mlrsnetv2" "domaingen_resisc45v2" "domaingen_rsicdv2"
default_run_order=("base2new_patternnet" "base2new_mlrsnet" "base2new_resisc45" "base2new_rsicd"
          "domaingen_patternnetv2" "domaingen_mlrsnetv2" "domaingen_resisc45v2" "domaingen_rsicdv2")

# 本地覆盖运行清单：
# 1) local_config.sh: RUN_ORDER=("base2new_rsicd" "domaingen_patternnetv2" "domaingen_rsicdv2")
# 2) .env: RUN_ORDER_STR="base2new_rsicd domaingen_patternnetv2 domaingen_rsicdv2"
if declare -p RUN_ORDER >/dev/null 2>&1; then
  run_order=("${RUN_ORDER[@]}")
elif [[ -n "${RUN_ORDER_STR:-}" ]]; then
  read -r -a run_order <<< "$RUN_ORDER_STR"
else
  run_order=("${default_run_order[@]}")
fi
# ====== 任务定义（只在这里维护一次）============
crossdata_source_models=("patternnet")
crossdata_target_models=("rsicd" "resisc45" "mlrsnet")
domaingen_source_models=("patternnetv2")
domaingen_target_models=("rsicdv2" "resisc45v2" "mlrsnetv2")
declare -a crossdata_sources_enabled=()
declare -a crossdata_targets_enabled=()
declare -a domaingen_sources_enabled=()
declare -a domaingen_targets_enabled=()
declare -a base2new_models_enabled=()

# 生成后的真正执行顺序
declare -a run_order_exec=()

# 可视化清单、展示清单也自动生成
declare -a VIS_ITEMS=()
declare -a base_output_order=()

add_unique() { # add_unique array_name value
  local -n arr="$1"; local v="$2"; local e
  for e in "${arr[@]}"; do [[ "$e" == "$v" ]] && return 0; done
  arr+=("$v")
}
in_list() { local x="$1"; shift; local e; for e in "$@"; do [[ "$e" == "$x" ]] && return 0; done; return 1; }

build_plan_from_tokens() {
  local -n tokens="$1"

  # 清空（避免重复调用时叠加）
  crossdata_sources_enabled=()
  crossdata_targets_enabled=()
  domaingen_sources_enabled=()
  domaingen_targets_enabled=()
  base2new_models_enabled=()
  run_order_exec=()

  local seen_cross=0 seen_domain=0
  local t name

  for t in "${tokens[@]}"; do
    if [[ "$t" == base2new_* ]]; then
      name="${t#base2new_}"
      add_unique base2new_models_enabled "$name"
      add_unique run_order_exec "$t"
      continue
    fi

    if [[ "$t" == crossdata_* ]]; then
      name="${t#crossdata_}"
      seen_cross=1

      if in_list "$name" "${crossdata_source_models[@]}"; then
        add_unique crossdata_sources_enabled "$name"
      elif in_list "$name" "${crossdata_target_models[@]}"; then
        add_unique crossdata_targets_enabled "$name"
      else
        echo "[WARN] unknown crossdata token: $t" >&2
      fi
      continue
    fi

    if [[ "$t" == domaingen_* ]]; then
      name="${t#domaingen_}"
      seen_domain=1

      if in_list "$name" "${domaingen_source_models[@]}"; then
        add_unique domaingen_sources_enabled "$name"
      elif in_list "$name" "${domaingen_target_models[@]}"; then
        add_unique domaingen_targets_enabled "$name"
      else
        echo "[WARN] unknown domaingen token: $t" >&2
      fi
      continue
    fi

    echo "[WARN] unknown token: $t" >&2
  done

  ########################################
  # 规则：有 target 就必须有 source（自动补）
  ########################################
  if [[ ${#crossdata_targets_enabled[@]} -gt 0 && ${#crossdata_sources_enabled[@]} -eq 0 ]]; then
    crossdata_sources_enabled=("${crossdata_source_models[0]}")
  fi
  if [[ ${#domaingen_targets_enabled[@]} -gt 0 && ${#domaingen_sources_enabled[@]} -eq 0 ]]; then
    domaingen_sources_enabled=("${domaingen_source_models[0]}")
  fi

  ########################################
  # 规则：如果只写了 source 没写 target
  # 解释：沿用你旧逻辑 -> 默认跑“全部 target”
  ########################################
  if (( seen_cross==1 )) && [[ ${#crossdata_targets_enabled[@]} -eq 0 ]]; then
    crossdata_targets_enabled=("${crossdata_target_models[@]}")
  fi
  if (( seen_domain==1 )) && [[ ${#domaingen_targets_enabled[@]} -eq 0 ]]; then
    domaingen_targets_enabled=("${domaingen_target_models[@]}")
  fi

  ########################################
  # 自动生成真正调度顺序
  # 说明：crossdata/domaingen 不再压成一个大任务，而是拆成
  # source/target + dataset 的独立调度单元，便于按 dataset+shot 画像显存。
  ########################################
  if (( seen_cross==1 )); then
    for s in "${crossdata_sources_enabled[@]}"; do
      add_unique run_order_exec "crossdata_source_${s}"
    done
    for tg in "${crossdata_targets_enabled[@]}"; do
      add_unique run_order_exec "crossdata_target_${tg}"
    done
  fi

  if (( seen_domain==1 )); then
    for s in "${domaingen_sources_enabled[@]}"; do
      add_unique run_order_exec "domaingen_source_${s}"
    done
    for tg in "${domaingen_targets_enabled[@]}"; do
      add_unique run_order_exec "domaingen_target_${tg}"
    done
  fi

  ########################################
  # 自动生成 base_output_order（用于展示）
  ########################################
  base_output_order=()
  local m s tg

  for m in "${base2new_models_enabled[@]}"; do
    base_output_order+=("base2new,base,${m}")
    base_output_order+=("base2new,new,${m}")
  done

  if (( seen_cross==1 )); then
    for s in "${crossdata_sources_enabled[@]}"; do
      base_output_order+=("crossdata,source,${s}")
    done
    for tg in "${crossdata_targets_enabled[@]}"; do
      base_output_order+=("crossdata,target,${tg}")
    done
  fi

  if (( seen_domain==1 )); then
    for s in "${domaingen_sources_enabled[@]}"; do
      base_output_order+=("domaingen,source,${s}")
    done
    for tg in "${domaingen_targets_enabled[@]}"; do
      base_output_order+=("domaingen,target,${tg}")
    done
  fi

  ########################################
  # 自动生成 VIS_ITEMS（最后可视化用）
  # base2new 用 v1 脚本，domaingen 用 v2 脚本
  ########################################
  VIS_ITEMS=()
  for m in "${base2new_models_enabled[@]}"; do
    VIS_ITEMS+=("base2new:${m}:vis_tsne.sh:vis_gradcam.sh")
  done
  if (( seen_cross==1 )); then
    for tg in "${crossdata_targets_enabled[@]}"; do
      VIS_ITEMS+=("crossdata:${tg}:vis_tsne.sh:vis_gradcam.sh")
    done
  fi
  if (( seen_domain==1 )); then
    for tg in "${domaingen_targets_enabled[@]}"; do
      VIS_ITEMS+=("domaingen:${tg}:vis_tsne_v2.sh:vis_gradcam_v2.sh")
    done
  fi
}

# 运行生成，并用 run_order_exec 覆盖 scheduler 真正用的 run_order
build_plan_from_tokens run_order
run_order=("${run_order_exec[@]}")
echo "[PLAN] run_order_exec=${run_order[*]}"
echo "[PLAN] crossdata_sources_enabled=${crossdata_sources_enabled[*]}"
echo "[PLAN] crossdata_targets_enabled=${crossdata_targets_enabled[*]}"
echo "[PLAN] domaingen_sources_enabled=${domaingen_sources_enabled[*]}"
echo "[PLAN] domaingen_targets_enabled=${domaingen_targets_enabled[*]}"

############################
# 1) shots与seed 计划生成
############################
(( SHOT_TRIES < 1 )) && SHOT_TRIES=1
(( SHOT_TRIES > ${#ALL_SHOTS[@]} )) && SHOT_TRIES=${#ALL_SHOTS[@]}

SHOTS_TO_RUN=("${ALL_SHOTS[@]:0:$SHOT_TRIES}")
echo "SHOT_TRIES=$SHOT_TRIES -> SHOTS_TO_RUN=${SHOTS_TO_RUN[*]}"
output_order=()
for item in "${base_output_order[@]}"; do
  for shot in "${SHOTS_TO_RUN[@]}"; do
    output_order+=("${item},shots_${shot}")
  done
done

SHOTS_TAG="$(IFS=-; echo "${SHOTS_TO_RUN[*]}")"
OUTPUT_LOG_FILE="${RUNLOG_DIR}/log_stat_${TRAINER}_${CFG}_shot${SHOTS_TAG}_seedstart${START_RUN}_end${END_RUN}.txt"
VIS_LOG_FILE="${RUNLOG_DIR}/log_vis_${TRAINER}_${CFG}_shot${SHOTS_TAG}_seedstart${START_RUN}_end${END_RUN}.txt"
RESULT_CSV_FILE="${OUTPUT_BASE}/0-results_${TRAINER}_${CFG}_shot${SHOTS_TAG}_seedstart${START_RUN}_end${END_RUN}.csv"
AVERAGE_CSV_FILE="${OUTPUT_BASE}/0-averages_${TRAINER}_${CFG}_shot${SHOTS_TAG}_seedstart${START_RUN}_end${END_RUN}.csv"
LOCK_FILE="${RUNLOG_DIR}/.lock_${TRAINER}_${CFG}_shot${SHOTS_TAG}_seedstart${START_RUN}_end${END_RUN}.lock"
SCHED_LOG="${RUNLOG_DIR}/scheduler.log"
MEM_PROFILE_FILE="${RUNLOG_DIR}/mem_profile_${TRAINER}_${CFG}.tsv"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_csv_for_shot() { # $1=shot
  echo "${OUTPUT_BASE}/1-results_${TRAINER}_${CFG}_shot${1}_seedstart${START_RUN}_end${END_RUN}.csv"
}
average_csv_for_shot() { # $1=shot
  echo "${OUTPUT_BASE}/1-averages_${TRAINER}_${CFG}_shot${1}_seedstart${START_RUN}_end${END_RUN}.csv"
}
############################
# 2) 全局文件锁
############################
command -v flock >/dev/null 2>&1 || { echo "Error: flock not found"; exit 1; }

LOCK_OWNER_BASHPID=0
LOCK_FD=-1

init_lock_fd() {
    local pid="${BASHPID:-$$}"
    if [[ "$LOCK_OWNER_BASHPID" -ne "$pid" || "$LOCK_FD" -lt 0 ]]; then
        exec {LOCK_FD}>"$LOCK_FILE"
        LOCK_OWNER_BASHPID="$pid"
    fi
}
lock_begin() { init_lock_fd; flock -x "$LOCK_FD"; }
lock_end()   { flock -u "$LOCK_FD"; }
lock_begin_wait() {
    local timeout="${1:-2}"
    init_lock_fd
    flock -w "$timeout" -x "$LOCK_FD"
}

# 清空输出文件
lock_begin
: > "$OUTPUT_LOG_FILE"
: > "$VIS_LOG_FILE"
if [[ "${SPLIT_CSV_BY_SHOT}" == "0" ]]; then
  : > "$RESULT_CSV_FILE"
  : > "$AVERAGE_CSV_FILE"
elif [[ "${SPLIT_CSV_BY_SHOT}" == "1" ]]; then
  for _s in "${SHOTS_TO_RUN[@]}"; do
    : > "$(result_csv_for_shot "$_s")"
    : > "$(average_csv_for_shot "$_s")"
  done
else # 2
  : > "$RESULT_CSV_FILE"
  : > "$AVERAGE_CSV_FILE"
  for _s in "${SHOTS_TO_RUN[@]}"; do
    : > "$(result_csv_for_shot "$_s")"
    : > "$(average_csv_for_shot "$_s")"
  done
fi
: > "$LOCK_FILE"
: > "$SCHED_LOG"
if [[ ! -f "$MEM_PROFILE_FILE" ]]; then
  printf "key\tpeak_mb\test_mb\tts\n" > "$MEM_PROFILE_FILE"
fi
lock_end

############################
# 3) monitor：写进度行（带超时锁，避免卡死）
############################
update_output_line() {
    local epoch_info="$1"
    local new_line="${key}${SEP}${epoch_info}"
    local tmp_dir tmp_file

    # 拿不到锁就跳过（不阻塞、不拖死外层）
    if ! lock_begin_wait 2; then
        return 0
    fi

    tmp_dir="$(dirname "$OUTPUT_LOG_FILE")"
    tmp_file="$(mktemp -p "$tmp_dir" ".$(basename "$OUTPUT_LOG_FILE").tmp.XXXXXX")" || { lock_end; return 0; }

    # 任何失败都要解锁并清理 tmp
    awk -v k="$key" -v sep="$SEP" -v nl="$new_line" '
        BEGIN{found=0}
        index($0, k sep)==1 {
            if (!found) { print nl; found=1 }
            next
        }
        { print }
        END{ if(!found) print nl }
    ' "$OUTPUT_LOG_FILE" > "$tmp_file" || { rm -f "$tmp_file"; lock_end; return 0; }

    mv -f "$tmp_file" "$OUTPUT_LOG_FILE" || { rm -f "$tmp_file"; lock_end; return 0; }
    lock_end
    return 0
}
monitor_logfile() {
    local task_type=$1 phase=$2 model=$3 run=$4 start_run=$5 end_run=$6 logfile=$7 epoch=$8 sid=$9
    export SHOTS

    local total_runs=$((end_run - start_run + 1))
    local current_index=$((run - start_run + 1))
    local seed_progress="seed${run}[${current_index}/${total_runs}]"

    local identifier="${task_type}${SEP}${phase}${SEP}${model}${SEP}shots_${SHOTS}"
    local key="${identifier}${SEP}${seed_progress}"

    # 初始化行：训练用 0/0，测试用 1/1
    local init_epoch="epoch [0/0]"
    [[ "$epoch" == "0" ]] && init_epoch="epoch [1/1]"
    local initial_line="${key}${SEP}${init_epoch}"

    
    # 初始化一行（拿不到锁就算了）
    if lock_begin_wait 2; then
        if ! awk -v k="$key" -v sep="$SEP" 'index($0, k sep)==1 {found=1} END{exit(found?0:1)}' \
            "$OUTPUT_LOG_FILE" 2>/dev/null; then
            printf "%s\n" "$initial_line" >> "$OUTPUT_LOG_FILE"
        fi
        lock_end
    fi

    # 从 log 中提 epoch：兼容 Epoch/epoch 大小写，并统一写成 epoch
    get_latest_epoch() {
        ( tail -n 40 "$1" 2>/dev/null || true ) \
            | grep -Eio 'epoch \[[0-9]+/[0-9]+\]' \
            | tail -n 1 \
            | sed -E 's/^[Ee]poch/epoch/' || true
    }

    # 只在 epoch 变化时更新
    local last_epoch_info=""
    while session_alive "$sid"; do
        local latest_epoch
        latest_epoch="$(get_latest_epoch "$logfile")"
        if [[ -n "$latest_epoch" ]]; then
            local epoch_info
            if [[ "$epoch" == "0" ]]; then
                epoch_info="epoch [1/1]"
            else
                epoch_info="$latest_epoch"
            fi
            if [[ "$epoch_info" != "$last_epoch_info" ]]; then
                update_output_line "$epoch_info"
                last_epoch_info="$epoch_info"
            fi
        fi
        sleep 1
    done

    # 补一次最终 epoch
    if [[ "$epoch" != "0" ]]; then
        local latest_epoch
        latest_epoch="$(get_latest_epoch "$logfile")"
        if [[ -n "$latest_epoch" && "$latest_epoch" != "$last_epoch_info" ]]; then
            update_output_line "$latest_epoch"
        fi
    fi
    return 0
}

############################
# 4) 结果提取
############################
log_to_csv() {
    local total_type=$1 type=$2 model=$3 run=$4 log_file=$5
    export SHOTS

    # 支持多候选
    local -a cands=()
    IFS='|' read -r -a cands <<< "${log_file//$'\r'/}"
    local f found=""
    for f in "${cands[@]}"; do
        f="$(realpath -m "$f")"
        if [[ -f "$f" ]]; then found="$f"; break; fi
    done

    if [[ -z "$found" ]]; then
        echo "Warn: Log file not found (tried): ${cands[*]}" >&2
        return 1
    fi
    log_file="$found"

    local last_lines accuracy kappa macro_f1 time_sec time_candidates
    last_lines="$(tail -n 20 "$log_file" 2>/dev/null || true)"

    accuracy=$(echo "$last_lines" | grep -oP 'accuracy: \K[0-9]+\.[0-9]+' | tail -n 1)
    kappa=$(echo "$last_lines" | grep -oP 'kappa: \K[0-9]+\.[0-9]+' | tail -n 1)
    macro_f1=$(echo "$last_lines" | grep -oP 'macro_f1: \K[0-9]+\.[0-9]+' | tail -n 1)

    time_candidates=$(echo "$last_lines" | grep -oP '\* time: .*?\(\K[0-9]+(\.[0-9]+)?(?=s\))')
    [[ -z "$time_candidates" ]] && time_candidates=$(echo "$last_lines" | grep -oP '\* time:\s*\K[0-9]+(\.[0-9]+)?(?=s\b)')
    time_sec=$(echo "$time_candidates" | tail -n 1)

    if [[ -z "$accuracy" || -z "$kappa" || -z "$macro_f1" || -z "$time_sec" ]]; then
        echo "Warn: Missing metrics in log: $log_file" >&2
        return 1
    fi

    local line="$total_type,$type,$model,$SHOTS,$run,$accuracy,$kappa,$macro_f1,$time_sec"
    local out_all="$RESULT_CSV_FILE"
    local out_shot
    out_shot="$(result_csv_for_shot "$SHOTS")"

    lock_begin
    case "${SPLIT_CSV_BY_SHOT}" in
    0) echo "$line" >> "$out_all" ;;
    1) echo "$line" >> "$out_shot" ;;
    2) echo "$line" >> "$out_all"; echo "$line" >> "$out_shot" ;;
    *) echo "$line" >> "$out_all" ;;  # 兜底按 0
    esac
    lock_end
}

############################
# 5) 运行 + monitor
############################
# 判断某个 session 是否仍然存在任何进程
session_alive() {
    local sid="$1"
    # ps -s <sid> 会列出属于该 session 的进程；为空表示 session 已无进程
    ps -s "$sid" -o pid= 2>/dev/null | grep -q '[0-9]'
}

# 等待 session 结束（避免太高频，默认 3 秒一查）
wait_session_done() {
    local sid="$1"
    local interval="${2:-3}"
    local timeout="${3:-86400}"  # 24h
    local waited=0

    while session_alive "$sid"; do
        sleep "$interval"
        waited=$((waited + interval))
        if (( waited >= timeout )); then
            return 124
        fi
    done
    return 0
}
# 提取最后一次 epoch [cur/tot]
extract_last_epoch_pair() { # $1=logfile  -> 输出 "cur tot" 或空
  local log="$1"
  local last
  last="$( (tail -n 200 "$log" 2>/dev/null || true) \
          | grep -Eio 'epoch \[[0-9]+/[0-9]+\]' \
          | tail -n 1 \
          | sed -E 's/^[Ee]poch \[([0-9]+)\/([0-9]+)\]/\1 \2/' )"
  echo "$last"
}

validate_train_log_done() { # $1=logfile
  local log="$1"
  local pair cur tot
  pair="$(extract_last_epoch_pair "$log")"
  [[ -n "$pair" ]] || return 1
  cur="${pair%% *}"; tot="${pair##* }"
  [[ "$cur" =~ ^[0-9]+$ && "$tot" =~ ^[0-9]+$ ]] || return 1
  (( tot > 0 )) || return 1
  (( cur >= tot )) || return 1
  return 0
}

validate_test_log_has_result() { # $1=logfile
  local log="$1"
  local last
  last="$(tail -n 200 "$log" 2>/dev/null || true)"

  echo "$last" | grep -q '=> result' || return 1
  echo "$last" | grep -qE 'accuracy:[[:space:]]*[0-9]+(\.[0-9]+)?%' || return 1
  echo "$last" | grep -qE 'kappa:[[:space:]]*[0-9]+(\.[0-9]+)?%?' || return 1
  echo "$last" | grep -qE 'macro_f1:[[:space:]]*[0-9]+(\.[0-9]+)?%' || return 1
  return 0
}

run_and_monitor() {
    local task_type="$1" phase="$2" model="$3" run="$4" logfile="$5" epoch="$6"
    shift 6

    mkdir -p "$(dirname "$logfile")"
    : > "$logfile"

    # 用 setsid 启动：把后续整个树放进新 session
    # 关键点：不要 wait 壳脚本，而是等 session 内所有进程结束
    setsid "$@" >>"$logfile" 2>&1 &
    local sid=$!  # setsid exec 后的 session leader pid == session id
    # 可选：把 sid 写入文件，供其他任务用 pid 等待
    if [[ -n "${EXPORT_SID_FILE:-}" ]]; then
        mkdir -p "$(dirname "$EXPORT_SID_FILE")"
        echo "$sid" > "$EXPORT_SID_FILE" 2>/dev/null || true
    fi

    monitor_logfile "$task_type" "$phase" "$model" "$run" "$START_RUN" "$END_RUN" "$logfile" "$epoch" "$sid" &
    local mon_pid=$!

    # 等 session 结束（默认 3 秒探测一次；你可以调大到 5/10 更省）
    local rc=0
    if ! wait_session_done "$sid" "${WAIT_POLL_SEC:-3}" "${STATUS_TIMEOUT_SEC:-86400}"; then
        rc=124
        printf '%s [ERROR] session timeout sid=%s name=%s_%s_%s_shots%s_seed%s\n' \
          "$(date '+%F %T')" "$sid" "$task_type" "$phase" "$model" "${SHOTS:-?}" "$run" >> "$SCHED_LOG"
    else
        rc=0
        # ======= 关键：用日志判定“真正成功” =======
        if [[ "$epoch" != "0" ]]; then
            # 训练：必须跑满 epoch
            if ! validate_train_log_done "$logfile"; then
                rc=2
                printf '%s [ERROR] train not finished (epoch not reached): %s\n' \
                  "$(date '+%F %T')" "$logfile" >> "$SCHED_LOG"
            fi
        else
            # 测试：必须出现 result + metrics
            if ! validate_test_log_has_result "$logfile"; then
                rc=3
                printf '%s [ERROR] test missing result/metrics: %s\n' \
                  "$(date '+%F %T')" "$logfile" >> "$SCHED_LOG"
            fi
        fi
    fi

    # 收尾 monitor
    kill "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    # 可选：写训练完成标记（done/fail）
    if [[ -n "${EXPORT_STATUS_PREFIX:-}" ]]; then
        mkdir -p "$(dirname "$EXPORT_STATUS_PREFIX")"
        if [[ "$rc" -eq 0 ]]; then
            : > "${EXPORT_STATUS_PREFIX}.done"
            rm -f "${EXPORT_STATUS_PREFIX}.fail" 2>/dev/null || true
        else
            : > "${EXPORT_STATUS_PREFIX}.fail"
            rm -f "${EXPORT_STATUS_PREFIX}.done" 2>/dev/null || true
        fi
    fi

    return "$rc"
}

############################
# 6) 任务函数
############################
b2n_sigdir_train() { # $1=model $2=shot $3=seed
  echo "${RUNLOG_DIR}/signals/base2new/train_base/${1}/shots_${2}/seed${3}"
}
b2n_sidfile_train() { # model shot seed
  echo "$(b2n_sigdir_train "$1" "$2" "$3")/train.sid"
}
b2n_donefile_train() { # model shot seed
  echo "$(b2n_sigdir_train "$1" "$2" "$3")/train.done"
}
b2n_failfile_train() { # model shot seed
  echo "$(b2n_sigdir_train "$1" "$2" "$3")/train.fail"
}

# pid 等待（kill -0 轮询）+ done/fail 判定
wait_b2n_train_done() { # $1=model $2=shot $3=seed
  local model="$1" shot="$2" seed="$3"
  local sigdir sidf donef failf
  sigdir="$(b2n_sigdir_train "$model" "$shot" "$seed")"
  sidf="$(b2n_sidfile_train "$model" "$shot" "$seed")"
  donef="$(b2n_donefile_train "$model" "$shot" "$seed")"
  failf="$(b2n_failfile_train "$model" "$shot" "$seed")"

  local timeout="${B2N_WAIT_TIMEOUT_SEC:-86400}"
  local interval="${B2N_WAIT_POLL_SEC:-2}"
  local t0 now pid

  t0="$(date +%s)"
  while true; do
    [[ -f "$donef" ]] && return 0
    [[ -f "$failf" ]] && return 1

    if [[ -f "$sidf" ]]; then
      pid="$(cat "$sidf" 2>/dev/null || echo "")"
      if [[ "$pid" =~ ^[0-9]+$ ]]; then
        if kill -0 "$pid" 2>/dev/null; then
          sleep "$interval"
        else
          # pid 已结束，但 done/fail 可能稍后写入，短等一下再判
          sleep 0.5
        fi
      else
        sleep "$interval"
      fi
    else
      # 训练还没启动/还没写 sidfile
      sleep "$interval"
    fi

    now="$(date +%s)"
    if (( now - t0 >= timeout )); then
      printf '%s [ERROR] wait_b2n_train_done timeout model=%s shot=%s seed=%s\n' \
        "$(date '+%F %T')" "$model" "$shot" "$seed" >> "$SCHED_LOG"
      return 124
    fi
  done
}

ensure_symlink_modeldir() { # $1=src_abs $2=dst_abs
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"

  # dst 已存在但不是 symlink：备份再替换，避免复用时拿到旧目录
  if [[ -e "$dst" && ! -L "$dst" ]]; then
    mv "$dst" "${dst}.bak.$(date +%s)" 2>/dev/null || rm -rf "$dst"
  fi

  # 创建/刷新软链接
  ln -sfn "$src" "$dst"
}

base2new_one_model() {
    local model="$1"
    local seed="$2"
    export SHOTS

    local log_train="${RUNLOG_DIR}/base2new/train_base/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
    echo "$log_train" >> "$SCHED_LOG"
    # train 信号文件（用于 crossdata 复用等待）
    local sigdir sidfile status_prefix
    sigdir="$(b2n_sigdir_train "$model" "$SHOTS" "$seed")"
    mkdir -p "$sigdir"
    sidfile="$(b2n_sidfile_train "$model" "$SHOTS" "$seed")"
    status_prefix="${sigdir}/train"
    rm -f "${status_prefix}.done" "${status_prefix}.fail" "$sidfile" 2>/dev/null || true

    # 训练：拿到真实 rc
    EXPORT_SID_FILE="$sidfile" EXPORT_STATUS_PREFIX="$status_prefix" \
    run_and_monitor "base2new" "base" "$model" "$seed" "$log_train" 1 \
        bash base2new_train.sh "$model" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 0
    local train_rc=$?

    if (( train_rc != 0 )); then
        printf '%s [ERROR] base2new train failed: model=%s shots=%s seed=%s rc=%s (skip test)\n' \
            "$(date '+%F %T')" "$model" "$SHOTS" "$seed" "$train_rc" >> "$SCHED_LOG"
        return "$train_rc"
    fi

    # 训练阶段通常没有 accuracy/kappa/macro_f1/time，避免无意义 warn
    # 如果你确实要写 train 指标，再把下一行放开（但要确保训练日志里真的有这些字段）
    log_to_csv "base2new" "base" "$model" "$seed" "$log_train" || true

    local log_test="${RUNLOG_DIR}/base2new/test_new/${model}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}/log.txt"
    echo "$log_test" >> "$SCHED_LOG"

    # 测试：同样检查 rc（可选）
    run_and_monitor "base2new" "new" "$model" "$seed" "$log_test" 0 \
        bash base2new_test.sh "$model" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 0
    local test_rc=$?

    if (( test_rc != 0 )); then
        printf '%s [ERROR] base2new test failed: model=%s shots=%s seed=%s rc=%s\n' \
            "$(date '+%F %T')" "$model" "$SHOTS" "$seed" "$test_rc" >> "$SCHED_LOG"
        return "$test_rc"
    fi

    log_to_csv "base2new" "new" "$model" "$seed" "$log_test" || true
    return 0
}

base2new_patternnet() { base2new_one_model "patternnet" "$1"; }
base2new_mlrsnet()    { base2new_one_model "mlrsnet"    "$1"; }
base2new_resisc45()   { base2new_one_model "resisc45"   "$1"; }
base2new_rsicd()      { base2new_one_model "rsicd"      "$1"; }
crossdata() {
  local seed="$1"
  export SHOTS
  local HAS_B2N_PATTERNNET=0
  local any_fail=0
  if in_list "patternnet" "${base2new_models_enabled[@]}"; then
    HAS_B2N_PATTERNNET=1
  fi


  [[ ${#crossdata_sources_enabled[@]} -eq 0 ]] && { echo "[crossdata][SKIP] not enabled"; return 0; }

  for source in "${crossdata_sources_enabled[@]}"; do
    if [[ "$source" == "patternnet" && "$HAS_B2N_PATTERNNET" == "1" ]]; then
      # 1) 等 base2new_patternnet 对应 shot+seed 训练完成（pid 轮询 + done/fail）
      if ! wait_b2n_train_done "patternnet" "$SHOTS" "$seed"; then
        printf '%s [ERROR] reuse failed: base2new_patternnet not ready shot=%s seed=%s (skip all targets)\n' \
          "$(date '+%F %T')" "$SHOTS" "$seed" >> "$SCHED_LOG"
        continue
      fi

      # 2) 建软链接：让 crossdata_test 默认的 model-dir 直接指向 base2new 输出
      local src dst
      src="${ROOT_DIR}/outputs/base2new/train_base/patternnet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}"
      dst="${ROOT_DIR}/outputs/crosstransfer/patternnet/${TRAINER}/${CFG}_shots${SHOTS}/seed${seed}"
      dst_log="${dst}/log.txt"

      if [[ ! -d "$src" ]]; then
        printf '%s [ERROR] reuse failed: src not found: %s (skip all targets)\n' \
          "$(date '+%F %T')" "$src" >> "$SCHED_LOG"
        continue
      fi

      ensure_symlink_modeldir "$src" "$dst"
      log_to_csv "crossdata" "source" "$source" "$seed" "$dst_log" || true
      # 复用模式：跳过 crossdata source 训练，直接进入 targets
      :
    else
      local log_train="${RUNLOG_DIR}/crosstransfer/${source}/${TRAINER}/${CFG}_shots${SHOTS}/seed${seed}/log.txt"

      run_and_monitor "crossdata" "source" "$source" "$seed" "$log_train" 1 \
        bash crossdata_train.sh "$source" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 1
      local train_rc=$?

      if (( train_rc != 0 )); then
        printf '%s [ERROR] crossdata train failed: source=%s shots=%s seed=%s rc=%s (skip all targets)\n' \
          "$(date '+%F %T')" "$source" "$SHOTS" "$seed" "$train_rc" >> "$SCHED_LOG"
        continue
      fi

      log_to_csv "crossdata" "source" "$source" "$seed" "$log_train" || true

    fi

    for target in "${crossdata_targets_enabled[@]}"; do
      local log_test="${RUNLOG_DIR}/crosstransfer/tests/${TRAINER}/${CFG}_shots${SHOTS}/${target}/seed${seed}/log.txt"

      run_and_monitor "crossdata" "target" "$target" "$seed" "$log_test" 0 \
        bash crossdata_test.sh "$target" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 1
      local test_rc=$?

      if (( test_rc != 0 )); then
        printf '%s [ERROR] crossdata test failed: target=%s shots=%s seed=%s rc=%s\n' \
          "$(date '+%F %T')" "$target" "$SHOTS" "$seed" "$test_rc" >> "$SCHED_LOG"
        any_fail=1
        continue
      fi

      log_to_csv "crossdata" "target" "$target" "$seed" "$log_test" || true
    done
  done

  return $any_fail
}

domaingen() {
  local seed="$1"
  export SHOTS
  local any_fail=0

  [[ ${#domaingen_sources_enabled[@]} -eq 0 ]] && { echo "[domaingen][SKIP] not enabled"; return 0; }

  for source in "${domaingen_sources_enabled[@]}"; do
    local log_train="${RUNLOG_DIR}/domain_generalization/${source}/${TRAINER}/${CFG}_shots${SHOTS}/seed${seed}/log.txt"

    run_and_monitor "domaingen" "source" "$source" "$seed" "$log_train" 1 \
      bash domaingen_train.sh "$source" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 1
    local train_rc=$?

    if (( train_rc != 0 )); then
      printf '%s [ERROR] domaingen train failed: source=%s shots=%s seed=%s rc=%s (skip all targets)\n' \
        "$(date '+%F %T')" "$source" "$SHOTS" "$seed" "$train_rc" >> "$SCHED_LOG"
      continue
    fi

    log_to_csv "domaingen" "source" "$source" "$seed" "$log_train" || true

    for target in "${domaingen_targets_enabled[@]}"; do
      local log_test="${RUNLOG_DIR}/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/${target}/seed${seed}/log.txt"

      run_and_monitor "domaingen" "target" "$target" "$seed" "$log_test" 0 \
        bash domaingen_test.sh "$target" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 1
      local test_rc=$?

      if (( test_rc != 0 )); then
        printf '%s [ERROR] domaingen test failed: target=%s shots=%s seed=%s rc=%s\n' \
          "$(date '+%F %T')" "$target" "$SHOTS" "$seed" "$test_rc" >> "$SCHED_LOG"
        any_fail=1
        continue
      fi

      log_to_csv "domaingen" "target" "$target" "$seed" "$log_test" || true
    done
  done
  return $any_fail
}


############################
# 6.5) 拆分后的 crossdata / domaingen 任务
############################
cd_sigdir_source() { # $1=source $2=shot $3=seed
  echo "${RUNLOG_DIR}/signals/crossdata/source/${1}/shots_${2}/seed${3}"
}
cd_sidfile_source() { echo "$(cd_sigdir_source "$1" "$2" "$3")/train.sid"; }
cd_donefile_source() { echo "$(cd_sigdir_source "$1" "$2" "$3")/train.done"; }
cd_failfile_source() { echo "$(cd_sigdir_source "$1" "$2" "$3")/train.fail"; }

dg_sigdir_source() { # $1=source $2=shot $3=seed
  echo "${RUNLOG_DIR}/signals/domaingen/source/${1}/shots_${2}/seed${3}"
}
dg_sidfile_source() { echo "$(dg_sigdir_source "$1" "$2" "$3")/train.sid"; }
dg_donefile_source() { echo "$(dg_sigdir_source "$1" "$2" "$3")/train.done"; }
dg_failfile_source() { echo "$(dg_sigdir_source "$1" "$2" "$3")/train.fail"; }

wait_signal_done() { # $1=done_file $2=fail_file $3=sid_file $4=desc
  local donef="$1" failf="$2" sidf="$3" desc="$4"
  local timeout="${SOURCE_WAIT_TIMEOUT_SEC:-86400}"
  local interval="${SOURCE_WAIT_POLL_SEC:-2}"
  local t0 now pid
  t0="$(date +%s)"
  while true; do
    [[ -f "$donef" ]] && return 0
    [[ -f "$failf" ]] && return 1

    if [[ -f "$sidf" ]]; then
      pid="$(cat "$sidf" 2>/dev/null || echo "")"
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        sleep "$interval"
      else
        sleep 0.5
      fi
    else
      sleep "$interval"
    fi

    now="$(date +%s)"
    if (( now - t0 >= timeout )); then
      printf '%s [ERROR] wait source timeout: %s\n' "$(date '+%F %T')" "$desc" >> "$SCHED_LOG"
      return 124
    fi
  done
}

wait_crossdata_source_done() { # $1=source $2=shot $3=seed
  wait_signal_done \
    "$(cd_donefile_source "$1" "$2" "$3")" \
    "$(cd_failfile_source "$1" "$2" "$3")" \
    "$(cd_sidfile_source "$1" "$2" "$3")" \
    "crossdata source=$1 shot=$2 seed=$3"
}

wait_domaingen_source_done() { # $1=source $2=shot $3=seed
  wait_signal_done \
    "$(dg_donefile_source "$1" "$2" "$3")" \
    "$(dg_failfile_source "$1" "$2" "$3")" \
    "$(dg_sidfile_source "$1" "$2" "$3")" \
    "domaingen source=$1 shot=$2 seed=$3"
}

crossdata_source_one() {
  local source="$1"
  local seed="$2"
  export SHOTS

  local sigdir sidfile status_prefix
  sigdir="$(cd_sigdir_source "$source" "$SHOTS" "$seed")"
  mkdir -p "$sigdir"
  sidfile="$(cd_sidfile_source "$source" "$SHOTS" "$seed")"
  status_prefix="${sigdir}/train"
  rm -f "${status_prefix}.done" "${status_prefix}.fail" "$sidfile" 2>/dev/null || true

  local HAS_B2N_PATTERNNET=0
  if in_list "patternnet" "${base2new_models_enabled[@]}"; then
    HAS_B2N_PATTERNNET=1
  fi

  if [[ "$source" == "patternnet" && "$HAS_B2N_PATTERNNET" == "1" ]]; then
    if ! wait_b2n_train_done "patternnet" "$SHOTS" "$seed"; then
      printf '%s [ERROR] crossdata reuse failed: base2new_patternnet not ready shot=%s seed=%s\n' \
        "$(date '+%F %T')" "$SHOTS" "$seed" >> "$SCHED_LOG"
      : > "${status_prefix}.fail"
      return 1
    fi

    local src dst dst_log
    src="${ROOT_DIR}/outputs/base2new/train_base/patternnet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}"
    dst="${ROOT_DIR}/outputs/crosstransfer/patternnet/${TRAINER}/${CFG}_shots${SHOTS}/seed${seed}"
    dst_log="${dst}/log.txt"

    if [[ ! -d "$src" ]]; then
      printf '%s [ERROR] crossdata reuse failed: src not found: %s\n' \
        "$(date '+%F %T')" "$src" >> "$SCHED_LOG"
      : > "${status_prefix}.fail"
      return 1
    fi

    ensure_symlink_modeldir "$src" "$dst"
    log_to_csv "crossdata" "source" "$source" "$seed" "$dst_log" || true
    echo "$$" > "$sidfile" 2>/dev/null || true
    : > "${status_prefix}.done"
    return 0
  fi

  local log_train="${RUNLOG_DIR}/crosstransfer/${source}/${TRAINER}/${CFG}_shots${SHOTS}/seed${seed}/log.txt"
  EXPORT_SID_FILE="$sidfile" EXPORT_STATUS_PREFIX="$status_prefix" \
  run_and_monitor "crossdata" "source" "$source" "$seed" "$log_train" 1 \
    bash crossdata_train.sh "$source" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 1
  local train_rc=$?

  if (( train_rc != 0 )); then
    printf '%s [ERROR] crossdata source train failed: source=%s shots=%s seed=%s rc=%s\n' \
      "$(date '+%F %T')" "$source" "$SHOTS" "$seed" "$train_rc" >> "$SCHED_LOG"
    return "$train_rc"
  fi

  log_to_csv "crossdata" "source" "$source" "$seed" "$log_train" || true
  return 0
}

crossdata_target_one() {
  local target="$1"
  local seed="$2"
  export SHOTS
  local any_fail=0 source

  [[ ${#crossdata_sources_enabled[@]} -eq 0 ]] && { echo "[crossdata-target][SKIP] no source enabled"; return 0; }

  for source in "${crossdata_sources_enabled[@]}"; do
    if ! wait_crossdata_source_done "$source" "$SHOTS" "$seed"; then
      printf '%s [ERROR] crossdata target skipped: source=%s not ready target=%s shots=%s seed=%s\n' \
        "$(date '+%F %T')" "$source" "$target" "$SHOTS" "$seed" >> "$SCHED_LOG"
      any_fail=1
      continue
    fi

    local log_test="${RUNLOG_DIR}/crosstransfer/tests/${TRAINER}/${CFG}_shots${SHOTS}/${target}/seed${seed}/log.txt"
    run_and_monitor "crossdata" "target" "$target" "$seed" "$log_test" 0 \
      bash crossdata_test.sh "$target" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 1
    local test_rc=$?

    if (( test_rc != 0 )); then
      printf '%s [ERROR] crossdata target test failed: source=%s target=%s shots=%s seed=%s rc=%s\n' \
        "$(date '+%F %T')" "$source" "$target" "$SHOTS" "$seed" "$test_rc" >> "$SCHED_LOG"
      any_fail=1
      continue
    fi

    log_to_csv "crossdata" "target" "$target" "$seed" "$log_test" || true
  done
  return "$any_fail"
}

domaingen_source_one() {
  local source="$1"
  local seed="$2"
  export SHOTS

  local sigdir sidfile status_prefix
  sigdir="$(dg_sigdir_source "$source" "$SHOTS" "$seed")"
  mkdir -p "$sigdir"
  sidfile="$(dg_sidfile_source "$source" "$SHOTS" "$seed")"
  status_prefix="${sigdir}/train"
  rm -f "${status_prefix}.done" "${status_prefix}.fail" "$sidfile" 2>/dev/null || true

  local log_train="${RUNLOG_DIR}/domain_generalization/${source}/${TRAINER}/${CFG}_shots${SHOTS}/seed${seed}/log.txt"
  EXPORT_SID_FILE="$sidfile" EXPORT_STATUS_PREFIX="$status_prefix" \
  run_and_monitor "domaingen" "source" "$source" "$seed" "$log_train" 1 \
    bash domaingen_train.sh "$source" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 1
  local train_rc=$?

  if (( train_rc != 0 )); then
    printf '%s [ERROR] domaingen source train failed: source=%s shots=%s seed=%s rc=%s\n' \
      "$(date '+%F %T')" "$source" "$SHOTS" "$seed" "$train_rc" >> "$SCHED_LOG"
    return "$train_rc"
  fi

  log_to_csv "domaingen" "source" "$source" "$seed" "$log_train" || true
  return 0
}

domaingen_target_one() {
  local target="$1"
  local seed="$2"
  export SHOTS
  local any_fail=0 source

  [[ ${#domaingen_sources_enabled[@]} -eq 0 ]] && { echo "[domaingen-target][SKIP] no source enabled"; return 0; }

  for source in "${domaingen_sources_enabled[@]}"; do
    if ! wait_domaingen_source_done "$source" "$SHOTS" "$seed"; then
      printf '%s [ERROR] domaingen target skipped: source=%s not ready target=%s shots=%s seed=%s\n' \
        "$(date '+%F %T')" "$source" "$target" "$SHOTS" "$seed" >> "$SCHED_LOG"
      any_fail=1
      continue
    fi

    local log_test="${RUNLOG_DIR}/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/${target}/seed${seed}/log.txt"
    run_and_monitor "domaingen" "target" "$target" "$seed" "$log_test" 0 \
      bash domaingen_test.sh "$target" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" 1
    local test_rc=$?

    if (( test_rc != 0 )); then
      printf '%s [ERROR] domaingen target test failed: source=%s target=%s shots=%s seed=%s rc=%s\n' \
        "$(date '+%F %T')" "$source" "$target" "$SHOTS" "$seed" "$test_rc" >> "$SCHED_LOG"
      any_fail=1
      continue
    fi

    log_to_csv "domaingen" "target" "$target" "$seed" "$log_test" || true
  done
  return "$any_fail"
}

run_task() { # $1=current_task $2=seed
  local current_task="$1"
  local seed="$2"

  case "$current_task" in
    base2new_*)
      base2new_one_model "${current_task#base2new_}" "$seed"
      ;;
    crossdata_source_*)
      crossdata_source_one "${current_task#crossdata_source_}" "$seed"
      ;;
    crossdata_target_*)
      crossdata_target_one "${current_task#crossdata_target_}" "$seed"
      ;;
    domaingen_source_*)
      domaingen_source_one "${current_task#domaingen_source_}" "$seed"
      ;;
    domaingen_target_*)
      domaingen_target_one "${current_task#domaingen_target_}" "$seed"
      ;;
    crossdata)
      crossdata "$seed"  # 兼容旧任务名
      ;;
    domaingen)
      domaingen "$seed"  # 兼容旧任务名
      ;;
    *)
      "$current_task" "$seed"
      ;;
  esac
}

############################
# 7) 显存检测（低频缓存）
############################
check_free_memory_raw() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
# 画像 key：一个“任务批次”的身份
# 新逻辑：优先按 task/phase/dataset + shot 区分显存画像。
# 例如：domaingen_target_rsicdv2__shot16 与 domaingen_target_resisc45v2__shot16 分开记录；
#      同一个 key 的 seed1/2/3 默认共享预算。
mem_core_for() { # $1=current_task -> 输出任务画像核心名
  local current_task="$1"
  case "$current_task" in
    base2new_*)          echo "base2new_${current_task#base2new_}" ;;
    crossdata_source_*)  echo "crossdata_source_${current_task#crossdata_source_}" ;;
    crossdata_target_*)  echo "crossdata_target_${current_task#crossdata_target_}" ;;
    domaingen_source_*)  echo "domaingen_source_${current_task#domaingen_source_}" ;;
    domaingen_target_*)  echo "domaingen_target_${current_task#domaingen_target_}" ;;
    *)                   echo "$current_task" ;;
  esac
}

mem_dataset_for() { # $1=current_task -> 输出数据集名
  local current_task="$1"
  case "$current_task" in
    base2new_*)          echo "${current_task#base2new_}" ;;
    crossdata_source_*)  echo "${current_task#crossdata_source_}" ;;
    crossdata_target_*)  echo "${current_task#crossdata_target_}" ;;
    domaingen_source_*)  echo "${current_task#domaingen_source_}" ;;
    domaingen_target_*)  echo "${current_task#domaingen_target_}" ;;
    *)                   echo "$current_task" ;;
  esac
}

mem_family_for() { # $1=current_task -> 输出任务族/阶段
  local current_task="$1"
  case "$current_task" in
    base2new_*)          echo "base2new" ;;
    crossdata_source_*)  echo "crossdata_source" ;;
    crossdata_target_*)  echo "crossdata_target" ;;
    domaingen_source_*)  echo "domaingen_source" ;;
    domaingen_target_*)  echo "domaingen_target" ;;
    *)                   echo "$current_task" ;;
  esac
}

mem_key_for() {
  local current_task="$1" shot="$2" seed="$3"
  local core dataset family
  core="$(mem_core_for "$current_task")"
  dataset="$(mem_dataset_for "$current_task")"
  family="$(mem_family_for "$current_task")"

  case "${MEM_PROFILE_DIM}" in
    global|none)             echo "global" ;;
    task)                    echo "$core" ;;
    dataset)                 echo "${family}_${dataset}" ;;
    shot)                    echo "${core}__shot${shot}" ;;              # 兼容旧 shot 粒度，但包含拆分后的 dataset
    dataset_shot|shot_dataset)
                              echo "${core}__shot${shot}" ;;
    seed)                    echo "${core}__seed${seed}" ;;
    seed_shot|seed_dataset_shot|dataset_shot_seed)
                              echo "${core}__shot${shot}__seed${seed}" ;;
    *)                       echo "${core}__shot${shot}" ;;              # 兜底：按 dataset+shot
  esac
}

profile_known_for_key() {
  local key="$1"
  [[ "${mem_est_mb[$key]:-0}" =~ ^[0-9]+$ && "${mem_est_mb[$key]:-0}" -gt 0 ]]
}

same_mem_key_running() { # $1=key
  local key="$1" p
  for p in "${running_pids[@]}"; do
    [[ "${pid_mem_key[$p]:-}" == "$key" ]] && return 0
  done
  return 1
}

reserve_for_key() {
  local key="$1"
  local v="${mem_est_mb[$key]:-0}"
  if [[ "$v" -le 0 ]]; then
    echo "$INIT_RESERVE_MB"
  else
    echo "$v"
  fi
}

# 读历史画像（同一个 key 取最后一次）
load_mem_profile() {
  [[ -f "$MEM_PROFILE_FILE" ]] || return 0
  while IFS=$'\t' read -r k peak est ts; do
    [[ -z "${k:-}" || "$k" == "key" ]] && continue
    [[ "${est:-}" =~ ^[0-9]+$ ]] && mem_est_mb["$k"]="$est"
    [[ "${peak:-}" =~ ^[0-9]+$ ]] && mem_peak_mb["$k"]="$peak"
  done < "$MEM_PROFILE_FILE"
}
round_up_mb() {
  local x="$1"
  local g="$RESERVE_ROUND_MB"
  (( g <= 0 )) && { echo "$x"; return 0; }
  echo $(( (x + g - 1) / g * g ))
}

update_mem_profile() {
    local key="$1" peak="$2"
    [[ "$peak" =~ ^[0-9]+$ ]] || return 0
    (( peak <= 0 )) && return 0

    local est=$(( peak + PEAK_MARGIN_MB ))
    est="$(round_up_mb "$est")"

    # 画像出来后，用 MIN_PROFILE_RESERVE_MB 做下限（而不是 INIT_RESERVE_MB）
    (( est < MIN_PROFILE_RESERVE_MB )) && est="$MIN_PROFILE_RESERVE_MB"


    mem_peak_mb["$key"]="$peak"
    mem_est_mb["$key"]="$est"

    # 追加记录（不覆盖，方便你回看变化）
    if [[ ! -f "$MEM_PROFILE_FILE" ]]; then
    printf "key\tpeak_mb\test_mb\tts\n" > "$MEM_PROFILE_FILE"
    fi
    printf "%s\t%s\t%s\t%s\n" "$key" "$peak" "$est" "$(date '+%F %T')" >> "$MEM_PROFILE_FILE"
}

# 判断 gpu_pid 是否属于 wrapper_pid 的子树（沿 PPID 往上爬）
is_descendant_of() {
  local gpu_pid="$1" wrapper_pid="$2"
  local p="$gpu_pid"
  while [[ "$p" -gt 1 ]]; do
    [[ "$p" -eq "$wrapper_pid" ]] && return 0
    # /proc/<pid>/stat 第4列是 ppid
    p="$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null || echo 0)"
    [[ -z "$p" || "$p" -le 0 ]] && break
  done
  return 1
}

# 单次采样：对每个 running wrapper pid 统计“本次 GPU used_memory 总和”，并更新峰值
sample_gpu_peak_once() {
  # 取 GPU 上所有 compute 进程 (pid, used_memory)
  local lines
  lines="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "$lines" ]] && return 0

  # 解析到数组（pid, mem）
  local -a gpu_pids=()
  local -a gpu_mems=()
  local pid mem
  while IFS=',' read -r pid mem; do
    pid="$(echo "${pid:-}" | tr -d ' ')"
    mem="$(echo "${mem:-}" | tr -d ' ')"
    [[ -z "$pid" || -z "$mem" ]] && continue
    [[ "$pid" =~ ^[0-9]+$ && "$mem" =~ ^[0-9]+$ ]] || continue
    gpu_pids+=("$pid")
    gpu_mems+=("$mem")
  done <<< "$lines"

  local w i sum
  for w in "${running_pids[@]}"; do
    sum=0
    for i in "${!gpu_pids[@]}"; do
      pid="${gpu_pids[$i]}"
      mem="${gpu_mems[$i]}"
      if is_descendant_of "$pid" "$w"; then
        sum=$(( sum + mem ))
      fi
    done
    # 更新该 wrapper 的峰值
    if [[ "$sum" -gt "${pid_peak_mb[$w]:-0}" ]]; then
      pid_peak_mb["$w"]="$sum"
    fi
  done
}

maybe_sample_gpu_peak() {
  local now
  now="$(date +%s)"
  # 没有 running 就不采
  [[ ${#running_pids[@]} -eq 0 ]] && return 0
  (( now - LAST_MEM_SAMPLE_TS < MEM_SAMPLE_INTERVAL )) && return 0
  LAST_MEM_SAMPLE_TS="$now"
  sample_gpu_peak_once || true
}
check_free_memory() {
  # 用法：
  #   check_free_memory          -> 允许缓存（给 UI/打印用）
  #   check_free_memory fresh    -> 强制实时查（给调度准入用）
  local mode="${1:-cached}"
  local now
  now="$(date +%s)"

  if [[ "$mode" != "fresh" ]]; then
    if (( now - GPU_MEM_CACHE_TS < GPU_MEM_QUERY_INTERVAL )) && [[ "${GPU_MEM_CACHE_VAL:-0}" -gt 0 ]]; then
      echo "$GPU_MEM_CACHE_VAL"
      return 0
    fi
  fi

  local v
  v="$(check_free_memory_raw || true)"
  if [[ -z "$v" || ! "$v" =~ ^[0-9]+$ ]]; then
    echo "${GPU_MEM_CACHE_VAL:-0}"
    return 0
  fi

  GPU_MEM_CACHE_TS="$now"
  GPU_MEM_CACHE_VAL="$v"
  echo "$v"
}
############################
# 8) 展示与调度
############################
print_task_status() {
    local running_n=${#running_pids[@]}
    local queued_n=${#tasks[@]}
    local done_n=${DONE_TASKS}
    local total_n=${TOTAL_TASKS}

    local running_names=()
    local pid
    for pid in "${running_pids[@]}"; do
        running_names+=("${task_map[$pid]:-UNKNOWN}")
    done

    echo "任务进度：进行中=${running_n}/${MAX_TASK_NUM} | 已完成=${done_n} | 总共=${total_n} | 待运行=${queued_n}"
    [[ ${#running_names[@]} -gt 0 ]] && echo "运行任务：${running_names[*]}" || echo "运行任务：(无)"
}
render_progress_compact() {
    local snapshot="$1"

    local shots_csv order_list
    shots_csv="$(IFS=,; echo "${SHOTS_TO_RUN[*]}")"
    order_list="$(IFS=';'; echo "${base_output_order[*]}")"

    awk -v FS=$'\t' -v OFS=$'\t' \
        -v start="$START_RUN" -v end="$END_RUN" \
        -v shots_csv="$shots_csv" \
        -v order_list="$order_list" '
    function parse_epoch(s, cur, tot, m){
        cur=tot=0
        if (match(s, /epoch \[([0-9]+)\/([0-9]+)\]/, m)) {
            cur=m[1]+0; tot=m[2]+0
        }
        return cur SUBSEP tot
    }
    function is_done(cur, tot){ return (tot>0 && cur>=tot) }

    BEGIN{
        nshots=split(shots_csv, shots_arr, ",")
        norder=split(order_list, order_arr, ";")
        nseeds=0
        for (i=start; i<=end; i++) seeds[++nseeds]=i
    }

    {
        task=$1; phase=$2; model=$3; shot_str=$4; seedp=$5; epoch=$6
        if (task==""||phase==""||model==""||shot_str==""||seedp==""||epoch=="") next

        shot=shot_str
        sub(/^shots_/, "", shot)
        shot+=0

        if (!match(seedp, /^seed([0-9]+)\[/, m)) next
        seed=m[1]+0

        key=task SUBSEP phase SUBSEP model
        sid=key SUBSEP shot
        idx=sid SUBSEP seed

        ep=parse_epoch(epoch); split(ep, epp, SUBSEP)
        cur=epp[1]+0; tot=epp[2]+0

        curv[idx]=cur
        if (tot > totv[sid]) totv[sid]=tot

        started[sid]=1
        started_key[key]=1
    }

    function shot_complete(key, shot,   sid, i, sd, idx, tot){
        sid=key SUBSEP shot
        tot=totv[sid]+0
        if (tot<=0) return 0
        for (i=1; i<=nseeds; i++){
            sd=seeds[i]
            idx=sid SUBSEP sd
            if (!(idx in curv)) return 0
            if (!is_done(curv[idx], tot)) return 0
        }
        return 1
    }

    function all_shots_done(key,   j, s){
        for (j=1; j<=nshots; j++){
            s=shots_arr[j]+0
            if (!shot_complete(key, s)) return 0
        }
        return 1
    }

    function fmt_seed_range(   i, txt, sep){
        txt=""; sep=""
        for (i=1; i<=nseeds; i++){ txt=txt sep seeds[i]; sep="-" }
        return "seed[" txt "/" nseeds "]"
    }

    function fmt_epoch_list(key, shot,   i, sd, sid, idx, txt, sep, tot){
        sid=key SUBSEP shot
        tot=totv[sid]+0
        tots=(tot>0?tot:"?")
        txt=""; sep=""
        for (i=1; i<=nseeds; i++){
            sd=seeds[i]
            idx=sid SUBSEP sd
            if (idx in curv) txt=txt sep curv[idx]
            else txt=txt sep "-"     # 缺失就 -
            sep="-"
        }
        return "epoch [" txt "/" tots "]"
    }

    END{
        for (i=1; i<=norder; i++){
            split(order_arr[i], f, ",")
            task=f[1]; phase=f[2]; model=f[3]
            key=task SUBSEP phase SUBSEP model

            if (!started_key[key]) continue

            # 规则1：如果有任何正在跑的 shot（任意 seed 未完成）→ 显示正在跑的全部 shot
            running_exist=0
            for (j=1; j<=nshots; j++){
                s=shots_arr[j]+0
                sid=key SUBSEP s
                if (!started[sid]) continue
                if (!shot_complete(key, s)) {
                    running_exist=1
                    seed_str=fmt_seed_range()
                    epoch_str=fmt_epoch_list(key, s)
                    print task, phase, model, ("shots_" s), seed_str, epoch_str
                }
            }

            if (running_exist) continue

            # 规则2：当前没有正在跑 → 显示已开始过的最大 shot（最近完成/最近开始）
            best_started=-1
            for (j=1; j<=nshots; j++){
                s=shots_arr[j]+0
                sid=key SUBSEP s
                if (started[sid] && s>best_started) best_started=s
            }
            if (best_started<0) continue

            seed_str=fmt_seed_range()
            epoch_str=fmt_epoch_list(key, best_started)
            line=task OFS phase OFS model OFS ("shots_" best_started) OFS seed_str OFS epoch_str

            # 规则3：所有 shots 完成 → 仍然显示最大 shot + DONE
            if (all_shots_done(key)) line=line OFS "DONE(all shots)"

            print line
        }
    }' "$snapshot"
}

logging() {
    local free_memory snapshot tmp_dir
    free_memory=$(check_free_memory)

    tmp_dir="$(dirname "$OUTPUT_LOG_FILE")"
    snapshot="$(mktemp -p "$tmp_dir" ".snapshot.$(basename "$OUTPUT_LOG_FILE").XXXXXX")"

    if lock_begin_wait 2; then
        cp -f "$OUTPUT_LOG_FILE" "$snapshot" 2>/dev/null || : >"$snapshot"
        lock_end
    else
        : >"$snapshot"
    fi

    clear
    print_task_status
    echo "显存: ${free_memory} MB | 失败: ${FAILED_TASKS}"
    echo "预算占用: ${TOTAL_RESERVED_MB} / ${GPU_USER_LIMIT_MB} MB | 画像粒度: ${MEM_PROFILE_DIM} | keep_free=${GPU_KEEP_FREE_MB}"
    echo "Progress: $OUTPUT_LOG_FILE"
    echo

    # 紧凑展示：一组(task,phase,model)只占一行，并且会“粘住”最近完成的 shot
    render_progress_compact "$snapshot" | column -t -s $'\t' || true

    rm -f "$snapshot"
    sleep "${LOG_INTERVAL:-5}"
}

reap_finished_jobs() {
    local new_running=()
    local pid stat rc name

    for pid in "${running_pids[@]}"; do
        name="${task_map[$pid]:-UNKNOWN}"

        # stat 为空：进程已不存在；stat 以 Z 开头：僵尸（必须 wait 回收）
        stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)"

        if [[ -z "$stat" || "$stat" == Z* ]]; then
            rc=0
            wait "$pid" 2>/dev/null || rc=$?
            # 结束时：写入画像 & 回收预算
            local key peak res
            key="${pid_mem_key[$pid]:-}"
            peak="${pid_peak_mb[$pid]:-0}"
            res="${pid_res_mb[$pid]:-0}"

            if [[ -n "$key" && "$peak" -gt 0 ]]; then
                update_mem_profile "$key" "$peak"
            fi

            if [[ "$res" -gt 0 ]]; then
                TOTAL_RESERVED_MB=$(( TOTAL_RESERVED_MB - res ))
            fi

            unset pid_mem_key["$pid"] pid_res_mb["$pid"] pid_peak_mb["$pid"]

            unset task_map["$pid"]
            DONE_TASKS=$((DONE_TASKS + 1))
            [[ $rc -ne 0 ]] && FAILED_TASKS=$((FAILED_TASKS + 1))

            printf '%s REAP pid=%s rc=%s stat=%s name=%s\n' \
                "$(date '+%F %T')" "$pid" "$rc" "${stat:-NA}" "$name" >> "$SCHED_LOG"
        else
            new_running+=("$pid")
        fi
    done

    running_pids=("${new_running[@]}")
}

scheduler() {
    running_pids=()
    tasks=()

    # --- 把 run_order 分组：base2new / other / crossdata ---
    local -a order_b2n=() order_other=() order_cross=()
    local t
    for t in "${run_order[@]}"; do
      if [[ "$t" == base2new_* ]]; then
        order_b2n+=("$t")
      elif [[ "$t" == "crossdata" || "$t" == crossdata_* ]]; then
        order_cross+=("$t")
      else
        order_other+=("$t")   # 比如 domaingen
      fi
    done

    # --- 生成 tasks：全局 base2new 全部在最前，crossdata 全部在最后 ---
    for shot in "${SHOTS_TO_RUN[@]}"; do
      for seed in $(seq "$START_RUN" "$END_RUN"); do
        for t in "${order_b2n[@]}"; do
          tasks+=("${t}_shots${shot}_seed${seed}")
        done
      done
    done

    for shot in "${SHOTS_TO_RUN[@]}"; do
      for seed in $(seq "$START_RUN" "$END_RUN"); do
        for t in "${order_other[@]}"; do
          tasks+=("${t}_shots${shot}_seed${seed}")
        done
      done
    done

    for shot in "${SHOTS_TO_RUN[@]}"; do
      for seed in $(seq "$START_RUN" "$END_RUN"); do
        for t in "${order_cross[@]}"; do
          tasks+=("${t}_shots${shot}_seed${seed}")
        done
      done
    done

    TOTAL_TASKS=${#tasks[@]}
    DONE_TASKS=0
    FAILED_TASKS=0
    echo "总任务数：$TOTAL_TASKS"
    local last_log_ts=0
    local now free_memory


    while [[ ${#tasks[@]} -gt 0 || ${#running_pids[@]} -gt 0 ]]; do
        # 1) 先回收：确保一结束就释放槽位
        reap_finished_jobs

        # 2) 补位：只要有槽位就尽量启动（直到满/没任务/显存不够）
        while [[ ${#running_pids[@]} -lt $MAX_TASK_NUM && ${#tasks[@]} -gt 0 ]]; do
            maybe_sample_gpu_peak

            local current_task_seed current_task tmp shot seed task_runlog pid
            current_task_seed="${tasks[0]}"

            current_task="${current_task_seed%_shots*}"
            tmp="${current_task_seed#*_shots}"
            shot="${tmp%_seed*}"
            seed="${current_task_seed#*_seed}"

            # 计算该任务批次 key 与预算
            local key reserve_mb free_memory
            key="$(mem_key_for "$current_task" "$shot" "$seed")"
            reserve_mb="$(reserve_for_key "$key")"

            # 未画像的同一 key 默认不并发：先让一个 seed 跑出真实峰值，后续 seed 复用该 dataset+shot 预算。
            # 这样可以避免 INIT_RESERVE_MB 低估时，同一高显存数据集的多个 seed 同时启动导致 OOM。
            if [[ "${ALLOW_PARALLEL_UNPROFILED_SAME_KEY:-0}" != "1" ]]; then
                if ! profile_known_for_key "$key" && same_mem_key_running "$key"; then
                    break
                fi
            fi

            # 两道门槛：
            # 1) 你自己的总预算上限
            if (( TOTAL_RESERVED_MB + reserve_mb > GPU_USER_LIMIT_MB )); then
                break
            fi

            # 2) GPU 真实空闲也要够，并给其他人留 GPU_KEEP_FREE_MB
            free_memory="$(check_free_memory fresh)"
            if (( free_memory < reserve_mb + GPU_KEEP_FREE_MB )); then
                break
            fi

            # 预算满足 -> 真正出队
            tasks=("${tasks[@]:1}")

            task_runlog="${TASKLOG_DIR}/${current_task_seed}.log"
            printf '%s START %s (shot=%s seed=%s reserve=%s key=%s)\n' \
                "$(date '+%F %T')" "$current_task_seed" "$shot" "$seed" "$reserve_mb" "$key" >> "$SCHED_LOG"

            # 启动后台 job（wrapper pid）
            SHOTS="$shot" run_task "$current_task" "$seed" >"$task_runlog" 2>&1 &
            pid=$!

            running_pids+=("$pid")
            task_map["$pid"]="$current_task_seed"

            # 记录预算与 key，并占用预算
            pid_mem_key["$pid"]="$key"
            pid_res_mb["$pid"]="$reserve_mb"
            pid_peak_mb["$pid"]=0
            TOTAL_RESERVED_MB=$(( TOTAL_RESERVED_MB + reserve_mb ))

            # 给 0.2s 让它真正起来，防止秒退占坑
            sleep 0.2
            if ! ps -p "$pid" >/dev/null 2>&1; then
                # 秒退：立刻回收并标失败，不占并发
                rc=0
                wait "$pid" 2>/dev/null || rc=$?
                unset task_map["$pid"]
                DONE_TASKS=$((DONE_TASKS + 1))
                [[ $rc -ne 0 ]] && FAILED_TASKS=$((FAILED_TASKS + 1))
                # 预算回收
                TOTAL_RESERVED_MB=$(( TOTAL_RESERVED_MB - pid_res_mb["$pid"] ))
                unset pid_mem_key["$pid"] pid_res_mb["$pid"] pid_peak_mb["$pid"]
                # 从 running_pids 移除
                local keep=() p
                for p in "${running_pids[@]}"; do [[ "$p" != "$pid" ]] && keep+=("$p"); done
                running_pids=("${keep[@]}")

                printf '%s QUICK-EXIT pid=%s rc=%s name=%s\n' \
                    "$(date '+%F %T')" "$pid" "$rc" "$current_task_seed" >> "$SCHED_LOG"

            fi
        done

        # 3) 定时刷新界面（不影响补位轮询）
        now="$(date +%s)"
        if (( now - last_log_ts >= LOG_INTERVAL )); then
            logging
            last_log_ts=$now
        fi
        maybe_sample_gpu_peak

        sleep "$POLL_INTERVAL"
    done

    # 最后再刷一次
    logging
    echo "所有任务已完成！DONE=$DONE_TASKS FAILED=$FAILED_TASKS TOTAL=$TOTAL_TASKS"
    echo "调度日志：$SCHED_LOG"
}

############################
# 9) 平均值
############################
average() {
    local in_csv="${1:-$RESULT_CSV_FILE}"
    local out_csv="${2:-$AVERAGE_CSV_FILE}"
    declare -A total_accuracy total_kappa total_macro_f1 total_time count

    lock_begin
    : > "$out_csv"
    lock_end

    while IFS=',' read -r total_type type model shot run accuracy kappa macro_f1 time_sec; do
        [[ -z "${total_type:-}" ]] && continue
        run="${run//$'\r'/}"
        shot="${shot//$'\r'/}"

        if [[ $run -ge $START_RUN && $run -le $END_RUN ]]; then
            local key="${total_type},${type},${model},shots_${shot}"

            if [[ ! "$accuracy" =~ ^[0-9]+(\.[0-9]+)?$ ]] || \
               [[ ! "$kappa" =~ ^[0-9]+(\.[0-9]+)?$ ]] || \
               [[ ! "$macro_f1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || \
               [[ ! "$time_sec" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                continue
            fi

            total_accuracy["$key"]=$(echo "${total_accuracy[$key]:-0} + $accuracy" | bc)
            total_kappa["$key"]=$(echo "${total_kappa[$key]:-0} + $kappa" | bc)
            total_macro_f1["$key"]=$(echo "${total_macro_f1[$key]:-0} + $macro_f1" | bc)
            total_time["$key"]=$(echo "${total_time[$key]:-0} + $time_sec" | bc)
            count["$key"]=$(( ${count[$key]:-0} + 1 ))
        fi
    done < "$in_csv"

    for key in "${output_order[@]}"; do
        local num_entries="${count["$key"]:-0}"
        (( num_entries <= 0 )) && continue

        local accuracy_sum="${total_accuracy["$key"]:-0}"
        local kappa_sum="${total_kappa["$key"]:-0}"
        local macro_f1_sum="${total_macro_f1["$key"]:-0}"
        local time_sum="${total_time["$key"]:-0}"

        local avg_accuracy avg_kappa avg_macro_f1 avg_time_sec
        avg_accuracy=$(echo "scale=2; $accuracy_sum / $num_entries" | bc)
        avg_kappa=$(echo "scale=2; $kappa_sum / $num_entries" | bc)
        avg_macro_f1=$(echo "scale=2; $macro_f1_sum / $num_entries" | bc)
        avg_time_sec=$(echo "scale=3; $time_sum / $num_entries" | bc)

        lock_begin
        echo "$key,$avg_accuracy,$avg_kappa,$avg_macro_f1,$avg_time_sec" >> "$out_csv"
        lock_end
    done
}
############################
# 10) 模型可视化
############################
run_visualizations() {
    if [[ "${ENABLE_VIS:-1}" == "0" ]]; then
        echo "[VIS] disabled, skip."
        return 0
    fi

    local shot seed item task model tsne_script cam_script

    echo "========== VIS START ==========" >>"$VIS_LOG_FILE"
    echo "TRAINER=$TRAINER CFG=$CFG SUB=$SUB SHOTS_TO_RUN=${SHOTS_TO_RUN[*]} SEED=${START_RUN}-${END_RUN}" >>"$VIS_LOG_FILE"

    # 显式遍历每个 shot（避免用到“残留的 SHOTS”）
    for shot in "${SHOTS_TO_RUN[@]}"; do
        export SHOTS="$shot"

        for seed in $(seq "$START_RUN" "$END_RUN"); do
            for item in "${VIS_ITEMS[@]}"; do
                IFS=':' read -r task model tsne_script cam_script <<< "$item"

                echo "[VIS] task=$task model=$model shots=$SHOTS seed=$seed" | tee -a "$VIS_LOG_FILE"

                # t-SNE
                bash "$tsne_script" "$model" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" >>"$VIS_LOG_FILE" 2>&1 \
                  || echo "[VIS][WARN] tsne failed: task=$task model=$model shots=$SHOTS seed=$seed" >>"$VIS_LOG_FILE"

                # Grad-CAM
                bash "$cam_script" "$model" "$seed" "$SHOTS" "$TRAINER" "$CFG" "$SUB" >>"$VIS_LOG_FILE" 2>&1 \
                  || echo "[VIS][WARN] gradcam failed: task=$task model=$model shots=$SHOTS seed=$seed" >>"$VIS_LOG_FILE"
            done
        done
    done

    echo "========== VIS DONE ==========" >>"$VIS_LOG_FILE"
}

############################
# 11) 主流程
############################
load_mem_profile
scheduler
clear
if [[ "${SPLIT_CSV_BY_SHOT}" == "0" ]]; then
  average "$RESULT_CSV_FILE" "$AVERAGE_CSV_FILE"
elif [[ "${SPLIT_CSV_BY_SHOT}" == "1" ]]; then
  for _s in "${SHOTS_TO_RUN[@]}"; do
    average "$(result_csv_for_shot "$_s")" "$(average_csv_for_shot "$_s")"
  done
else # 2
  # 先出分 shot
  for _s in "${SHOTS_TO_RUN[@]}"; do
    average "$(result_csv_for_shot "$_s")" "$(average_csv_for_shot "$_s")"
  done
  # 再出总汇总
  average "$RESULT_CSV_FILE" "$AVERAGE_CSV_FILE"
fi
run_visualizations
if [[ "${ENABLE_VIS:-1}" == "1" ]]; then
  echo "平均值与可视化计算完毕，脚本结束。"
else
  echo "平均值计算完毕（已关闭可视化），脚本结束。"
fi