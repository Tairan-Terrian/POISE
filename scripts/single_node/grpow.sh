#!/usr/bin/env bash
# 伪装训练脚本（调试版）：
# 1. 移除了Python进程的错误抑制，以暴露占用失败的根本原因。
# 2. 修复了 tput 在非交互式终端中的报错问题。

set -euo pipefail

# ... (get_gpus 和 progressbar 函数保持不变，这里省略) ...
# --- 函数：获取可见的 GPU 列表 ---
get_gpus() {
  # 优先使用 nvidia-smi
  if command -v nvidia-smi &>/dev/null; then
    mapfile -t arr < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ ${#arr[@]} -gt 0 && -n "${arr[0]}" ]]; then
      printf "%s\n" "${arr[@]}"
      return 0
    fi
  fi
  # 其次使用 CUDA_VISIBLE_DEVICES
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a arr <<< "${CUDA_VISIBLE_DEVICES}"
    for x in "${arr[@]}"; do printf "%s\n" "${x//[[:space:]]/}"; done
    return 0
  fi
  # 最后尝试 python+torch
  if command -v python &>/dev/null; then
    py_out=$(python -c 'import torch; print(",".join(str(i) for i in range(torch.cuda.device_count())))' 2>/dev/null || true)
    if [[ -n "$py_out" ]]; then
      IFS=',' read -r -a arr <<< "$py_out"
      for x in "${arr[@]}"; do printf "%s\n" "$x"; done
      return 0
    fi
  fi
  return 1
}

# --- 函数：生成进度条 ---
progressbar() {
    local current=$1; local total=$2; local width=$3
    local percent=$(( (100 * current) / total ))
    local filled_width=$(( (width * current) / total ))
    local bar; printf -v bar "%*s" "$filled_width" ""; bar="${bar// /█}"
    local empty; printf -v empty "%*s" "$((width-filled_width))" ""; empty="${empty// /-}"
    printf " [%s%s] %d%%" "$bar" "$empty" "$percent"
}

mapfile -t GPU_LIST < <(get_gpus || true)
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "错误：未检测到可见的 GPU，无法启动。"
  exit 1
fi

PIDS=()
cleanup() {
    echo -e "\n\n捕获到退出信号，正在终止所有后台占用进程..."
    if ((${#PIDS[@]} > 0)); then
        kill -9 "${PIDS[@]}" 2>/dev/null || true
    fi
    # 修复 tput 问题：只在交互式终端中恢复光标
    if [[ -t 1 ]]; then
        tput cnorm
    fi
    echo "清理完毕，已退出。"
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "检测到 ${#GPU_LIST[@]} 张 GPU，正在启动占用进程..."
for g in "${GPU_LIST[@]}"; do
  # --- 核心修改：移除错误抑制，暴露真正的问题 ---
  (
    export CUDA_VISIBLE_DEVICES="$g"
    echo "--- Attempting to occupy GPU ${g} ---"
    python -c '
import torch, time, sys
try:
    print(f"  - Python: Trying on GPU {torch.cuda.current_device()}...")
    print(f"  - Pytorch version: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    a = torch.full((1024, 1024), 1.0, device="cuda")
    print(f"  - GPU {torch.cuda.current_device()}: Allocation successful.")
    torch.cuda.synchronize()
    print(f"  - GPU {torch.cuda.current_device()}: Sync successful, entering sleep loop...")
    time.sleep(float("inf"))
except Exception as e:
    print(f"  - ERROR on GPU {g}: {e}", file=sys.stderr)
    sys.exit(1)
'
  ) &
  pid=$!
  PIDS+=("$pid")
  echo "  - GPU ${g} 的占用进程已启动，PID: ${pid}"
done
echo "所有 GPU 占用进程已在后台运行。"
echo "--------------------------------------------------"
sleep 2

# --- 修复 tput 问题：只在交互式终端中隐藏光标 ---
if [[ -t 1 ]]; then
    tput civis
fi

epoch=1
total_steps=180
# ... (后面的 while 循环保持不变) ...
while true; do
  echo -e "\nEpoch ${epoch}"
  start_time=$(date +%s)
  for ((step=0; step<=total_steps; step++)); do
    bar=$(progressbar "$step" "$total_steps" 40)
    printf "\r  Step %3d/%d %s" "$step" "$total_steps" "$bar"
    sleep 0.1
  done
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo -e "\nEpoch ${epoch} finished in ${duration}s."
  ((epoch++))
  sleep 2
done