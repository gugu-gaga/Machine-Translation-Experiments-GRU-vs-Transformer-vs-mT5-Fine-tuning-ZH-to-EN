#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/mt5_run.yaml}

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 保证只用两张卡
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# 让 python 能 import main.*
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 让日志更“实时”（避免缓冲导致日志晚出现）
export PYTHONUNBUFFERED=1

echo "[run_mt5_2gpus] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_mt5_2gpus] CONFIG=$CONFIG"
echo "[run_mt5_2gpus] PWD=$(pwd)"

# ====== 从 YAML 读取 train.output_dir，并在其下保存 console 日志 ======
OUTDIR="$(
python - <<'PY' "$CONFIG" 2>/dev/null || true
import sys
try:
    import yaml
except Exception:
    sys.exit(1)
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg["train"]["output_dir"])
PY
)"

# 如果 python/yaml 读取失败，做一个简易 fallback（要求 output_dir 单行写法）
if [[ -z "${OUTDIR}" ]]; then
  OUTDIR="$(grep -E '^\s*output_dir\s*:\s*' -m 1 "$CONFIG" | sed -E 's/^\s*output_dir\s*:\s*//')"
fi

# 去掉可能的引号 & 尾部 /
OUTDIR="${OUTDIR%\"}"; OUTDIR="${OUTDIR#\"}"
OUTDIR="${OUTDIR%\'}"; OUTDIR="${OUTDIR#\'}"
OUTDIR="${OUTDIR%/}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="$(basename "$CONFIG")"
LOG_DIR="${OUTDIR}/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/console.${EXP_NAME}.${RUN_ID}.log"

echo "[run_mt5_2gpus] OUTPUT_DIR=$OUTDIR"
echo "[run_mt5_2gpus] LOG_FILE=$LOG_FILE"

# 这行非常关键：把后续所有 stdout/stderr 既输出到屏幕，又写入文件
exec > >(tee -a "$LOG_FILE") 2>&1

torchrun --nproc_per_node=2 --master_port=29501 \
  -m main.train_mt5 --config "$CONFIG"
