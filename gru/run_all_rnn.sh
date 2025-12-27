#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

NPROC=${NPROC:-2}
for c in configs/rnn_*yaml; do
  echo "=== RUN $c ==="
  torchrun --nproc_per_node=$NPROC main/train_rnn.py --config $c
done
