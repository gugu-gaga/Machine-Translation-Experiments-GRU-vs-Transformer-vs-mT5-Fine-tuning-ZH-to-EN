#!/usr/bin/env bash
set -euo pipefail
python pre_data/prepare_data.py
python pre_data/train_spm.py
python pre_data/encode_dataset.py --spm_model data/spm/spm_zh_en_16k.model --max_len 128
echo "Done. Reports in data/report/"
