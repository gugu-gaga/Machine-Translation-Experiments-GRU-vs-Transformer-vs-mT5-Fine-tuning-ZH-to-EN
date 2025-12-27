#!/usr/bin/env bash
set -euo pipefail

SPM_MODEL=${SPM_MODEL:-data/spm/spm_zh_en_16k.model}
CLEAN_DIR=${CLEAN_DIR:-data/clean}
OUT_DIR=${OUT_DIR:-data/spm}
CORPUS_DIR=${CORPUS_DIR:-data/spm_corpus}
DIM=${DIM:-512}
EPOCHS=${EPOCHS:-10}
WS=${WS:-5}
NEG=${NEG:-10}
THREADS=${THREADS:-8}

python pre_train/export_spm_pieces_corpus.py   --spm_model "$SPM_MODEL"   --clean_dir "$CLEAN_DIR"   --out_dir "$CORPUS_DIR"   --splits train   --join_src_tgt

python pre_train/pretrain_fasttext.py   --corpus "$CORPUS_DIR/train.joint.pieces"   --out_dir "$OUT_DIR"   --dim "$DIM"   --epochs "$EPOCHS"   --ws "$WS"   --neg "$NEG"   --threads "$THREADS"

python pre_train/export_emb_init.py   --spm_model "$SPM_MODEL"   --ft_model "$OUT_DIR/fasttext_sg_dim${DIM}.model"   --out_npy "$OUT_DIR/emb_init_fasttext_${DIM}.npy"   --out_meta "$OUT_DIR/emb_init_fasttext_${DIM}.meta.json"   --pad_zero   --special_mean   --l2_normalize

echo "Done -> $OUT_DIR/emb_init_fasttext_${DIM}.npy"
