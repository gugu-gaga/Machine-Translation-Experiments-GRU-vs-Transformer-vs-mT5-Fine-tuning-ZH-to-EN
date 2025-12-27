# Machine Translation Experiments: GRU vs Transformer vs mT5 Fine-tuning (ZH→EN)

This repository implements a **Chinese → English machine translation** pipeline with **three model families**:

1) GRU-based seq2seq (6 experiments, trained from scratch)
2) Transformer (7 experiments, trained from scratch)
3) mT5-small (fine-tuned from a pretrained checkpoint; config-driven)

The intended workflow is:

1. Train (inside each experiment folder)  
2. Evaluate on the test set (root `inference.py`)  
3. Collect model size/parameter stats (root `model_stats.py`)  
4. Optional interactive translation (`translate_chat.py`)  

## Environment

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Project layout (root)

From the repository root:

```bash
ls
# final_test_eval  gru  inference.py  mT5_finetune  model_stats  model_stats.py
# models.txt  mt5_small_offline  transformer  translate_chat.py requirements.txt README.md

# -----------------------------
# (A) Train GRU experiments
# -----------------------------
cd gru

# run_predata.sh:
#   Build the SentencePiece tokenizer and preprocess the dataset for the GRU pipeline
#   (e.g., train SPM, tokenize, and generate processed files needed for training).
bash run_predata.sh

# run_pretrain.sh:
#   Train fastText word embeddings used by the GRU seq2seq model (embedding initialization).
bash run_pretrain.sh

# run_all_rnn.sh:
#   Launch all GRU experiments (6 runs) defined in ./configs/
#   and save checkpoints/logs under ./runs/.
bash run_all_rnn.sh
cd ..

# -----------------------------
# (B) Train Transformer experiments
# -----------------------------
cd transformer

# run_predata.sh:
#   Build the SentencePiece tokenizer and preprocess the dataset for the Transformer pipeline.
bash run_predata.sh

# run_pretrain.sh:
#   Train fastText word embeddings (if required by your Transformer configs / initialization).
bash run_pretrain.sh

# run_all_transformer.sh:
#   Launch all Transformer experiments (7 runs) defined in ./configs/
#   and save checkpoints/logs under ./runs/ (and/or ./outputs/ depending on config).
bash run_all_transformer.sh
cd ..

# -----------------------------
# (C) Fine-tune mT5-small
# -----------------------------
cd mT5_finetune

# run_mt5.sh:
#   Run mT5-small fine-tuning using configuration files under ./configs/
#   and save HuggingFace-style checkpoints under ./runs/.
bash run_mt5.sh
cd ..

# -----------------------------
# (D) Test-set evaluation (all models)
# -----------------------------
# inference.py:
#   Unified evaluation script for ALL trained models (GRU / Transformer / mT5).
#   It runs decoding on the test set, computes BLEU with sacrebleu,
#   and writes detailed outputs (predictions/metrics/analysis artifacts) to --outdir.
#
# Notes:
#   - For full evaluation across many checkpoints, you typically pass --models_list <file>
#     (a text file with one checkpoint path per line).
#   - --beams evaluates multiple decoding beam sizes (e.g., 1 and 5).
#   - --max_examples dumps some examples for qualitative inspection.
python inference.py \
  --outdir final_test_eval \
  --beams 1 5 \
  --split test \
  --max_new_tokens 128 \
  --max_examples 200

# -----------------------------
# (E) Model stats (params + size)
# -----------------------------
# model_stats.py:
#   Utility script to summarize model statistics, including:
#   - parameter count (e.g., counting tensors in .pt or .safetensors)
#   - checkpoint file size on disk
#   It outputs a JSONL summary file to --out, typically used in your final report.
python model_stats.py --root . --out final_test_eval/model_stats.jsonl

# -----------------------------
# (F) Interactive translation
# -----------------------------
# translate_chat.py:
#   Interactive translator for quick qualitative testing:
#   - you select a model from a TSV model list (models.txt)
#   - type any Chinese sentence
#   - get the English translation with chosen decoding settings 
#
# --models:
#   TSV model list (e.g., "models.txt") with 14 entries, formatted as:
#     <model_name>\t<checkpoint_path>

python translate_chat.py \
  --models models_14.txt \
  --device cuda \
  --beam 5 \
  --inference_py /path/to/your/repo_root/inference.py
```