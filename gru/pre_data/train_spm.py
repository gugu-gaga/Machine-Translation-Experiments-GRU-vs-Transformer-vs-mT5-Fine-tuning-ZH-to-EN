#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: Train SentencePiece joint BPE (ZH+EN) on TRAIN split only.
- No shuffling (SPM reads files sequentially)
Output:
  data/spm/spm_zh_en_16k.model
  data/spm/spm_zh_en_16k.vocab
"""

import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", default="data/clean")
    ap.add_argument("--out_dir", default="data/spm")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--character_coverage", type=float, default=0.9995)
    ap.add_argument("--model_type", default="bpe", choices=["bpe", "unigram", "char", "word"])
    ap.add_argument("--model_prefix", default="spm_zh_en_16k")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_src = os.path.join(args.clean_dir, "train.src")
    train_tgt = os.path.join(args.clean_dir, "train.tgt")

    try:
        import sentencepiece as spm
    except ImportError as e:
        raise SystemExit("Install sentencepiece: pip install sentencepiece") from e

    input_files = f"{train_src},{train_tgt}"
    out_prefix = os.path.join(args.out_dir, args.model_prefix)

    spm.SentencePieceTrainer.Train(
        input=input_files,
        model_prefix=out_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        user_defined_symbols=[],
        train_extremely_large_corpus=True
    )
    print(f"Saved: {out_prefix}.model / {out_prefix}.vocab")
if __name__ == "__main__":
    main()
