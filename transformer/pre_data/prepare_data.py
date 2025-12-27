#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Read JSONL -> basic clean -> write .src/.tgt text (ZH->EN)

Input (default):
  data/raw/train_100k.jsonl
  data/raw/valid.jsonl
  data/raw/test.jsonl

Each line example:
  {"en": "...", "zh": "...", "index": 0}

Output:
  data/clean/train.src, train.tgt
  data/clean/valid.src, valid.tgt
  data/clean/test.src,  test.tgt

Notes:
- This stage does NOT do max_len filtering, because it needs a tokenizer (SPM).
- Only minimal normalization: remove control chars, normalize whitespace.
"""

import argparse
import json
import os
import re
from typing import Tuple

CTRL_RE = re.compile(r"[\u0000-\u001F\u007F]")

def normalize_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = CTRL_RE.sub("", s)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def read_jsonl(path: str, src_key: str, tgt_key: str) -> Tuple[list, list, dict]:
    srcs, tgts = [], []
    stats = {"total": 0, "kept": 0, "dropped_empty": 0, "dropped_badjson": 0}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stats["total"] += 1
            line = line.strip()
            if not line:
                stats["dropped_empty"] += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["dropped_badjson"] += 1
                continue
            src = normalize_text(str(obj.get(src_key, "")))
            tgt = normalize_text(str(obj.get(tgt_key, "")))
            if not src or not tgt:
                stats["dropped_empty"] += 1
                continue
            srcs.append(src)
            tgts.append(tgt)
            stats["kept"] += 1
    return srcs, tgts, stats

def write_lines(path: str, lines: list):
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/clean")
    ap.add_argument("--train_file", default="train_100k.jsonl")
    ap.add_argument("--valid_file", default="valid.jsonl")
    ap.add_argument("--test_file", default="test.jsonl")
    ap.add_argument("--src_key", default="zh")
    ap.add_argument("--tgt_key", default="en")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split, fn in [("train", args.train_file), ("valid", args.valid_file), ("test", args.test_file)]:
        in_path = os.path.join(args.raw_dir, fn)
        srcs, tgts, st = read_jsonl(in_path, args.src_key, args.tgt_key)
        write_lines(os.path.join(args.out_dir, f"{split}.src"), srcs)
        write_lines(os.path.join(args.out_dir, f"{split}.tgt"), tgts)
        print(f"[{split}] {in_path}")
        print(f"  total={st['total']} kept={st['kept']} dropped_empty={st['dropped_empty']} dropped_badjson={st['dropped_badjson']}")
if __name__ == "__main__":
    main()
