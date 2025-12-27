#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: Encode dataset with SentencePiece, apply:
- TRAIN: drop sentence pairs where src_len>max_len OR tgt_len>max_len
- VALID/TEST: DO NOT drop; truncate ONLY src ids to max_len for model input;
              keep target/reference FULL (no truncation) for BLEU.

Outputs (out_dir, default data/clean):
- {split}.src / {split}.tgt (text; train may be filtered)
- {split}.src.ids (int ids; valid/test: truncated src)
- {split}.src.trunc_flag (0/1 per line; valid/test only)
- {split}.tgt.ids (FULL target ids; valid/test never truncated)
Reports (report_dir):
- data_report.json
- data_report.md
"""

import argparse
import os
import json
import statistics
from typing import List, Dict, Any

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]

def write_lines(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")

def write_ids(path: str, ids_list: List[List[int]]):
    with open(path, "w", encoding="utf-8") as f:
        for ids in ids_list:
            f.write(" ".join(map(str, ids)) + "\n")

def write_flags(path: str, flags: List[int]):
    with open(path, "w", encoding="utf-8") as f:
        for x in flags:
            f.write(str(int(x)) + "\n")

def summarize(lengths: List[float]) -> Dict[str, Any]:
    if not lengths:
        return {}
    xs = sorted(lengths)
    def q(p: float) -> float:
        idx = int(p * (len(xs) - 1))
        return xs[idx]
    mean = sum(xs) / len(xs)
    std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return {
        "count": len(xs),
        "mean": mean,
        "std": std,
        "median": q(0.5),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": xs[-1],
    }

def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else 100.0 * n / d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", default="data/clean")
    ap.add_argument("--out_dir", default="data/clean")
    ap.add_argument("--report_dir", default="data/report")
    ap.add_argument("--spm_model", required=True)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    try:
        import sentencepiece as spm
    except ImportError as e:
        raise SystemExit("Install sentencepiece: pip install sentencepiece") from e

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    unk_id = sp.unk_id()

    report: Dict[str, Any] = {
        "task": "ZH->EN",
        "spm_model": args.spm_model,
        "max_len_subword": args.max_len,
        "policy": {
            "train": "drop pairs if src_len>max_len or tgt_len>max_len",
            "valid_test": "truncate ONLY src ids to max_len; keep full target/reference",
        },
        "splits": {},
        "unk": {},
        "duplication": {},
    }

    # duplication stats on TRAIN only (exact text pairs)
    train_pairs_seen = set()
    train_dup = 0

    global_total_tokens = 0
    global_unk_tokens = 0
    unk_examples = {"src": [], "tgt": []}

    for split in ["train", "valid", "test"]:
        src_path = os.path.join(args.clean_dir, f"{split}.src")
        tgt_path = os.path.join(args.clean_dir, f"{split}.tgt")
        src_lines = load_lines(src_path)
        tgt_lines = load_lines(tgt_path)
        assert len(src_lines) == len(tgt_lines), f"Line mismatch: {split}"

        before = len(src_lines)

        kept_src, kept_tgt = [], []
        src_ids_out, tgt_ids_out = [], []
        trunc_flags = []
        dropped_overlong = 0
        trunc_count = 0

        src_lens_full, tgt_lens_full = [], []
        src_lens_used = []
        len_ratio = []

        for s, t in zip(src_lines, tgt_lines):
            s_ids_full = sp.encode(s, out_type=int)
            t_ids_full = sp.encode(t, out_type=int)

            s_len_full = len(s_ids_full)
            t_len_full = len(t_ids_full)

            if split == "train":
                key = (s, t)
                if key in train_pairs_seen:
                    train_dup += 1
                else:
                    train_pairs_seen.add(key)

                if s_len_full > args.max_len or t_len_full > args.max_len:
                    dropped_overlong += 1
                    continue
                s_ids_used = s_ids_full
            else:
                if s_len_full > args.max_len:
                    s_ids_used = s_ids_full[:args.max_len]
                    trunc_flags.append(1)
                    trunc_count += 1
                else:
                    s_ids_used = s_ids_full
                    trunc_flags.append(0)

            kept_src.append(s)
            kept_tgt.append(t)

            src_ids_out.append(s_ids_used)
            tgt_ids_out.append(t_ids_full)  # FULL targets

            src_lens_full.append(s_len_full)
            tgt_lens_full.append(t_len_full)
            src_lens_used.append(len(s_ids_used))
            if t_len_full > 0:
                len_ratio.append(len(s_ids_used) / t_len_full)

            global_total_tokens += (s_len_full + t_len_full)
            global_unk_tokens += sum(1 for x in s_ids_full if x == unk_id)
            global_unk_tokens += sum(1 for x in t_ids_full if x == unk_id)

            if unk_id in s_ids_full and len(unk_examples["src"]) < 10:
                unk_examples["src"].append(s)
            if unk_id in t_ids_full and len(unk_examples["tgt"]) < 10:
                unk_examples["tgt"].append(t)

        after = len(kept_src)

        write_lines(os.path.join(args.out_dir, f"{split}.src"), kept_src)
        write_lines(os.path.join(args.out_dir, f"{split}.tgt"), kept_tgt)
        write_ids(os.path.join(args.out_dir, f"{split}.src.ids"), src_ids_out)
        write_ids(os.path.join(args.out_dir, f"{split}.tgt.ids"), tgt_ids_out)

        if split in ["valid", "test"]:
            write_flags(os.path.join(args.out_dir, f"{split}.src.trunc_flag"), trunc_flags)

        report["splits"][split] = {
            "num_lines_before": before,
            "num_lines_after": after,
            "dropped_overlong_pairs": dropped_overlong,
            "src_len_subword_full": summarize([float(x) for x in src_lens_full]),
            "tgt_len_subword_full": summarize([float(x) for x in tgt_lens_full]),
            "src_len_subword_used_for_model_input": summarize([float(x) for x in src_lens_used]),
            "valid_test_src_trunc_count": trunc_count if split in ["valid", "test"] else 0,
            "len_ratio_used_src_over_full_tgt": summarize([float(x) for x in len_ratio]),
        }

        print(f"[{split}] before={before} after={after} dropped_overlong={dropped_overlong} src_trunc_count={trunc_count}")

    report["unk"] = {
        "unk_id": int(unk_id),
        "unk_token_rate_percent": pct(global_unk_tokens, global_total_tokens),
        "total_tokens_full": int(global_total_tokens),
        "unk_tokens_full": int(global_unk_tokens),
        "examples": unk_examples,
    }

    report["duplication"] = {
        "train_unique_pairs": int(len(train_pairs_seen)),
        "train_exact_duplicate_pairs": int(train_dup),
        "train_total_pairs_after_filter": int(report["splits"]["train"]["num_lines_after"]),
        "train_duplicate_rate_percent_over_after": pct(train_dup, report["splits"]["train"]["num_lines_after"]),
    }

    json_path = os.path.join(args.report_dir, "data_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(args.report_dir, "data_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Data Report (ZH→EN MT)\n\n")
        f.write(f"- SPM model: `{args.spm_model}`\n")
        f.write(f"- max_len (subword): `{args.max_len}`\n")
        f.write("- Train policy: drop pairs over max_len on either side\n")
        f.write("- Valid/Test policy: truncate ONLY source ids; keep full target reference\n\n")

        for split in ["train", "valid", "test"]:
            sr = report["splits"][split]
            f.write(f"## Split: {split}\n")
            f.write(f"- lines before: {sr['num_lines_before']}\n")
            f.write(f"- lines after: {sr['num_lines_after']}\n")
            f.write(f"- dropped_overlong_pairs: {sr['dropped_overlong_pairs']}\n")
            if split in ["valid", "test"]:
                f.write(f"- src_trunc_count: {sr['valid_test_src_trunc_count']}\n")
            f.write("\n### Subword length (src full)\n")
            f.write(f"```json\n{json.dumps(sr['src_len_subword_full'], ensure_ascii=False, indent=2)}\n```\n")
            f.write("### Subword length (tgt full)\n")
            f.write(f"```json\n{json.dumps(sr['tgt_len_subword_full'], ensure_ascii=False, indent=2)}\n```\n")
            f.write("### Subword length (src used for model input)\n")
            f.write(f"```json\n{json.dumps(sr['src_len_subword_used_for_model_input'], ensure_ascii=False, indent=2)}\n```\n")
            f.write("### Length ratio (used src / full tgt)\n")
            f.write(f"```json\n{json.dumps(sr['len_ratio_used_src_over_full_tgt'], ensure_ascii=False, indent=2)}\n```\n\n")

        f.write("## UNK statistics\n")
        f.write(f"- unk token rate (%): {report['unk']['unk_token_rate_percent']:.6f}\n")
        f.write(f"- total tokens (full): {report['unk']['total_tokens_full']}\n")
        f.write(f"- unk tokens (full): {report['unk']['unk_tokens_full']}\n\n")
        f.write("### UNK examples (up to 10 each)\n")
        f.write("#### src examples\n")
        for x in report["unk"]["examples"]["src"]:
            f.write(f"- {x}\n")
        f.write("\n#### tgt examples\n")
        for x in report["unk"]["examples"]["tgt"]:
            f.write(f"- {x}\n")
        f.write("\n\n## Duplication (train, exact text pairs)\n")
        f.write(f"- train unique pairs: {report['duplication']['train_unique_pairs']}\n")
        f.write(f"- train exact duplicate pairs: {report['duplication']['train_exact_duplicate_pairs']}\n")
        f.write(f"- duplicate rate (% over after-filter train): {report['duplication']['train_duplicate_rate_percent_over_after']:.6f}\n")

    print(f"Saved reports: {md_path}, {json_path}")
if __name__ == "__main__":
    main()
