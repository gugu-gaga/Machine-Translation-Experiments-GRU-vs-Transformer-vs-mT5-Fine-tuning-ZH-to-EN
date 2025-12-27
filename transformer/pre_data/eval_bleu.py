#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate EN outputs with sacreBLEU for ZH->EN.

Reports:
- BLEU(all)
- BLEU(no-trunc subset): only examples with src_len<=max_len
- Bucket BLEU by src subword length (FULL length, before trunc):
    short (<=32), medium (33-64), long (65-max_len), overlong (>max_len)
- Signature printed & saved
- Saves JSON report including 5 example triplets.

Notes:
- Reference EN is NEVER truncated.
- Hypothesis EN should be detokenized text (SPM-decoded) before evaluation.
"""

import argparse
import json
from typing import List, Tuple, Dict

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]

def bucket_id(src_len_full: int, max_len: int) -> str:
    if src_len_full > max_len:
        return f"overlong(>{max_len})"
    if src_len_full <= 32:
        return "short(<=32)"
    if src_len_full <= 64:
        return "medium(33-64)"
    return f"long(65-{max_len})"

def compute_bleu(hyps: List[str], refs: List[str], tokenize: str) -> Tuple[float, str]:
    try:
        import sacrebleu
    except ImportError as e:
        raise SystemExit("Install sacrebleu: pip install sacrebleu") from e
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
    return float(bleu.score), bleu.signature

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--spm_model", required=True)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--tokenize", default="13a")
    ap.add_argument("--out_json", default="bleu_report.json")
    ap.add_argument("--num_examples", type=int, default=5)
    args = ap.parse_args()

    src = load_lines(args.src)
    ref = load_lines(args.ref)
    hyp = load_lines(args.hyp)
    assert len(src) == len(ref) == len(hyp), "src/ref/hyp line count mismatch"

    try:
        import sentencepiece as spm
    except ImportError as e:
        raise SystemExit("Install sentencepiece: pip install sentencepiece") from e

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    src_lens_full = [len(sp.encode(s, out_type=int)) for s in src]
    trunc_flags = [1 if L > args.max_len else 0 for L in src_lens_full]

    overall_bleu, signature = compute_bleu(hyp, ref, tokenize=args.tokenize)

    idx_keep = [i for i, flag in enumerate(trunc_flags) if flag == 0]
    bleu_keep = None
    if idx_keep:
        bleu_keep, _ = compute_bleu([hyp[i] for i in idx_keep], [ref[i] for i in idx_keep], tokenize=args.tokenize)

    idx_by_bucket: Dict[str, List[int]] = {}
    for i, L in enumerate(src_lens_full):
        b = bucket_id(L, args.max_len)
        idx_by_bucket.setdefault(b, []).append(i)

    buckets = {}
    for b, idxs in idx_by_bucket.items():
        b_bleu, _ = compute_bleu([hyp[i] for i in idxs], [ref[i] for i in idxs], tokenize=args.tokenize)
        buckets[b] = {"count": len(idxs), "bleu": b_bleu}

    examples = []
    for i in range(min(args.num_examples, len(src))):
        examples.append({
            "src_zh": src[i],
            "ref_en": ref[i],
            "hyp_en": hyp[i],
            "src_len_subword_full": src_lens_full[i],
            "is_truncated_input": bool(trunc_flags[i]),
            "bucket": bucket_id(src_lens_full[i], args.max_len),
        })

    report = {
        "overall": {"bleu": overall_bleu, "signature": signature, "n": len(src)},
        "subset_no_trunc": {"bleu": bleu_keep, "n": len(idx_keep)},
        "truncation": {"max_len": args.max_len, "num_truncated": sum(trunc_flags),
                       "rate_percent": 0.0 if len(src)==0 else 100.0*sum(trunc_flags)/len(src)},
        "buckets": buckets,
        "examples": examples,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"BLEU(all) = {overall_bleu:.4f}")
    print(f"Signature: {signature}")
    if bleu_keep is None:
        print("BLEU(no-trunc subset) = N/A")
    else:
        print(f"BLEU(no-trunc subset) = {bleu_keep:.4f} (n={len(idx_keep)}/{len(src)})")
    print(f"Truncated sources: {sum(trunc_flags)} / {len(src)} ({report['truncation']['rate_percent']:.2f}%)")
    print("Bucket BLEU:")
    for b in sorted(buckets.keys()):
        print(f"  {b}: n={buckets[b]['count']} BLEU={buckets[b]['bleu']:.4f}")
    print(f"Saved: {args.out_json}")

if __name__ == "__main__":
    main()
