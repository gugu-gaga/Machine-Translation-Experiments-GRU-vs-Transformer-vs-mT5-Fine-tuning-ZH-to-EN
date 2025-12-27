#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from typing import List

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]

def bucket_id(src_len: int, max_len: int) -> str:
    if src_len > max_len:
        return f"overlong(>{max_len})"
    if src_len <= 32:
        return "short(<=32)"
    if src_len <= 64:
        return "medium(33-64)"
    return f"long(65-{max_len})"

def compute_bleu(hyps: List[str], refs: List[str], tokenize: str):
    """
    Robust across sacrebleu versions:
    - score objects may NOT have `.signature`
    - use metric.get_signature() instead
    """
    import sacrebleu
    metric = sacrebleu.metrics.BLEU(tokenize=tokenize)
    score = metric.corpus_score(hyps, [refs])

    # signature from metric (stable)
    sig_obj = metric.get_signature() if hasattr(metric, "get_signature") else None
    if sig_obj is None:
        signature = f"legacy_sacrebleu(tokenize={tokenize})"
    else:
        signature = sig_obj.format() if hasattr(sig_obj, "format") else str(sig_obj)

    # score.score exists across versions
    return float(score.score), signature

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--spm_model", required=True)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--tokenize", default="13a")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--num_examples", type=int, default=5)
    args = ap.parse_args()

    src = load_lines(args.src)
    ref = load_lines(args.ref)
    hyp = load_lines(args.hyp)
    assert len(src) == len(ref) == len(hyp), "src/ref/hyp must have same number of lines"

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    src_lens = [len(sp.encode(s, out_type=int)) for s in src]

    overall_bleu, signature = compute_bleu(hyp, ref, tokenize=args.tokenize)

    idx_by_bucket = {}
    for i, L in enumerate(src_lens):
        b = bucket_id(L, args.max_len)
        idx_by_bucket.setdefault(b, []).append(i)

    buckets = {}
    for b, idxs in idx_by_bucket.items():
        bh = [hyp[i] for i in idxs]
        br = [ref[i] for i in idxs]
        b_bleu, _ = compute_bleu(bh, br, tokenize=args.tokenize)
        buckets[b] = {"count": len(idxs), "bleu": float(b_bleu)}

    examples = []
    for i in range(min(args.num_examples, len(src))):
        examples.append({
            "src_zh": src[i],
            "ref_en": ref[i],
            "hyp_en": hyp[i],
            "src_len_subword": int(src_lens[i]),
            "bucket": bucket_id(src_lens[i], args.max_len),
        })

    report = {
        "overall": {"bleu": overall_bleu, "signature": signature, "n": len(src)},
        "buckets": buckets,
        "examples": examples,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"BLEU = {overall_bleu:.4f}")
    print(f"Signature: {signature}")
    for b in sorted(buckets.keys()):
        print(f"{b}: n={buckets[b]['count']} BLEU={buckets[b]['bleu']:.4f}")
    print(f"Saved: {args.out_json}")

if __name__ == "__main__":
    main()
