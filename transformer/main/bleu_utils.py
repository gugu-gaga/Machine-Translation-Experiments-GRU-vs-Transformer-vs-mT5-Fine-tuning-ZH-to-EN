from __future__ import annotations
from typing import List, Dict, Tuple
import sentencepiece as spm
import sacrebleu

def bucket_id(L: int, max_len: int) -> str:
    if L > max_len: return f"overlong(>{max_len})"
    if L <= 32: return "short(<=32)"
    if L <= 64: return "medium(33-64)"
    return f"long(65-{max_len})"

def bleu_and_signature(hyps: List[str], refs: List[str], tokenize: str = "13a") -> Tuple[float, str]:
    metric = sacrebleu.metrics.BLEU(tokenize=tokenize)
    score = metric.corpus_score(hyps, [refs])
    sig_obj = metric.get_signature() if hasattr(metric, "get_signature") else None
    sig = sig_obj.format() if (sig_obj is not None and hasattr(sig_obj, "format")) else (str(sig_obj) if sig_obj is not None else f"legacy_sacrebleu(tokenize={tokenize})")
    return float(score.score), sig

def bucket_bleu(src_lines: List[str], hyps: List[str], refs: List[str], spm_model: str, max_len: int, tokenize: str="13a"):
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    src_lens = [len(sp.encode(s, out_type=int)) for s in src_lines]
    idx_by: Dict[str, List[int]] = {}
    for i,L in enumerate(src_lens):
        b = bucket_id(L, max_len)
        idx_by.setdefault(b, []).append(i)
    out = {}
    for b, idxs in idx_by.items():
        bh = [hyps[i] for i in idxs]
        br = [refs[i] for i in idxs]
        bb,_ = bleu_and_signature(bh, br, tokenize=tokenize)
        out[b] = {"count": len(idxs), "bleu": float(bb)}
    return out, src_lens
