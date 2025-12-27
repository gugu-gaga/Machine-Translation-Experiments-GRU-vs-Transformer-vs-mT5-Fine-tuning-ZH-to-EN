# main/bleu.py
from typing import List, Tuple

def bleu_and_signature(hyps: List[str], refs: List[str], tokenize: str = "13a") -> Tuple[float, str]:
    import sacrebleu
    metric = sacrebleu.metrics.BLEU(tokenize=tokenize)
    score = metric.corpus_score(hyps, [refs])
    sig = getattr(score, "signature", None)
    if sig is None:
        sig = f"tok:{tokenize}|sacrebleu:{getattr(sacrebleu, '__version__', 'unknown')}"
    else:
        sig = sig.format() if hasattr(sig, "format") else str(sig)
    return float(score.score), str(sig)
