#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu


def _read_parallel(src_path: str, tgt_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows = []
    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        for i, (s, t) in enumerate(zip(fs, ft)):
            if limit is not None and i >= limit:
                break
            rows.append({"zh": s.rstrip("\n"), "en": t.rstrip("\n")})
    return rows


def _read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            rows.append({"zh": obj["zh"], "en": obj["en"]})
    return rows


def resolve_dataset(cfg: Dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    d = cfg["data"]
    def load_split(kind: str) -> List[Dict[str, str]]:
        jsonl = d.get(f"{kind}_jsonl")
        src = d.get(f"{kind}_src_txt")
        tgt = d.get(f"{kind}_tgt_txt")
        lim = d.get(f"limit_{kind}")
        if jsonl and os.path.exists(jsonl):
            return _read_jsonl(jsonl, lim)
        if src and tgt and os.path.exists(src) and os.path.exists(tgt):
            return _read_parallel(src, tgt, lim)
        raise FileNotFoundError(f"Missing data for split={kind}. Provide {kind}_jsonl or {kind}_src_txt/{kind}_tgt_txt.")
    return load_split("train"), load_split("valid"), load_split("test")


def bleu_and_sig(hyps: List[str], refs: List[str], tokenize: str="13a"):
    metric = sacrebleu.metrics.BLEU(tokenize=tokenize)
    score = metric.corpus_score([h.strip() for h in hyps], [[r.strip() for r in refs]])
    return float(score.score), score.format(signature=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["valid", "test"], default="test")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = cfg["train"]["output_dir"]
    os.makedirs(os.path.join(run_dir, "decode"), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base_model = cfg["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt), use_safetensors=True).to(device)
    model.eval()

    prefix = cfg["data"].get("prefix", "")
    max_src = int(cfg["data"].get("max_source_length", 128))
    dec_cfg = cfg.get("decode", {})
    max_new = int(dec_cfg.get("max_new_tokens", 128))
    bs = int(dec_cfg.get("batch_size", 32))
    num_beams = int(args.beam if args.beam is not None else dec_cfg.get("num_beams", 4))

    tok_bleu = cfg.get("eval", {}).get("sacrebleu_tokenize", "13a")

    _, valid_rows, test_rows = resolve_dataset(cfg)
    rows = valid_rows if args.split == "valid" else test_rows
    srcs = [prefix + r["zh"] for r in rows]
    refs = [r["en"] for r in rows]

    hyps: List[str] = []
    t0 = time.time()
    for i in range(0, len(srcs), bs):
        batch = srcs[i:i+bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_src).to(device)
        out = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new, do_sample=False)
        hyps.extend([x.strip() for x in tokenizer.batch_decode(out, skip_special_tokens=True)])
    dt = time.time() - t0

    bleu, sig = bleu_and_sig(hyps, refs, tokenize=tok_bleu)

    hyp_path = os.path.join(run_dir, "decode", f"{args.split}.beam{num_beams}.hyp.en.txt")
    with open(hyp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")

    out_json = os.path.join(run_dir, "decode", f"{args.split}.beam{num_beams}.bleu.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "overall": {"bleu": bleu, "signature": sig, "n": len(refs)},
            "decode_seconds": dt,
            "sent_per_sec": len(refs)/max(dt,1e-9),
            "gen_kwargs": {"num_beams": num_beams, "max_new_tokens": max_new}
        }, f, ensure_ascii=False, indent=2)

    print(f"{args.split} beam={num_beams} BLEU={bleu:.4f}")
    print(f"Signature: {sig}")
    print(hyp_path)
    print(out_json)


if __name__ == "__main__":
    main()
