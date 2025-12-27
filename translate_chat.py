#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Chinese->English translator for:
- mT5 HF checkpoints (load in-process via transformers, cached)
- GRU/Transformer best.pt checkpoints (call inference_v8.py as subprocess for 1 sentence)

Exit: type 'q' + Enter
Go back to model menu: type 'm' + Enter
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_PREFIX = "translate Chinese to English: "


@dataclass
class ModelEntry:
    name: str
    path: str
    kind: str  # "mt5_hf" or "pt_subproj"


def _looks_like_hf_seq2seq_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    names = {x.name for x in p.iterdir()}
    return ("config.json" in names) and ("model.safetensors" in names or "pytorch_model.bin" in names)


def _abs_from_cwd(path_str: str) -> str:
    # preserve relative paths but normalize
    return str(Path(path_str).expanduser())


def read_models_tsv(path: Path) -> List[ModelEntry]:
    if not path.exists():
        raise FileNotFoundError(f"models list not found: {path}")

    entries: List[ModelEntry] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    auto_i = 1
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue

        if "\t" in ln:
            name, p = ln.split("\t", 1)
            name, p = name.strip(), p.strip()
        else:
            name, p = f"model{auto_i}", ln
            auto_i += 1

        p = _abs_from_cwd(p)
        pp = Path(p)

        if _looks_like_hf_seq2seq_dir(pp):
            kind = "mt5_hf"
        else:
            kind = "pt_subproj"
        entries.append(ModelEntry(name=name, path=p, kind=kind))

    return entries


@dataclass
class LoadedHF:
    tokenizer: object
    model: object
    device: str


def load_hf_model(ckpt_dir: str, device: str, bf16: bool) -> LoadedHF:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    dtype = torch.bfloat16 if bf16 else None
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return LoadedHF(tokenizer=tok, model=model, device=device)


def translate_hf(lm: LoadedHF, text_zh: str, beam: int, max_new_tokens: int, prefix: str) -> str:
    import torch

    inp = (prefix or "") + text_zh.strip()
    enc = lm.tokenizer([inp], return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(lm.device) for k, v in enc.items()}
    with torch.no_grad():
        gen = lm.model.generate(**enc, num_beams=int(beam), max_new_tokens=int(max_new_tokens))
    out = lm.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return out


def translate_via_inference_v8(
    inference_py: str,
    model_path: str,
    text_zh: str,
    beam: int,
    device: str,
    prefix: str,
) -> str:
    """
    Works for both:
      - best.pt (gru/transformer) through inference_v8's own pipeline
      - HF mt5 too, but we only use it for pt models; HF is faster in-process
    """
    # Make a temp jsonl with one example. inference expects {"en","zh","index"} in many projects.
    # We'll store zh, keep en empty.
    tmpdir = Path(tempfile.mkdtemp(prefix="one_sent_"))
    data = tmpdir / "one.jsonl"
    data.write_text(json.dumps({"en": "", "zh": text_zh, "index": 0}, ensure_ascii=False) + "\n", encoding="utf-8")

    # models_list file with one model path
    mlist = tmpdir / "one_model.txt"
    mlist.write_text(model_path + "\n", encoding="utf-8")

    outdir = tmpdir / "out"

    # Call inference_v8.py:
    # We rely on inference_v8's existing logic to route model type (gru/transformer/mt5).
    # Important: use beams 1 entry; max_examples 1.
    cmd = [
        sys.executable,
        inference_py,
        "--models_list",
        str(mlist),
        "--outdir",
        str(outdir),
        "--beams",
        str(beam),
        "--max_examples",
        "1",
    ]

    # If your inference_v8 has device option you can add it here; otherwise it will use its defaults.
    env = os.environ.copy()
    # Optional: set visible device
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    if proc.returncode != 0:
        raise RuntimeError(
            "inference_v8 failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    # Find the hyp file. Different pipelines may name it differently; we try a few.
    # First, search for any *.hyp*.txt or *hyp*.txt under outdir.
    cand = []
    if outdir.exists():
        for p in outdir.rglob("*"):
            if p.is_file():
                n = p.name.lower()
                if ("hyp" in n) and n.endswith(".txt"):
                    cand.append(p)
                elif n in ("hyp.txt", "hyps.txt", "test.hyp.txt"):
                    cand.append(p)

    # If multiple, take the newest.
    if not cand:
        # As a fallback, search any .txt that looks like outputs
        for p in outdir.rglob("*.txt"):
            cand.append(p)

    if not cand:
        raise FileNotFoundError(f"Could not find hyp output under {outdir}. See stdout/stderr above.")

    hyp_path = max(cand, key=lambda p: p.stat().st_mtime)
    lines = hyp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return ""
    return lines[0].strip()


def print_menu(models: List[ModelEntry]) -> None:
    print("\n=== Model Menu (enter number) ===")
    for i, m in enumerate(models, start=1):
        kind = "mT5" if m.kind == "mt5_hf" else "gru/transformer"
        print(f"{i:2d}) {m.name:<14} [{kind}]  {m.path}")
    print("================================\n")
    print('Tips: enter "q" to quit, enter "m" to show menu again.\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="models.txt", help="TSV: name<TAB>path (14 lines).")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    ap.add_argument("--bf16", action="store_true", help="HF mT5: load weights in bf16 (cuda recommended)")
    ap.add_argument("--inference_py", type=str, default="inference_v8.py", help="Path to inference_v8.py (for pt models).")
    args = ap.parse_args()

    models = read_models_tsv(Path(args.models))
    if len(models) == 0:
        print("[error] no models found in models file.")
        sys.exit(1)

    # Cache for HF models only (pt models are run via subprocess)
    hf_cache: Dict[str, LoadedHF] = {}

    device = args.device
    beam = int(args.beam)
    max_new_tokens = int(args.max_new_tokens)
    prefix = args.prefix
    inference_py = args.inference_py

    print_menu(models)

    current_idx: Optional[int] = None

    while True:
        if current_idx is None:
            sel = input("Select model (1-{}), or q: ".format(len(models))).strip()
            if sel.lower() == "q":
                print("[exit]")
                return
            if sel.lower() in ("m", "menu"):
                print_menu(models)
                continue
            if not sel.isdigit() or not (1 <= int(sel) <= len(models)):
                print("Please enter a number in 1..{} (or q).".format(len(models)))
                continue
            current_idx = int(sel)
            chosen = models[current_idx - 1]
            print(f"[model] {current_idx}) {chosen.name}  ({chosen.path})")
            if chosen.kind == "mt5_hf" and chosen.path not in hf_cache:
                try:
                    print(f"[load] loading mT5 (HF) on {device} ...")
                    hf_cache[chosen.path] = load_hf_model(chosen.path, device=device, bf16=bool(args.bf16))
                    print("[load] done.")
                except Exception as e:
                    print(f"[error] failed to load HF model: {type(e).__name__}: {e}")
                    current_idx = None
            continue

        chosen = models[current_idx - 1]
        prompt = f'[{chosen.name}] 输入中文（q 退出 / m 换模型）: '
        s = input(prompt).strip()
        if not s:
            continue
        if s.lower() == "q":
            print("[exit]")
            return
        if s.lower() in ("m", "menu"):
            current_idx = None
            print_menu(models)
            continue

        try:
            if chosen.kind == "mt5_hf":
                out = translate_hf(hf_cache[chosen.path], s, beam=beam, max_new_tokens=max_new_tokens, prefix=prefix)
            else:
                out = translate_via_inference_v8(
                    inference_py=inference_py,
                    model_path=chosen.path,
                    text_zh=s,
                    beam=beam,
                    device=device,
                    prefix=prefix,
                )
            print(out)
        except Exception as e:
            print(f"[error] translate failed: {type(e).__name__}: {e}")
            print('Tip: type "m" to switch model, or "q" to quit.')


if __name__ == "__main__":
    main()
