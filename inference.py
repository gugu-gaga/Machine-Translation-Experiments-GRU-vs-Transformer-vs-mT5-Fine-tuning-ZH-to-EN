#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified test-set inference + metrics for:
- GRU (./gru/runs/**/checkpoints/best.pt)
- Transformer (./transformer/runs/**/checkpoints/best.pt)
- mT5 (HF checkpoint dir)

Project-verified behavior:
- GRU/Transformer test decoding writes to <subproj>/configs/decode/test.beam{K}.*
  (fixed names; overwritten between runs). We therefore archive those outputs
  into a unified outdir immediately after each run.

This version exports plot-friendly tables:
- outdir/summary.jsonl    (one record per line; easy for pandas)
- outdir/summary.csv      (flat table; ready for Excel / plotting)
- outdir/summary.parquet  (fast analytics; optional if pyarrow installed)

Per (model,beam) directory also includes:
- hyp.txt, bleu.json (if produced by subproject), decode_meta.json
- bucket_bleu.json, fluency.json, examples.json

Test loss is disabled (per your decision).
"""

from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
import yaml
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Defaults (edit if needed)
# -----------------------------

DEFAULT_MODELS = [
    "./gru/runs/rnn_additive_tf05/checkpoints/best.pt",
    "./gru/runs/rnn_additive_tf1/checkpoints/best.pt",
    "./gru/runs/rnn_dot_tf05/checkpoints/best.pt",
    "./gru/runs/rnn_dot_tf1/checkpoints/best.pt",
    "./gru/runs/rnn_general_tf05/checkpoints/best.pt",
    "./gru/runs/rnn_general_tf1/checkpoints/best.pt",
    "./transformer/runs/tfm_t0_baseline/checkpoints/best.pt",
    "./transformer/runs/tfm_t1_pos_relative/checkpoints/best.pt",
    "./transformer/runs/tfm_t2_norm_rms/checkpoints/best.pt",
    "./transformer/runs/tfm_t3_big_batch/checkpoints/best.pt",
    "./transformer/runs/tfm_t4_big_lr/checkpoints/best.pt",
    "./transformer/runs/tfm_t5_small_scale/checkpoints/best.pt",
    "./transformer/runs/tfm_t5_small_scale_final/checkpoints/best.pt",
    "./mT5_finetune/runs/checkpoint-step99000-bleu12.0730",
]

# Run under each subproject cwd:
GRU_INFER_TEMPLATE = (
    "python -m main.infer_rnn --config {config} --ckpt {ckpt} --split {split} "
    "--beam {beam} --max_new_tokens {max_new_tokens}"
)
TFM_INFER_TEMPLATE = (
    "python -m main.infer_transformer --config {config} --ckpt {ckpt} --split {split} "
    "--beam {beam} --max_new_tokens {max_new_tokens}"
)


# -----------------------------
# Small utilities
# -----------------------------

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_lines(p: Path) -> List[str]:
    return p.read_text(encoding="utf-8").splitlines()

def _write_lines(p: Path, lines: Sequence[str]) -> None:
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def _json_dump(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def _resolve_path(repo_root: Path, subproj_root: Path, p: str) -> str:
    """Resolve a path string that may be:
    - absolute (keep)
    - subproject-relative like 'data/spm/...' (resolve under subproj_root)
    - repo-root-relative like 'gru/data/...' or 'transformer/data/...' (resolve under repo_root)
    - other relative (fallback: repo_root)
    """
    if not p:
        return p
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)

    s = p.replace('\\', '/')
    # Most configs use subproject-relative 'data/...'
    if s.startswith("data/") or s.startswith("./data/") or s.startswith("configs/") or s.startswith("./configs/"):
        return str((subproj_root / pp).resolve())

    # Some snapshots were saved with repo-root prefixes like 'gru/data/...'
    if s.startswith("gru/") or s.startswith("transformer/") or s.startswith("mT5_finetune/") or s.startswith("./gru/") or s.startswith("./transformer/") or s.startswith("./mT5_finetune/"):
        return str((repo_root / pp).resolve())

    # Default: interpret as repo-root-relative
    return str((repo_root / pp).resolve())


def _write_patched_yaml_config(
    orig_cfg: Path,
    patched_cfg: Path,
    *,
    repo_root: Path,
    subproj_root: Path,
) -> None:
    """Patch a config_snapshot.yaml so it works regardless of cwd.

    We only patch *paths*; hyperparams stay intact.

    Key idea:
    - paths like 'data/...' should stay subproject-relative (resolve under subproj_root)
    - paths like 'gru/data/...' should be resolved under repo_root
    """
    with orig_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def _patch_in_place(d: Dict[str, Any], key: str) -> None:
        v = d.get(key, None)
        if isinstance(v, str) and v:
            d[key] = _resolve_path(repo_root, subproj_root, v)

    # Common path keys
    if isinstance(cfg, dict):
        if isinstance(cfg.get("model", None), dict):
            _patch_in_place(cfg["model"], "emb_init_npy")

        if isinstance(cfg.get("data", None), dict):
            for k in ["spm_model", "spm_path", "sentencepiece_model"]:
                _patch_in_place(cfg["data"], k)

        # Sometimes stored at top-level
        for k in ["spm_model", "spm_path"]:
            _patch_in_place(cfg, k)

    patched_cfg.parent.mkdir(parents=True, exist_ok=True)
    with patched_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)



def _split_tokens(s: str) -> List[str]:
    return s.strip().split()

def repetition_rate(lines: Sequence[str]) -> float:
    rep = 0
    total = 0
    for ln in lines:
        toks = _split_tokens(ln)
        total += 1
        if any(toks[i] == toks[i+1] for i in range(len(toks)-1)):
            rep += 1
    return (rep / total) if total else 0.0

def distinct_n(lines: Sequence[str], n: int) -> float:
    ngrams = set()
    total = 0
    for ln in lines:
        toks = _split_tokens(ln)
        for i in range(len(toks) - n + 1):
            ngrams.add(tuple(toks[i:i+n]))
            total += 1
    return (len(ngrams) / total) if total else 0.0

def length_ratio(hyps: Sequence[str], refs: Sequence[str]) -> float:
    def avg_len(xs: Sequence[str]) -> float:
        if not xs:
            return 0.0
        return sum(len(_split_tokens(x)) for x in xs) / len(xs)
    r = avg_len(refs)
    h = avg_len(hyps)
    return (h / r) if r > 0 else 0.0

def bucket_name(i: int, edges: Sequence[int]) -> str:
    e1, e2, e3 = edges
    return [
        f"short(<= {e1})",
        f"medium({e1+1}-{e2})",
        f"long({e2+1}-{e3})",
        f"xlong(> {e3})",
    ][i]

def bucketize(lengths: Sequence[int], edges: Sequence[int]) -> List[int]:
    e1, e2, e3 = edges
    out = []
    for x in lengths:
        if x <= e1:
            out.append(0)
        elif x <= e2:
            out.append(1)
        elif x <= e3:
            out.append(2)
        else:
            out.append(3)
    return out


def mt5_source_token_lengths(
    srcs: List[str],
    *,
    ckpt_dir: Path,
    prefix: str,
    max_length: int = 256,
) -> List[int]:
    """Compute source lengths for bucketing using the mT5 checkpoint's own tokenizer.
    We mirror the encode-time truncation used in decode_mt5() (truncation=True, max_length=max_length).
    Lengths are measured in tokenizer subword tokens (including special tokens as returned by the tokenizer).
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    lens: List[int] = []
    # 200 lines only; per-sentence tokenization is fine and avoids padding effects.
    for s in srcs:
        ids = tok(prefix + s, truncation=True, max_length=int(max_length)).get("input_ids", [])
        lens.append(len(ids))
    return lens


def load_ids_lengths(ids_path: Path) -> List[int]:
    lens = []
    for ln in _read_lines(ids_path):
        ln = ln.strip()
        if not ln:
            continue
        lens.append(len(ln.split()))
    return lens

def try_import_sacrebleu():
    try:
        import sacrebleu  # type: ignore
        return sacrebleu
    except Exception as e:
        raise RuntimeError("sacrebleu is required (pip install sacrebleu).") from e

def corpus_bleu(hyps: Sequence[str], refs: Sequence[str], tokenize: str = "13a") -> Dict[str, Any]:
    sacrebleu = try_import_sacrebleu()
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)

    # sacrebleu API varies across versions:
    # - some versions expose .signature (str)
    # - some expose .signature() (callable)
    # - some expose .get_signature() (callable)
    sig = None
    if hasattr(bleu, "signature"):
        s = getattr(bleu, "signature")
        sig = s() if callable(s) else s
    elif hasattr(bleu, "get_signature"):
        sig = bleu.get_signature()
    return {"bleu": float(getattr(bleu, "score", bleu)), "signature": (str(sig) if sig is not None else None)}

def compute_bucket_bleu(
    hyps: Sequence[str],
    refs: Sequence[str],
    buckets: Sequence[int],
    edges: Sequence[int],
    tokenize: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"edges": list(edges), "buckets": {}}
    for b in range(4):
        idxs = [i for i, bi in enumerate(buckets) if bi == b]
        sub_h = [hyps[i] for i in idxs]
        sub_r = [refs[i] for i in idxs]
        name = bucket_name(b, edges)
        if sub_h and sub_r:
            out["buckets"][name] = {"n": len(sub_h), **corpus_bleu(sub_h, sub_r, tokenize=tokenize)}
        else:
            out["buckets"][name] = {"n": 0, "bleu": None, "signature": None}
    out["overall"] = {"n": len(hyps), **corpus_bleu(hyps, refs, tokenize=tokenize)}
    return out

def compute_fluency_stats(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, Any]:
    return {
        "repetition_rate": repetition_rate(hyps),
        "distinct_1": distinct_n(hyps, 1),
        "distinct_2": distinct_n(hyps, 2),
        "length_ratio": length_ratio(hyps, refs),
    }


# -----------------------------
# Model specs
# -----------------------------

@dataclass
class ModelSpec:
    name: str
    type: str  # "gru" | "transformer" | "mt5"
    ckpt: Path
    run_dir: Path
    config_path: Optional[Path] = None

@dataclass
class DecodeResult:
    hyps: List[str]
    seconds: float
    tokens: int
    notes: Dict[str, Any]

def infer_model_type(p: Path) -> str:
    s = str(p).lower()
    if "mt5" in s or (p.is_dir() and (p / "model.safetensors").exists()):
        return "mt5"
    if "/gru/" in s or s.startswith("gru/") or "/rnn_" in s:
        return "gru"
    if "/transformer/" in s or s.startswith("transformer/") or "/tfm_" in s:
        return "transformer"
    if p.suffix == ".pt":
        return "transformer"
    return "mt5"

def make_model_spec(path_str: str) -> ModelSpec:
    ckpt = Path(path_str).expanduser()
    mtype = infer_model_type(ckpt)

    if ckpt.is_file() and ckpt.name.endswith(".pt") and ckpt.parent.name == "checkpoints":
        run_dir = ckpt.parent.parent
    else:
        run_dir = ckpt if ckpt.is_dir() else ckpt.parent

    name = run_dir.name if run_dir.name else ckpt.stem

    # Prefer the canonical config in subproject configs/ when available.
    # This avoids path quirks in some config_snapshot files while keeping decode behavior consistent.
    cfg: Optional[Path] = None
    subproj_root: Optional[Path] = None
    if mtype in ("gru", "transformer"):
        # run_dir looks like <repo>/<subproj>/runs/<name>
        try:
            subproj_root = run_dir.parent.parent
        except Exception:
            subproj_root = None

    if mtype == "gru" and subproj_root is not None:
        cand = subproj_root / "configs" / f"{name}.yaml"
        if cand.exists():
            cfg = cand
    elif mtype == "transformer" and subproj_root is not None:
        # transformer run names are like 'tfm_t0_baseline' -> config 't0_baseline.yaml'
        short = name[4:] if name.startswith("tfm_") else name
        cand = subproj_root / "configs" / f"{short}.yaml"
        if cand.exists():
            cfg = cand

    # Fallback to snapshot configs if canonical one is not found.
    if cfg is None:
        if (run_dir / "config_snapshot.yaml").exists():
            cfg = run_dir / "config_snapshot.yaml"
        elif (run_dir / "config_snapshot.json").exists():
            cfg = run_dir / "config_snapshot.json"

    return ModelSpec(name=name, type=mtype, ckpt=ckpt, run_dir=run_dir, config_path=cfg)


# -----------------------------
# Subprocess decode (GRU/TFM)
# -----------------------------

def run_subprocess_decode(
    cmd: str,
    log_path: Path,
    *,
    cwd: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    _mkdir(log_path.parent)
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v is not None})
    run_cwd = str(cwd) if cwd is not None else None

    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"[cmd] {cmd}\n")
        if run_cwd:
            lf.write(f"[cwd] {run_cwd}\n")
        lf.flush()

        p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=lf, env=env, cwd=run_cwd)
        ret = p.wait()
        lf.write(f"\n[exit_code] {ret}\n")

    if ret != 0:
        raise RuntimeError(f"Decode subprocess failed (exit={ret}). See log: {log_path}")

def decode_gru_or_transformer(
    spec: ModelSpec,
    split: str,
    beam: int,
    out_hyp: Path,
    max_new_tokens: int,
) -> DecodeResult:
    out_hyp.parent.mkdir(parents=True, exist_ok=True)

    proj_root = Path(__file__).resolve().parent
    if spec.type == "gru":
        subproj = proj_root / "gru"
        cmd_template = GRU_INFER_TEMPLATE
    elif spec.type == "transformer":
        subproj = proj_root / "transformer"
        cmd_template = TFM_INFER_TEMPLATE
    else:
        raise ValueError(f"decode_gru_or_transformer called with type={spec.type}")

    if spec.config_path is None or not spec.config_path.exists():
        raise FileNotFoundError(f"Missing config_snapshot for {spec.name}: {spec.run_dir}/config_snapshot.*")

    # IMPORTANT: subproject infer scripts run with cwd=subproj and configs may contain repo-root-relative
    # paths like "gru/data/...". Patch such paths to absolute to avoid cwd-related breakage.
    repo_root = Path(__file__).resolve().parent
    patched_cfg = out_hyp.parent / "config_patched.yaml"
    _write_patched_yaml_config(spec.config_path, patched_cfg, repo_root=repo_root, subproj_root=subproj)

    # Use config path relative to subproj if possible (otherwise absolute).
    try:
        config_arg = str(patched_cfg.resolve().relative_to(subproj.resolve()))
    except Exception:
        config_arg = str(patched_cfg.resolve())

    cmd = cmd_template.format(
        config=config_arg,
        ckpt=str(Path(spec.ckpt).resolve()),
        beam=int(beam),
        split=str(split),
        max_new_tokens=int(max_new_tokens),
    )

    # The subproject writes decode outputs next to the config file: <dir_of_config>/decode.
    # This is how the student-provided infer scripts behave (they set run_dir=os.path.dirname(config)).
    decode_dir = patched_cfg.parent / "decode"

    # Scheme C: isolate each run by clearing the decode directory *before* decoding.
    # This guarantees any files we read afterwards were produced by this run.
    decode_dir.mkdir(parents=True, exist_ok=True)
    for child in decode_dir.glob("*"):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception:
            pass

    log_path = out_hyp.parent / "decode.log"
    t0 = time.time()
    run_subprocess_decode(cmd, log_path, cwd=subproj)
    t1 = time.time()

    hyp_src = decode_dir / f"{split}.beam{beam}.hyp.en.txt"
    bleu_src = decode_dir / f"{split}.beam{beam}.bleu.json"

    if not hyp_src.exists():
        listing = "\n".join([p.name for p in sorted(decode_dir.glob("*"))][:200])
        raise FileNotFoundError(
            f"Expected hyp file not found: {hyp_src}\n"
            f"Files in decode_dir (first 200):\n{listing}"
        )

    out_hyp.write_text(hyp_src.read_text(encoding="utf-8"), encoding="utf-8")
    if bleu_src.exists():
        (out_hyp.parent / "bleu.json").write_text(bleu_src.read_text(encoding="utf-8"), encoding="utf-8")

    hyps = _read_lines(out_hyp)
    tok = sum(len(_split_tokens(h)) for h in hyps)
    return DecodeResult(
        hyps=hyps,
        seconds=float(t1 - t0),
        tokens=int(tok),
        notes={
            "cmd": cmd,
            "cwd": str(subproj),
            "decode_dir": str(decode_dir),
            "hyp_src": str(hyp_src),
            "bleu_src": str(bleu_src) if bleu_src.exists() else None,
            "beam_requested": int(beam),
            "beam_actual": (int(re.search(r"\.beam(\d+)\.", hyp_src.name).group(1)) if re.search(r"\.beam(\d+)\.", hyp_src.name) else None),
        },
    )


# -----------------------------
# mT5 decode (in-process)
# -----------------------------

def batch_iter(xs: Sequence[str], bs: int) -> Sequence[List[str]]:
    for i in range(0, len(xs), bs):
        yield list(xs[i : i + bs])

def decode_mt5(
    spec: ModelSpec,
    srcs: List[str],
    *,
    beam: int,
    max_new_tokens: int,
    batch_size: int,
    prefix: str,
    device: str,
    bf16: bool,
) -> DecodeResult:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ckpt_dir = spec.ckpt if spec.ckpt.is_dir() else spec.run_dir
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(ckpt_dir),
        torch_dtype=(torch.bfloat16 if bf16 else None),
    ).to(device)
    model.eval()

    hyps: List[str] = []
    total_out_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in batch_iter([prefix + s for s in srcs], batch_size):
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            gen = model.generate(**enc, num_beams=int(beam), max_new_tokens=int(max_new_tokens))
            outs = tok.batch_decode(gen, skip_special_tokens=True)
            hyps.extend([o.strip() for o in outs])
            total_out_tokens += int(gen.numel())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()

    return DecodeResult(hyps=hyps, seconds=float(t1 - t0), tokens=int(total_out_tokens), notes={"ckpt_dir": str(ckpt_dir)})


# -----------------------------
# Export helpers (plot-friendly)
# -----------------------------

def write_summary_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    _mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def try_write_parquet(path: Path, rows: List[Dict[str, Any]]) -> Optional[str]:
    # Optional: only if pandas+pyarrow available
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return None
    except Exception as e:
        return str(e)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_list", type=str, default="", help="Text file with checkpoint paths (one per line).")
    ap.add_argument("--outdir", type=str, default=f"test_eval_{_now_tag()}", help="Output directory.")
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 5], help="Beam sizes to evaluate.")
    ap.add_argument("--max_examples", type=int, default=200, help="How many examples to dump for qualitative report.")
    ap.add_argument("--length_edges", type=int, nargs=3, default=[32, 64, 128], help="Source-length bucket edges.")
    ap.add_argument("--sacrebleu_tokenize", type=str, default="13a", help="sacrebleu tokenization (e.g., 13a).")
    ap.add_argument("--split", type=str, default="test", help="Split name (GRU/Transformer).")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens (GRU/Transformer).")

    ap.add_argument("--batch_size_decode", type=int, default=32, help="Decode batch size (mT5 only).")
    ap.add_argument("--mt5_prefix", type=str, default="translate Chinese to English: ", help="mT5 text prefix.")
    ap.add_argument("--mt5_max_new_tokens", type=int, default=128, help="mT5 max_new_tokens in generate().")
    ap.add_argument("--mt5_bf16", action="store_true", help="Use bf16 for mT5 eval (if supported).")
    ap.add_argument("--device", type=str, default="cuda", help="Device for mT5 (cuda or cpu).")

    ap.add_argument("--gru_test_jsonl", type=str, default="./gru/data/raw/test.jsonl")
    ap.add_argument("--src_key", type=str, default="zh", help="JSONL field name for source text (default: zh).")
    ap.add_argument("--tgt_key", type=str, default="en", help="JSONL field name for target/reference text (default: en).")
    ap.add_argument("--tfm_test_jsonl", type=str, default="./transformer/data/raw/test.jsonl")
    ap.add_argument("--mt5_test_jsonl", type=str, default="./mT5_finetune/data/raw/test.jsonl")
    ap.add_argument("--gru_src_ids", type=str, default="./gru/data/clean/test.src.ids")
    ap.add_argument("--tfm_src_ids", type=str, default="./transformer/data/clean/test.src.ids")

    args = ap.parse_args()

    out_root = Path(args.outdir)
    _mkdir(out_root)

    models: List[str]
    if args.models_list:
        models = [ln.strip() for ln in _read_lines(Path(args.models_list)) if ln.strip() and not ln.strip().startswith("#")]
    else:
        models = list(DEFAULT_MODELS)

    specs = [make_model_spec(m) for m in models]
    summary_rows: List[Dict[str, Any]] = []

    for spec in specs:
        # ckpt tag for folder naming
        if spec.type == "mt5":
            ckpt_tag = spec.ckpt.name if spec.ckpt.is_dir() else spec.ckpt.stem
            if spec.ckpt.is_dir():
                # for your case: checkpoint-step99000-bleu12.0730
                ckpt_tag = spec.ckpt.name
            else:
                ckpt_tag = "model"
        else:
            ckpt_tag = spec.ckpt.stem  # best

        model_out = out_root / "models" / spec.type / spec.name / ckpt_tag
        _mkdir(model_out)

        # Load test data and length buckets
        if spec.type == "gru":
            test_jsonl = Path(args.gru_test_jsonl)
            src_ids = Path(args.gru_src_ids)
            src_lens = load_ids_lengths(src_ids)
        elif spec.type == "transformer":
            test_jsonl = Path(args.tfm_test_jsonl)
            src_ids = Path(args.tfm_src_ids)
            src_lens = load_ids_lengths(src_ids)
        else:
            test_jsonl = Path(args.mt5_test_jsonl)
            # mT5 bucket lengths: use the checkpoint tokenizer (subword length) for fair within-mT5 bucketing
            src_lens = None

        raw = [json.loads(ln) for ln in _read_lines(test_jsonl)]
        # Project test.jsonl uses {"zh": ..., "en": ...} (confirmed).
        # Keep flexible via --src_key/--tgt_key.
        try:
            srcs = [r[args.src_key] for r in raw]
            refs = [r[args.tgt_key] for r in raw]

            # Length bucketing:
            # - GRU/Transformer: use pre-tokenized .ids lengths
            # - mT5: use the checkpoint tokenizer subword lengths on source (with prefix), mirroring decode_mt5() truncation
            if spec.type == "mt5":
                ckpt_dir = spec.ckpt if spec.ckpt.is_dir() else spec.run_dir
                src_lens = mt5_source_token_lengths(
                    srcs,
                    ckpt_dir=ckpt_dir,
                    prefix=str(args.mt5_prefix),
                    max_length=256,
                )
        except KeyError as e:
            keys0 = list(raw[0].keys()) if raw else []
            raise KeyError(f"Missing expected key {e} in test.jsonl. First record keys={keys0}. Use --src_key/--tgt_key.")

        if src_lens is None:
            src_lens = [len(s.split()) for s in srcs]
        buckets = bucketize(src_lens, args.length_edges)

        for beam in args.beams:
            print(f"[run] {spec.type}/{spec.name} beam={beam}", flush=True)
            beam_out = model_out / f"beam{beam}"
            _mkdir(beam_out)
            out_hyp = beam_out / "hyp.txt"

            if spec.type == "mt5":
                dec = decode_mt5(
                    spec,
                    srcs,
                    beam=int(beam),
                    max_new_tokens=int(args.mt5_max_new_tokens),
                    batch_size=int(args.batch_size_decode),
                    prefix=str(args.mt5_prefix),
                    device=str(args.device),
                    bf16=bool(args.mt5_bf16),
                )
                _write_lines(out_hyp, dec.hyps)
                _json_dump(beam_out / "decode_meta.json", {"seconds": dec.seconds, "tokens": dec.tokens, **dec.notes})
                hyps = dec.hyps
            else:
                dec = decode_gru_or_transformer(
                    spec,
                    split=str(args.split),
                    beam=int(beam),
                    out_hyp=out_hyp,
                    max_new_tokens=int(args.max_new_tokens),
                )
                _json_dump(beam_out / "decode_meta.json", {"seconds": dec.seconds, "tokens": dec.tokens, **dec.notes})
                hyps = dec.hyps

            # Align lengths safely
            n = min(len(hyps), len(refs), len(srcs), len(buckets))
            hyps = hyps[:n]
            refs_eval = refs[:n]
            srcs_eval = srcs[:n]
            buckets_eval = buckets[:n]

            bucket_bleu = compute_bucket_bleu(hyps, refs_eval, buckets_eval, args.length_edges, args.sacrebleu_tokenize)
            _json_dump(beam_out / "bucket_bleu.json", bucket_bleu)

            flu = compute_fluency_stats(hyps, refs_eval)
            _json_dump(beam_out / "fluency.json", flu)

            # examples.json (flat array; easy for plotting / filtering)
            k = min(int(args.max_examples), n)
            examples = []
            for i in range(k):
                examples.append({
                    "i": i,
                    "bucket": bucket_name(buckets_eval[i], args.length_edges),
                    "src": srcs_eval[i],
                    "ref": refs_eval[i],
                    "hyp": hyps[i],
                })
            _json_dump(beam_out / "examples.json", examples)

            # Flatten bucket BLEU for summary rows (plot-friendly columns)
            b = bucket_bleu["buckets"]
            row = {
                "model_type": spec.type,
                "model_name": spec.name,
                "ckpt": str(spec.ckpt),
                "beam": int(beam),
                "decode_seconds": float(dec.seconds),
                "tokens": int(dec.tokens),
                "sent_per_sec": (n / float(dec.seconds)) if dec.seconds > 0 else None,
                "bleu_overall": bucket_bleu["overall"]["bleu"],
                "bleu_short": b[bucket_name(0, args.length_edges)]["bleu"],
                "bleu_medium": b[bucket_name(1, args.length_edges)]["bleu"],
                "bleu_long": b[bucket_name(2, args.length_edges)]["bleu"],
                "bleu_xlong": b[bucket_name(3, args.length_edges)]["bleu"],
                "n_short": b[bucket_name(0, args.length_edges)]["n"],
                "n_medium": b[bucket_name(1, args.length_edges)]["n"],
                "n_long": b[bucket_name(2, args.length_edges)]["n"],
                "n_xlong": b[bucket_name(3, args.length_edges)]["n"],
                "repetition_rate": flu["repetition_rate"],
                "distinct_1": flu["distinct_1"],
                "distinct_2": flu["distinct_2"],
                "length_ratio": flu["length_ratio"],
                "length_edges": ",".join(map(str, args.length_edges)),
            }
            summary_rows.append(row)

    # Plot-friendly exports
    write_summary_jsonl(out_root / "summary.jsonl", summary_rows)
    write_summary_csv(out_root / "summary.csv", summary_rows)
    parquet_err = try_write_parquet(out_root / "summary.parquet", summary_rows)
    if parquet_err:
        # Don't fail the run if parquet isn't available.
        _json_dump(out_root / "summary_parquet_error.json", {"error": parquet_err})

    print(f"[done] wrote results to: {out_root}")
    print(f" - summary.jsonl (pandas read_json(lines=True))")
    print(f" - summary.csv   (Excel-friendly)")
    print(f" - summary.parquet (if available)")

if __name__ == "__main__":
    main()
