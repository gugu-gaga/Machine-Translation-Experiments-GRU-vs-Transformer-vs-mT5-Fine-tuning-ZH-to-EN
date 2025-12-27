# main/train_ddp_mt5.py
import os
import re
import json
import math
import time
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from main.bleu import bleu_and_signature


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def setup_ddp():
    # torchrun sets these
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        # single process
        pass


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


def save_json(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]


def resolve_dataset(cfg: dict) -> Tuple[List[dict], List[dict], List[dict]]:
    d = cfg["data"]
    # Prefer JSONL if exists
    def exists(p: Optional[str]) -> bool:
        return bool(p) and os.path.exists(p)

    if exists(d.get("train_jsonl")) and exists(d.get("valid_jsonl")) and exists(d.get("test_jsonl")):
        train_rows = load_jsonl(d["train_jsonl"])
        valid_rows = load_jsonl(d["valid_jsonl"])
        test_rows  = load_jsonl(d["test_jsonl"])
        # Expect keys zh/en
        return train_rows, valid_rows, test_rows

    # Fallback parallel txt
    train_src = load_lines(d["train_src_txt"])
    train_tgt = load_lines(d["train_tgt_txt"])
    valid_src = load_lines(d["valid_src_txt"])
    valid_tgt = load_lines(d["valid_tgt_txt"])
    test_src  = load_lines(d["test_src_txt"])
    test_tgt  = load_lines(d["test_tgt_txt"])

    def pack(srcs, tgts):
        return [{"zh": s, "en": t} for s, t in zip(srcs, tgts)]

    return pack(train_src, train_tgt), pack(valid_src, valid_tgt), pack(test_src, test_tgt)


# -------------------------
# Dataset / Collate
# -------------------------
class MtDataset(Dataset):
    def __init__(self, rows: List[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


@dataclass
class Collator:
    tokenizer: any
    prefix: str
    max_src: int
    max_tgt: int

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        srcs = [self.prefix + x["zh"] for x in batch]
        tgts = [x["en"] for x in batch]

        enc = self.tokenizer(
            srcs,
            padding=True,
            truncation=True,
            max_length=self.max_src,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                tgts,
                padding=True,
                truncation=True,
                max_length=self.max_tgt,
                return_tensors="pt",
            )

        labels = dec["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        enc["labels"] = labels
        return enc


# -------------------------
# LR schedule (cosine + warmup)
# -------------------------
def build_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# -------------------------
# Eval (greedy, i.e., beam=1)
# -------------------------
@torch.no_grad()
def eval_greedy_bleu(
    model: nn.Module,
    tokenizer,
    rows: List[dict],
    prefix: str,
    max_src: int,
    max_new: int,
    batch_size: int,
    tokenize: str = "13a",
    device: torch.device = torch.device("cuda"),
) -> Tuple[float, str, float]:
    model.eval()

    srcs = [prefix + r["zh"] for r in rows]
    refs = [r["en"] for r in rows]

    hyps: List[str] = []
    t0 = time.time()
    for i in range(0, len(srcs), batch_size):
        batch = srcs[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_src,
        ).to(device)

        out = model.generate(
            **enc,
            num_beams=1,            # IMPORTANT: greedy baseline
            do_sample=False,
            max_new_tokens=max_new,
        )
        txt = tokenizer.batch_decode(out, skip_special_tokens=True)
        hyps.extend([x.strip() for x in txt])
    dt = time.time() - t0

    bleu, sig = bleu_and_signature(hyps, refs, tokenize=tokenize)
    return bleu, sig, dt


# -------------------------
# Checkpoint manager: keep top-k by BLEU
# -------------------------
class TopKManager:
    def __init__(self, out_dir: str, k: int):
        self.out_dir = Path(out_dir)
        self.k = k
        self.best: List[Tuple[float, str]] = []  # (bleu, ckpt_dir)

    def _ckpt_name(self, step: int, bleu: float) -> str:
        return f"checkpoint-step{step}-bleu{bleu:.4f}"

    def maybe_save(self, step: int, bleu: float, model_to_save: nn.Module, tokenizer, meta: dict):
        ckpt_dir = self.out_dir / self._ckpt_name(step, bleu)

        # If not in top-k, skip
        if len(self.best) >= self.k and bleu <= min(x[0] for x in self.best):
            return False, None

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model (safetensors)
        model_to_save.save_pretrained(str(ckpt_dir), safe_serialization=True)
        # Save tokenizer files once per ckpt (cheap, but makes ckpt standalone)
        tokenizer.save_pretrained(str(ckpt_dir))

        save_json(str(ckpt_dir / "eval_meta.json"), meta)

        self.best.append((bleu, str(ckpt_dir)))
        self.best.sort(key=lambda x: x[0], reverse=True)

        # Remove extras
        while len(self.best) > self.k:
            _, rm_dir = self.best.pop(-1)
            try:
                import shutil
                shutil.rmtree(rm_dir, ignore_errors=True)
            except Exception:
                pass

        # Write summary
        save_json(str(self.out_dir / "best_checkpoints.json"), {
            "topk": [{"bleu": b, "path": p} for b, p in self.best]
        })
        # Also write best pointer
        save_json(str(self.out_dir / "best_checkpoint.json"), {
            "best": {"bleu": self.best[0][0], "path": self.best[0][1]}
        })
        return True, str(ckpt_dir)


# -------------------------
# Main train
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # hard offline mode (avoid any hub call)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    setup_ddp()

    cfg = read_yaml(args.config)
    tr = cfg["train"]
    d = cfg["data"]
    m = cfg["model"]
    dec = cfg.get("decode", {})
    ev = cfg.get("eval", {})

    out_dir = tr["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    seed = int(tr.get("seed", 42))
    set_seed(seed + get_rank())

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load data
    train_rows, valid_rows, _ = resolve_dataset(cfg)

    # Load tokenizer/model from local offline dir ONLY
    base_dir = m["name_or_path"]  # /data/.../mt5_small_offline
    tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=False, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_dir, local_files_only=True, use_safetensors=True)
    model.to(device)

    # DDP wrap
    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    prefix = d.get("prefix", "")
    max_src = int(d["max_source_length"])
    max_tgt = int(d["max_target_length"])

    train_ds = MtDataset(train_rows)
    valid_ds = MtDataset(valid_rows)

    collate = Collator(tokenizer=tokenizer, prefix=prefix, max_src=max_src, max_tgt=max_tgt)

    bs = int(tr.get("batch_size_per_gpu", 4))
    grad_accum = int(tr.get("grad_accum", 8))

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )

    # Optim
    lr = float(tr.get("lr", 2e-4))
    wd = float(tr.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    max_steps = int(tr.get("max_steps", 20000))
    warmup_ratio = float(tr.get("warmup_ratio", 0.03))
    warmup_steps = int(max_steps * warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_lr_lambda(warmup_steps, max_steps)
    )

    bf16 = bool(tr.get("bf16", True))
    max_grad_norm = float(tr.get("max_grad_norm", 1.0))

    log_steps = int(tr.get("log_steps", 50))
    eval_steps = int(tr.get("eval_steps", 1000))
    save_topk = int(tr.get("save_topk", 3))

    tok_bleu = ev.get("sacrebleu_tokenize", "13a")
    eval_bs = int(dec.get("batch_size", 32))
    eval_max_new = int(dec.get("max_new_tokens", max_tgt))

    topk = TopKManager(out_dir=out_dir, k=save_topk)

    # Snapshot config (rank0)
    if rank0():
        save_json(os.path.join(out_dir, "config_snapshot.json"), cfg)

    model_train = model
    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    model_train.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0
    t_start = time.time()

    # We'll iterate epochs until max_steps
    epoch = 0
    while step < max_steps:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            step += 1
            if step > max_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                out = model_train(**batch)
                loss = out.loss / grad_accum

            loss.backward()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if rank0() and (step % log_steps == 0):
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                print(json.dumps({
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item() * grad_accum),
                    "lr": lr_now,
                    "seconds": elapsed
                }, ensure_ascii=False))

            # -------- Eval on rank0 only (greedy BLEU)
            if (step % eval_steps == 0) or (step == max_steps):
                barrier()
                if rank0():
                    bleu, sig, dt = eval_greedy_bleu(
                        model=model_eval,
                        tokenizer=tokenizer,
                        rows=valid_rows,
                        prefix=prefix,
                        max_src=max_src,
                        max_new=eval_max_new,
                        batch_size=eval_bs,
                        tokenize=tok_bleu,
                        device=device,
                    )
                    meta = {
                        "step": step,
                        "bleu": bleu,
                        "signature": sig,
                        "eval_seconds": dt,
                        "max_new_tokens": eval_max_new,
                        "note": "greedy eval (num_beams=1)",
                    }
                    print(json.dumps({"eval": meta}, ensure_ascii=False))

                    saved, ckpt_dir = topk.maybe_save(
                        step=step,
                        bleu=bleu,
                        model_to_save=model_eval,
                        tokenizer=tokenizer,
                        meta=meta,
                    )
                    if saved:
                        print(f"[rank0] saved checkpoint: {ckpt_dir}")
                barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()
