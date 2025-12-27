#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect model file size + parameter count for GRU/Transformer (.pt) and mT5 (.safetensors).

Usage (from repo root /data/250010012):
  python model_stats.py --root . --out model_stats.jsonl

Auto-discovers:
  - gru/runs/*/checkpoints/best.pt
  - transformer/runs/*/checkpoints/best.pt
  - mT5_finetune/runs/checkpoint-*/model.safetensors  (or use --mt5_dir)

Notes:
  - .pt: loads state_dict on CPU and counts numel
  - .safetensors: reads tensor shapes/dtypes from header via safe_open.get_tensor_info (no full tensor materialization)
"""
import argparse, json
from pathlib import Path

def bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)

def count_pt_params(pt_path: Path):
    import torch
    obj = torch.load(str(pt_path), map_location="cpu")
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        else:
            # maybe already state_dict
            tensor_vals = [v for v in obj.values() if hasattr(v, "numel")]
            sd = obj if len(tensor_vals) > 0 else None
    else:
        sd = None
    if sd is None:
        raise ValueError(f"Unrecognized checkpoint format: {pt_path}")

    total = 0
    dtype_hist = {}
    for _, v in sd.items():
        if hasattr(v, "numel"):
            n = int(v.numel())
            total += n
            dt = str(v.dtype)
            dtype_hist[dt] = dtype_hist.get(dt, 0) + n
    return total, dtype_hist

def count_safetensors_params(st_path: Path):
    from safetensors import safe_open
    total = 0
    dtype_hist = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        if not hasattr(f, "get_tensor_info"):
            raise RuntimeError("Your safetensors version lacks get_tensor_info(); upgrade safetensors or fallback to loading tensors.")
        for name in f.keys():
            info = f.get_tensor_info(name)  # metadata-only
            shape = info.shape
            n = 1
            for d in shape:
                n *= int(d)
            total += n
            dt = str(info.dtype)
            dtype_hist[dt] = dtype_hist.get(dt, 0) + n
    return total, dtype_hist

def discover(root: Path, mt5_dir: Path | None):
    items = []
    for p in sorted((root / "gru" / "runs").glob("*/checkpoints/best.pt")):
        items.append({"model_type": "gru", "model_name": p.parent.parent.name, "ckpt": str(p)})
    for p in sorted((root / "transformer" / "runs").glob("*/checkpoints/best.pt")):
        items.append({"model_type": "transformer", "model_name": p.parent.parent.name, "ckpt": str(p)})
    if mt5_dir is not None:
        st = mt5_dir / "model.safetensors"
        if st.exists():
            items.append({"model_type": "mt5", "model_name": mt5_dir.name, "ckpt": str(mt5_dir), "safetensors": str(st)})
    else:
        for st in sorted((root / "mT5_finetune" / "runs").glob("checkpoint-*/model.safetensors")):
            items.append({"model_type": "mt5", "model_name": st.parent.name, "ckpt": str(st.parent), "safetensors": str(st)})
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="repo root (e.g., /data/250010012)")
    ap.add_argument("--out", type=str, default="model_stats.jsonl")
    ap.add_argument("--mt5_dir", type=str, default=None, help="specific mt5 checkpoint dir (contains model.safetensors)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    mt5_dir = Path(args.mt5_dir).resolve() if args.mt5_dir else None

    rows = []
    for it in discover(root, mt5_dir):
        file_path = Path(it["ckpt"]).resolve()
        if it["model_type"] == "mt5":
            file_path = Path(it["safetensors"]).resolve()

        row = dict(it)
        row["ckpt_file"] = str(file_path)
        size_bytes = file_path.stat().st_size
        row["ckpt_bytes"] = int(size_bytes)
        row["ckpt_mb"] = round(bytes_to_mb(size_bytes), 3)

        try:
            if file_path.suffix == ".pt":
                n_params, dtype_hist = count_pt_params(file_path)
            elif file_path.suffix == ".safetensors":
                n_params, dtype_hist = count_safetensors_params(file_path)
            else:
                raise ValueError(f"Unsupported checkpoint type: {file_path.suffix}")
            row["param_count"] = int(n_params)
            row["param_m"] = round(n_params / 1e6, 3)
            row["param_dtypes"] = dtype_hist
        except Exception as e:
            row["param_count"] = None
            row["param_m"] = None
            row["param_dtypes"] = None
            row["error"] = f"{type(e).__name__}: {e}"

        rows.append(row)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {outp}")

if __name__ == "__main__":
    main()
