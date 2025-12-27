import os
from typing import List
from .common import ensure_dir

def load_ids(path: str) -> List[List[int]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            out.append([int(x) for x in line.split()] if line else [])
    return out

def load_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]

def save_lines(path: str, lines: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")
