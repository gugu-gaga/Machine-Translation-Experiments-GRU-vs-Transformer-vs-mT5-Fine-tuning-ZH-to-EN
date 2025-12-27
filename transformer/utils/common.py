import os, json, time, random
from typing import Any, Dict
import numpy as np

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

class JsonlLogger:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.f = open(path, "a", encoding="utf-8")
    def log(self, obj: Dict[str, Any]):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass

def save_json(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
