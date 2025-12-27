import os, json
from typing import Any, Dict

class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
    def log(self, obj: Dict[str, Any]):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass
