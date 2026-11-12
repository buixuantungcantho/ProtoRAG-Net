from __future__ import annotations
import hashlib, json, os, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
import yaml

def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_update(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def sha256_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

def capture_environment(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(info, out_dir/"environment.json")
    try:
        freeze = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
        (out_dir/"pip_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as e:
        (out_dir/"pip_freeze_error.txt").write_text(str(e), encoding="utf-8")

def code_manifest(code_root: Path, out_file: Path):
    rows = []
    for p in sorted(code_root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rows.append(f"{sha256_file(p)}  {p.relative_to(code_root).as_posix()}")
    out_file.write_text("\n".join(rows)+"\n", encoding="utf-8")
