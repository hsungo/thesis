"""Small IO / environment helpers shared by all scripts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_model_path(name_or_path: str) -> Path:
    """Accept either a path or a folder name under ./models (e.g. layerskip-llama3-8B)."""
    p = Path(name_or_path)
    if p.exists():
        return p
    q = MODELS_DIR / name_or_path
    if q.exists():
        return q
    raise FileNotFoundError(f"model not found: {name_or_path} (also looked in {MODELS_DIR})")


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    """Commit hash of the repo (+ '-dirty' if there are uncommitted changes), if it is a git repo."""
    import subprocess

    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        if h.returncode != 0:
            return None
        d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=5)
        return h.stdout.strip() + ("-dirty" if d.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def env_info(device: torch.device) -> dict[str, Any]:
    try:
        import transformers
        tf_version = transformers.__version__
    except ImportError:  # pragma: no cover
        tf_version = None
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": tf_version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "device": str(device),
        "git_commit": git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2)
    return info


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    os.replace(tmp, path)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def parse_int_list(text: str) -> list[int]:
    """'1,4,8' or '2-30' or '2-30:2' -> list of ints."""
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            rng, _, step = part.partition(":")
            a, b = rng.split("-")
            out.extend(range(int(a), int(b) + 1, int(step) if step else 1))
        else:
            out.append(int(part))
    return out
