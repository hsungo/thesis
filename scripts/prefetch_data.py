"""Download every dataset used by the study into the Hugging Face cache once, so long
experiment runs never stall on (or fail because of) the network.

    python scripts/prefetch_data.py                       # all datasets
    python scripts/prefetch_data.py --datasets gsm8k human_eval
Set HF_HOME (e.g. on a large disk) before running if the home directory is small.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import DATASETS, load_prompts, load_tokenizer  # noqa: E402
from common.utils import resolve_model_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=[d for d in DATASETS if d != "custom_jsonl"])
    ap.add_argument("--model", default="layerskip-llama3-8B", help="tokenizer to load")
    args = ap.parse_args()
    tok = load_tokenizer(resolve_model_path(args.model))
    failed = []
    for ds in args.datasets:
        t0 = time.time()
        try:
            ps = load_prompts(ds, tok, 4, scan_limit=50)
            print(f"[prefetch] {ds:22s} OK  ({time.time() - t0:5.1f}s, example prompt {len(ps[0].tokens)} tokens)")
        except Exception as e:  # noqa: BLE001
            failed.append(ds)
            print(f"[prefetch] {ds:22s} FAILED: {type(e).__name__}: {str(e)[:150]}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
