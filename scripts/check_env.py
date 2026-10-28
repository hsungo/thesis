"""Check the GPU environment before running experiments.

    python scripts/check_env.py                     # versions, GPU, SDPA kernels, memory plan
    python scripts/check_env.py --smoke layerskip-llama3-1B   # + load a model and time a few decode steps

Prints which batch sizes fit for each local model at several sequence lengths, so
you can pick `--batch-sizes` / `--max-prompt-tokens` that work on this GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from common.memory import plan_table, weight_bytes  # noqa: E402
from common.model import ModelConfig  # noqa: E402
from common.utils import MODELS_DIR, env_info  # noqa: E402

GiB = 2**30


def check_sdpa(dev: torch.device) -> None:
    """The model needs (a) SDPA with a boolean mask (decode / verify) and (b) causal SDPA (prefill)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(4, 8, 4 * 5, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(4, 8, 1024, 128, device=dev, dtype=torch.bfloat16)
    mask = torch.ones(4, 1, 20, 1024, dtype=torch.bool, device=dev).tril(diagonal=1004)
    for name, backend in [("efficient (masked decode)", SDPBackend.EFFICIENT_ATTENTION),
                          ("math (fallback)", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q, k, k, attn_mask=mask)
            print(f"  SDPA {name:28s} OK")
        except RuntimeError as e:
            print(f"  SDPA {name:28s} NOT available ({str(e).splitlines()[0][:80]})")
    qc = torch.randn(1, 32, 2048, 128, device=dev, dtype=torch.bfloat16)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.nn.functional.scaled_dot_product_attention(qc, qc, qc, is_causal=True)
        print(f"  SDPA {'flash (causal prefill)':28s} OK")
    except RuntimeError as e:
        print(f"  SDPA {'flash (causal prefill)':28s} NOT available ({str(e).splitlines()[0][:80]})")


def smoke(model_name: str, dev: torch.device) -> None:
    from common.engine import ARBatchEngine, RunConfig
    from common.data import Prompt
    from common.model import LayerSkipLlama

    path = MODELS_DIR / model_name
    print(f"\n[smoke] loading {path}")
    t0 = time.time()
    m = LayerSkipLlama.from_pretrained(path, device=str(dev), dtype=torch.bfloat16, max_seq=2048)
    print(f"[smoke] loaded in {time.time() - t0:.1f}s, GPU memory used {torch.cuda.memory_allocated() / GiB:.1f} GiB")
    prompts = [Prompt(f"p{i}", [128000] + list(range(1000 + i, 1500 + i))) for i in range(8)]
    for B in (1, 8):
        eng = ARBatchEngine(m, B, 600, 64, (), ignore_eos=True)
        res = eng.run(prompts, RunConfig(mode="static"))
        ms = res["metrics"]["steps"]["ar"]["p50_ms"]
        print(f"[smoke] AR decode B={B}: {ms:.2f} ms/step  ({B * 1000 / ms:.0f} tok/s)")
        m.free_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", default=None, help="model folder under ./models to load and time")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = env_info(dev)
    for k, v in info.items():
        print(f"  {k:22s} {v}")
    if dev.type != "cuda":
        print("\n!! CUDA is not available: install a CUDA build of PyTorch (scripts/setup_venv.sh).")
        return
    cap = torch.cuda.get_device_capability()
    print(f"  {'compute capability':22s} sm_{cap[0]}{cap[1]}   bf16: {torch.cuda.is_bf16_supported()}")
    check_sdpa(dev)

    total = torch.cuda.get_device_properties(dev).total_memory
    print(f"\nMemory plan for {total / GiB:.1f} GiB (contiguous KV cache, bf16; max batch per max_seq = prompt + new tokens):")
    seqs = [512, 1024, 2048, 3072, 4096, 8192]
    for d in sorted(MODELS_DIR.glob("*/config.json")):
        cfg = ModelConfig.from_hf_dir(d.parent)
        rows = plan_table(cfg, total, seqs)
        print(f"  {d.parent.name:24s} weights {weight_bytes(cfg) / GiB:5.1f} GiB, KV {cfg.kv_bytes_per_token // 1024} KiB/token  ->  "
              + "  ".join(f"{r['max_seq']}:{r['max_batch']}" for r in rows))
        if not any(d.parent.glob("*.safetensors")):
            print(f"    !! no *.safetensors in {d.parent}")
    if args.smoke:
        smoke(args.smoke, dev)


if __name__ == "__main__":
    main()
