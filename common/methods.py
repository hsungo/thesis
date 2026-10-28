"""Registry of decoding methods for `evaluation/benchmark.py`.

Each method contributes its own CLI arguments and an engine factory; the benchmark
driver, output format and metrics are shared, so every baseline / solution is
measured exactly the same way.  Add a method with:

    from common.methods import MethodSpec, register

    def _args(p):  p.add_argument("--rho-c", type=float, default=0.25)
    def _build(model, args, batch_size, max_prompt_len, eos):
        return MyEngine(model, batch_size, max_prompt_len, args.max_new_tokens, eos, ...)
    register(MethodSpec("coalesce", "S1+S2+S3", _args, _build, tag=lambda a: f"-k{a.exit_layer}"))

Methods defined under `solutions/` are picked up automatically if
`solutions/__init__.py` imports the modules that call `register` (see load_plugins).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from common.engine import ARBatchEngine, DecodeEngine, SyncSpecEngine


@dataclass
class MethodSpec:
    name: str
    description: str
    add_args: Callable                                # (argparse group) -> None
    build: Callable[..., DecodeEngine]                # (model, args, batch_size, max_prompt_len, eos) -> engine
    tag: Callable = field(default=lambda args: "")    # extra file-name tag, e.g. "-k8-g4"
    supports_sampling: bool = False
    spec_slack: int = 0                               # extra KV positions per request beyond max_new
    step_tokens: Callable = field(default=lambda args: 1)  # max tokens per request in one step (memory planning)
    params: Callable = field(default=lambda args: {})  # method parameters recorded in the result JSON


METHODS: dict[str, MethodSpec] = {}


def register(spec: MethodSpec) -> MethodSpec:
    if spec.name in METHODS:
        raise ValueError(f"method {spec.name!r} already registered")
    METHODS[spec.name] = spec
    return spec


def load_plugins(modules: Optional[list[str]] = None) -> list[str]:
    """Import `solutions` (and any extra modules) so they can register methods."""
    loaded = []
    for name in ["solutions"] + list(modules or []):
        try:
            importlib.import_module(name)
            loaded.append(name)
        except ModuleNotFoundError as e:
            if e.name != name:  # the plugin exists but one of its imports is missing
                raise
    return loaded


# ----------------------------------------------------------------------------- built-in methods

def _sampling_kw(args) -> dict:
    return dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)


def _ar_args(g) -> None:
    pass


def _ar_build(model, args, batch_size, max_prompt_len, eos):
    return ARBatchEngine(model, batch_size, max_prompt_len, args.max_new_tokens, eos,
                         ignore_eos=args.ignore_eos, **_sampling_kw(args))


register(MethodSpec(
    "autoregressive", "all N layers per token (baseline)", _ar_args, _ar_build, supports_sampling=True))


def _sync_args(g) -> None:
    g.add_argument("--exit-layer", type=int, default=None, help="[sync_layerskip] exit layer k")
    g.add_argument("--gamma", type=int, default=4, help="[sync_layerskip] draft length")


def _sync_build(model, args, batch_size, max_prompt_len, eos):
    if args.exit_layer is None:
        raise SystemExit("--exit-layer is required for sync_layerskip")
    return SyncSpecEngine(model, batch_size, max_prompt_len, args.max_new_tokens, eos,
                          ignore_eos=args.ignore_eos, exit_layer=args.exit_layer, gamma=args.gamma)


register(MethodSpec(
    "sync_layerskip", "synchronous batched LayerSkip (fixed k, gamma; plan P1 baseline)",
    _sync_args, _sync_build,
    tag=lambda a: f"-k{a.exit_layer}-g{a.gamma}",
    spec_slack=64,
    step_tokens=lambda a: a.gamma + 1,
    params=lambda a: {"exit_layer": a.exit_layer, "gamma": a.gamma},
))
