"""Prompt datasets.  Prompt formats follow LayerSkip / DEL (`DEL/data.py`) so bs=1
numbers stay comparable with those papers; LongBench prompts follow ASPIRE.

Every loader returns `list[Prompt]` with token ids that start with BOS.
Length control (plan sec.6.2 "context 長度分桶"):
    * `min_tokens` / `max_tokens` filter prompts by token length;
    * `truncate_to` left-truncates longer prompts to exactly N tokens (BOS kept,
      the instruction at the end of the prompt is kept) and drops shorter ones.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

DATASETS = (
    "cnn_dm_summarization",
    "cnn_dm_lm",
    "xsum_summarization",
    "human_eval",
    "gsm8k",
    "math500",
    "longbench",
    "custom_jsonl",
)

LONGBENCH_PROMPTS = {
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
# summarisation-style subsets generate long outputs, which is what decoding-throughput studies need
LONGBENCH_DEFAULT_SUBSETS = ("gov_report", "multi_news", "qmsum")

# Task-standard stop strings (`--stop auto`).  The LayerSkip checkpoints are *base* models:
# with EOS as the only stop criterion they keep going after the answer (GSM8K writes new
# "Question:" blocks, HumanEval writes test functions, 0-shot summaries continue the
# article), and those degenerate, often repetitive tails are trivially predictable by a
# shallow exit - they inflate acceptance and show up as fake cross-request heterogeneity.
#   human_eval: bigcode-evaluation-harness / Codex stop sequences
#   gsm8k:      lm-eval-harness style ("Question:" starts the next few-shot block)
#   summaries:  the few-shot summaries are single lines
STOP_STRINGS = {
    "cnn_dm_summarization": ("\n",),
    "xsum_summarization": ("\n",),
    "human_eval": ("\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"),
    "gsm8k": ("\nQuestion:", "\n\n"),
    "math500": ("\nProblem:", "\n\n\n"),
    "cnn_dm_lm": (),
    "longbench": (),
    "custom_jsonl": (),
}
# few-shot defaults used with `--n-shot auto` (prompt formats that make a base model answer
# in the expected shape; GSM8K 8-shot is the usual chain-of-thought setting)
DEFAULT_N_SHOT = {"cnn_dm_summarization": 1, "xsum_summarization": 1, "gsm8k": 8}


@dataclass
class Prompt:
    pid: str
    tokens: list[int]
    reference: str = ""


# ----------------------------------------------------------------------------- raw loaders

def _load(path: str, *args, **kwargs):
    """`datasets.load_dataset` with a fallback to the auto-converted parquet branch
    (datasets>=4 no longer runs loading scripts)."""
    from datasets import load_dataset

    try:
        return load_dataset(path, *args, **kwargs)
    except Exception as first:  # noqa: BLE001
        try:
            return load_dataset(path, *args, revision="refs/convert/parquet", **kwargs)
        except Exception:  # noqa: BLE001
            raise first


def _shots(rows, n_shot: int, fmt: Callable[[dict], str]) -> str:
    if n_shot <= 0:
        return ""
    return "".join(fmt(rows[i]) for i in range(n_shot)) + "\n"


def _cnn_dm_summarization(n_shot: int, seed: int) -> Iterable[tuple[str, str, str]]:
    shots = ""
    if n_shot:
        tr = _load("abisee/cnn_dailymail", "3.0.0", split="train").shuffle(seed=seed).select(range(n_shot))
        shots = _shots(tr, n_shot, lambda r: f"Article: {r['article']}\nSummary: {r['highlights'].replace(chr(10), '')}\n")
    for i, r in enumerate(_load("abisee/cnn_dailymail", "3.0.0", split="test")):
        yield f"cnn{i}", shots + f"Article: {r['article']}\nSummary:", " " + r["highlights"]


def _cnn_dm_lm(n_shot: int, seed: int, prefix_words: int = 100):
    for i, r in enumerate(_load("abisee/cnn_dailymail", "3.0.0", split="test")):
        words = r["article"].split()
        yield f"cnnlm{i}", " ".join(words[:prefix_words]), " ".join(words[prefix_words:])


def _xsum(n_shot: int, seed: int):
    shots = ""
    if n_shot:
        tr = _load("EdinburghNLP/xsum", split="train").shuffle(seed=seed).select(range(n_shot))
        shots = _shots(tr, n_shot, lambda r: f"Article: {r['document']}\nSummary: {r['summary'].replace(chr(10), '')}\n")
    for i, r in enumerate(_load("EdinburghNLP/xsum", split="test")):
        yield f"xsum{i}", shots + f"Article: {r['document']}\nSummary:", " " + r["summary"]


def _human_eval(n_shot: int, seed: int):
    for r in _load("openai/openai_humaneval", split="test"):
        yield r["task_id"], r["prompt"], r["canonical_solution"]


def _gsm8k(n_shot: int, seed: int):
    shots = ""
    if n_shot:
        tr = _load("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(n_shot))
        shots = _shots(tr, n_shot, lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}\n")
    for i, r in enumerate(_load("openai/gsm8k", "main", split="test")):
        yield f"gsm{i}", shots + f"Question: {r['question']}\nAnswer:", " " + r["answer"]


def _math500(n_shot: int, seed: int):
    for r in _load("HuggingFaceH4/MATH-500", split="test"):
        yield r["unique_id"], f"Problem: {r['problem']}\nSolution:", " " + r["solution"]


def _longbench_rows(sub: str):
    """zai-org/LongBench only ships a loading script + data.zip (datasets>=4 refuses the
    script), so read data/<subset>.jsonl straight from the zip."""
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(f"/{sub}.jsonl") or n == f"{sub}.jsonl")
        with z.open(name) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _longbench(n_shot: int, seed: int, subsets: Optional[list[str]] = None):
    for sub in subsets or LONGBENCH_DEFAULT_SUBSETS:
        tmpl = LONGBENCH_PROMPTS[sub]
        for i, r in enumerate(_longbench_rows(sub)):
            yield f"{sub}{i}", tmpl.format(context=r["context"], input=r.get("input", "")), str(r.get("answers", ""))


def _custom_jsonl(path: str, prompt_field: str = "prompt", response_field: str = "response"):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                yield str(r.get("id", i)), r[prompt_field], str(r.get(response_field, ""))


# ----------------------------------------------------------------------------- public API

def load_prompts(
    dataset: str,
    tokenizer,
    limit: int,
    *,
    seed: int = 42,
    shuffle: bool = True,
    n_shot: int = 0,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    truncate_to: Optional[int] = None,
    subsets: Optional[list[str]] = None,
    data_path: Optional[str] = None,
    scan_limit: int = 20000,
) -> list[Prompt]:
    """Tokenize up to `limit` prompts that satisfy the length constraints.

    Shuffling is done on the *indices seen* (reservoir over the first `scan_limit`
    rows) so that we never tokenize the whole CNN/DM test split."""
    if dataset == "cnn_dm_summarization":
        rows = _cnn_dm_summarization(n_shot, seed)
    elif dataset == "cnn_dm_lm":
        rows = _cnn_dm_lm(n_shot, seed)
    elif dataset == "xsum_summarization":
        rows = _xsum(n_shot, seed)
    elif dataset == "human_eval":
        rows = _human_eval(n_shot, seed)
    elif dataset == "gsm8k":
        rows = _gsm8k(n_shot, seed)
    elif dataset == "math500":
        rows = _math500(n_shot, seed)
    elif dataset == "longbench":
        rows = _longbench(n_shot, seed, subsets)
    elif dataset == "custom_jsonl":
        if not data_path:
            raise ValueError("--data-path is required for custom_jsonl")
        rows = _custom_jsonl(data_path)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")

    # gather raw rows (bounded), shuffle deterministically, then tokenize lazily
    raw = []
    for j, row in enumerate(rows):
        if j >= scan_limit:
            break
        raw.append(row)
    if shuffle:
        random.Random(seed).shuffle(raw)

    bos = tokenizer.bos_token_id
    out: list[Prompt] = []
    for pid, text, ref in raw:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        if truncate_to is not None:
            if len(ids) < truncate_to:
                continue
            ids = ids[:1] + ids[len(ids) - (truncate_to - 1):] if truncate_to > 1 else ids[:1]
        if min_tokens is not None and len(ids) < min_tokens:
            continue
        if max_tokens is not None and len(ids) > max_tokens:
            continue
        out.append(Prompt(pid=pid, tokens=[int(t) for t in ids], reference=ref))
        if len(out) >= limit:
            break
    if not out:
        raise ValueError(f"dataset {dataset!r} produced no prompt within the length constraints")
    return out


def load_tokenizer(model_path: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path))


def eos_ids(tokenizer, model_cfg) -> tuple[int, ...]:
    """EOS set = config eos ids + tokenizer eos + <|eot_id|> if the vocab has it."""
    ids = list(model_cfg.eos_token_ids)
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>") if hasattr(tokenizer, "convert_tokens_to_ids") else None
    if isinstance(eot, int) and eot != tokenizer.unk_token_id:
        ids.append(eot)
    return tuple(dict.fromkeys(ids))


def add_data_args(parser, default_dataset: str = "cnn_dm_summarization", default_limit: int = 64) -> None:
    g = parser.add_argument_group("data")
    g.add_argument("--dataset", default=default_dataset, choices=DATASETS)
    g.add_argument("--limit", type=int, default=default_limit, help="number of prompts in the pool")
    g.add_argument("--n-shot", default="0", help="int, or 'auto' = task default (DEFAULT_N_SHOT)")
    g.add_argument("--stop", default="none",
                   help="'auto' = task-standard stop strings (STOP_STRINGS), 'none' = EOS only, "
                        "or a '|'-separated list (use \\n for newline)")
    g.add_argument("--min-prompt-tokens", type=int, default=None)
    g.add_argument("--max-prompt-tokens", type=int, default=None)
    g.add_argument("--truncate-to", type=int, default=None,
                   help="left-truncate every prompt to exactly N tokens (drops shorter ones); for context buckets")
    g.add_argument("--longbench-subsets", default=None, help="comma list, default gov_report,multi_news,qmsum")
    g.add_argument("--data-path", default=None, help="jsonl with a 'prompt' field (dataset=custom_jsonl)")
    g.add_argument("--seed", type=int, default=42)


def resolve_n_shot(args) -> int:
    v = str(args.n_shot)
    return DEFAULT_N_SHOT.get(args.dataset, 0) if v == "auto" else int(v)


def stop_strings_from_args(args) -> tuple[str, ...]:
    v = getattr(args, "stop", "none")
    if v == "auto":
        return STOP_STRINGS.get(args.dataset, ())
    if v in ("none", "", None):
        return ()
    return tuple(x.replace("\\n", "\n") for x in v.split("|") if x)


class StopChecker:
    """Detects task stop strings in a request's generated tokens.

    `__call__(output, n_new)` returns the length the output should be cut to (the token
    that completes the first stop string is kept, so the trace stays aligned with what an
    AR run with this stop criterion produces), or None.  Only a tail window is decoded."""

    def __init__(self, tokenizer, stops: tuple[str, ...], window: int = 24):
        self.tok, self.stops, self.window = tokenizer, tuple(s for s in stops if s), window

    def __bool__(self) -> bool:
        return bool(self.stops)

    def _hit(self, ids) -> bool:
        text = self.tok.decode(ids, skip_special_tokens=False)
        return any(s in text for s in self.stops)

    def __call__(self, output: list[int], n_new: int):
        lo = max(len(output) - n_new - self.window, 0)
        if not self._hit(output[lo:]):
            return None
        for n in range(len(output) - n_new + 1, len(output) + 1):   # first prefix containing a stop
            if self._hit(output[max(n - n_new - self.window, 0):n]):
                return n
        return len(output)


def prompts_from_args(args, tokenizer) -> list[Prompt]:
    return load_prompts(
        args.dataset, tokenizer, args.limit, seed=args.seed, n_shot=resolve_n_shot(args),
        min_tokens=args.min_prompt_tokens, max_tokens=args.max_prompt_tokens,
        truncate_to=args.truncate_to,
        subsets=args.longbench_subsets.split(",") if args.longbench_subsets else None,
        data_path=args.data_path,
    )
