"""Autoregressive (AR) baseline - plan sec.6.3, the reference every speedup is measured against.

Same weights, same engine code and the same batch size as the speculative methods;
every step runs all N layers for one token per request.  This is
`evaluation/benchmark.py --method autoregressive`; see that file for the protocol
(ASPIRE-style cyclic replay, decode-only throughput) and the output format.

Examples
    python evaluation/autoregressive.py --model layerskip-llama3-8B --dataset cnn_dm_summarization \
        --batch-sizes 1,4,8,16,max --output-dir evaluation/results/cnn_dm
    python evaluation/autoregressive.py --model layerskip-llama3-8B --dataset human_eval \
        --mode static --batch-sizes 16 --save-outputs --output-dir evaluation/results/humaneval_static
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark import main  # noqa: E402

if __name__ == "__main__":
    main(default_method="autoregressive")
