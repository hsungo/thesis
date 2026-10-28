#!/usr/bin/env bash
# Baseline sweeps for plan sec.6 on one RTX 4090.  Run from the repo root inside the venv:
#   tmux new -s eval 'bash evaluation/run_baselines.sh'
#   MODEL=layerskip-llama3-1B BATCHES=1,4,16,64,128,max bash evaluation/run_baselines.sh
#
# 1. AR throughput per dataset x batch size  (reference of every speedup)
# 2. AR greedy outputs in static mode        (reference of the losslessness check)
# 3. optional: synchronous LayerSkip at (K, G) if K is set
# Then:  python evaluation/compare.py $R --out $R/table.csv
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-layerskip-llama3-8B}
R=${R:-evaluation/results/${MODEL}}
DATASETS=${DATASETS:-"cnn_dm_summarization xsum_summarization human_eval gsm8k"}
BATCHES=${BATCHES:-1,4,8,16,32,max}       # sizes that do not fit are skipped automatically
MAX_NEW=${MAX_NEW:-512}
MAX_PROMPT=${MAX_PROMPT:-1536}            # keeps B=32 possible for 8B on 24 GB
K=${K:-}                                  # exit layer (k* from observation O3), optional
G=${G:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$R/logs"

for ds in $DATASETS; do
  common=(--model "$MODEL" --dataset "$ds" --max-new-tokens "$MAX_NEW" --max-prompt-tokens "$MAX_PROMPT")
  python evaluation/benchmark.py --method autoregressive "${common[@]}" --batch-sizes "$BATCHES" \
      --output-dir "$R/$ds" 2>&1 | tee -a "$R/logs/ar.log"
  python evaluation/benchmark.py --method autoregressive "${common[@]}" --mode static --limit 64 \
      --batch-sizes 16 --save-outputs --output-dir "$R/$ds/static" 2>&1 | tee -a "$R/logs/ar_static.log"
  if [[ -n "$K" ]]; then
    python evaluation/benchmark.py --method sync_layerskip --exit-layer "$K" --gamma "$G" "${common[@]}" \
        --batch-sizes "$BATCHES" --output-dir "$R/$ds" 2>&1 | tee -a "$R/logs/sync.log"
    python evaluation/benchmark.py --method sync_layerskip --exit-layer "$K" --gamma "$G" "${common[@]}" \
        --mode static --limit 64 --batch-sizes 16 --save-outputs --output-dir "$R/$ds/static" 2>&1 | tee -a "$R/logs/sync.log"
    python evaluation/check_lossless.py --reference "$R/$ds/static/autoregressive-B16-outputs.jsonl" \
        --candidate "$R/$ds/static/sync_layerskip-k$K-g$G-B16-outputs.jsonl" | tee -a "$R/logs/lossless.log" || true
  fi
done
python evaluation/compare.py "$R" --out "$R/table.csv"
