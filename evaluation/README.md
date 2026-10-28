# Evaluation

| 檔案 | 用途 |
|---|---|
| `benchmark.py` | 評測任何已註冊的方法（`--method autoregressive | sync_layerskip | <solutions 註冊的方法>`），同一協定、同一輸出格式 |
| `autoregressive.py` | AR baseline（= `benchmark.py --method autoregressive`） |
| `compare.py` | 彙整結果 JSON，依相同設定配對 AR，輸出相對 AR 加速比表格（markdown + CSV） |
| `check_lossless.py` | 比對方法與 AR 的 greedy 輸出（計畫 6.5「與 AR 的輸出一致性」） |
| `run_baselines.sh` | 4090 上的 baseline 掃描（AR 吞吐量、AR static 輸出、可選的同步 LayerSkip + 無損檢查） |

目前內建方法：`autoregressive`（計畫 6.3 第一個 baseline）、`sync_layerskip`（第二個 baseline：固定 k、γ 的同步批次 LayerSkip）。
其餘 baseline 與完整方法在 `solutions/` 註冊後即可使用（見 `solutions/README.md`）。

## 評測協定（沿用 ASPIRE `benchmark.py`）
* `--mode cyclic`（預設）：batch 永遠保持滿載，完成的請求立即以下一個 prompt 補上；前 `--warmup-samples` 個完成不計；
  累積 decode throughput 在 `--converge-window` 個完成內變動 < `--converge-threshold`（1%）即停止，或達 `--max-eval-samples`。
* `--mode static`：每個 prompt 恰好生成一次；搭配 `--save-outputs` 保存 greedy token。
* 只計 decode forward 時間（CUDA events），prefill 不計；另輸出 wall-clock throughput 與每請求 TPOT。
* `--batch-sizes 1,4,16,max`：`max` = 放得下的最大 batch；放不下的 batch size 會被跳過並印出原因（4090 上的 8B 很常見）。

```bash
# AR 吞吐量（掃 batch size）
python evaluation/benchmark.py --method autoregressive --model layerskip-llama3-8B --dataset cnn_dm_summarization \
    --batch-sizes 1,4,8,16,max --max-prompt-tokens 1536 --output-dir evaluation/results/cnn_dm

# 同步 LayerSkip（k* 來自 observation O3）
python evaluation/benchmark.py --method sync_layerskip --exit-layer 8 --gamma 4 --model layerskip-llama3-8B \
    --dataset cnn_dm_summarization --batch-sizes 1,4,8,16,max --max-prompt-tokens 1536 --output-dir evaluation/results/cnn_dm

# 無損檢查
python evaluation/benchmark.py --method autoregressive  ... --mode static --save-outputs --batch-sizes 16 --output-dir R/static
python evaluation/benchmark.py --method sync_layerskip ... --mode static --save-outputs --batch-sizes 16 --output-dir R/static
python evaluation/check_lossless.py --reference R/static/autoregressive-B16-outputs.jsonl \
    --candidate R/static/sync_layerskip-k8-g4-B16-outputs.jsonl

# 彙整
python evaluation/compare.py evaluation/results/cnn_dm --out evaluation/results/cnn_dm/table.csv
```

輸出（每個 batch size、每次 repetition 一個檔）：`<method><tag>-B<B>-rep<r>.json`，內含
`method_params`、`config`、`dataset`（prompt 數、長度、token sha256）、`env`（GPU、torch/transformers 版本、git commit）、
`metrics`（`throughput_tok_s`、`wall_throughput_tok_s`、`mean_tpot_ms`、每步平均時間、停止原因；speculative 方法另有
`spec.acceptance_rate`、`tokens_per_verify`、`rejected_draft_ratio`、`overdraft_step_ratio` 等）、`outputs`。

**相對 AR 的加速比** = 同一模型、資料集、prompt 長度設定、生成長度與 batch size 下，方法 `throughput_tok_s` ÷ AR 的（`compare.py` 自動配對）。

**bf16 與無損**：不同 batch 組成會改變 kernel 形狀，bf16 下極少數接近平手的位置 argmax 可能翻轉，之後整段輸出分岔。
`check_lossless.py` 會報告第一個分岔位置；嚴格驗證請加 `--dtype fp32`（8B fp32 需 32 GB，4090 上請用 1B）。
