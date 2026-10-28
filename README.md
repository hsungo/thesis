# 批次化 Early-Exit Self-Speculative Decoding（碩士論文研究程式碼）

以非同步排程與 Verification Coalescing 提升 layer-skip 自我推測解碼的批次吞吐量。
研究計畫見 `研究計畫_批次化EarlyExit自我推測解碼.pdf`；參考實作：
[ASPIRE](https://github.com/Amir-zsh/ASPIRE)（評測協定、排程公式）、[DEL](https://github.com/hoenza/DEL)（LayerSkip 前向語意、prompt 格式）。

## 目錄

```
common/        共用核心（所有實驗與之後的 solutions 都用這一份）
  model.py       純 PyTorch LayerSkip-Llama：任意層區間 / 任意請求子集 forward、逐層計時、靜態 KV cache
  engine.py      批次 decode engine：AR、同步 LayerSkip；ASPIRE 式 cyclic replay driver
  policies.py    排程策略介面 + ASPIRE 的 fixed / FSM / latency-aware 策略
  simulator.py   trace 驅動的非同步批次模擬器（可先在模擬中驗證 S2/S3 排程）
  traces.py      一次 AR 生成取得所有 (exit layer, draft length) 的精確接受長度（計畫 3.1）
  cost_model.py  以層為單位的批次感知成本模型（計畫 S2，O5 擬合）
  memory.py      KV cache 記憶體規劃（4090 上哪些 batch size 放得下）
  methods.py     解碼方法註冊表（evaluation/benchmark.py 用）
  data.py        資料集與 prompt 格式；profiler.py / timing.py / cli.py / utils.py
observation/   O1–O7 動機實驗（見 observation/README.md）
  v2/            補充實驗（全部實機量測）：設定核對、兩段式 engine、邊際成本、roofline、實機 grid、到達模式
  results/       各實驗的 SUMMARY / REPORT / CSV / 圖（原始 run JSON、trace、log 不進 git）
solutions/     EESD engine：合併 shallow/deep 執行、M1 批次觸發、M2 動態退出層、M3 prefill 搭載、
               可攜式成本模型、校準、無損檢查、消融（見 solutions/README.md）
evaluation/    benchmark（任意方法）、AR baseline、結果彙整、無損檢查（見 evaluation/README.md）
results/eesd/  solutions 的輸出：calibration/、lossless/、<preset>/（runs.csv、summary.csv、SUMMARY.md）
scripts/       setup_venv.sh、check_env.py、prefetch_data.py
tests/         pytest（CPU）：HF 一致性、無損（含 EESD 全部開關組合）、成本模型、排程、消融 driver
models/        LayerSkip checkpoints（不納入版本控制；從 HF 下載 facebook/layerskip-*）
```

## 在 Linux GPU server（RTX 4090）上開始

```bash
# 0. 把整個資料夾（含 models/）複製到 server，進入 repo 根目錄
bash scripts/setup_venv.sh            # 建 .venv、依 driver 安裝 CUDA 版 PyTorch、pip install -e .[dev]、跑測試與環境檢查
source .venv/bin/activate
python scripts/check_env.py --smoke layerskip-llama3-1B   # GPU、SDPA kernel、記憶體規劃、decode 速度
python scripts/prefetch_data.py       # 先把所有資料集下載到 HF cache（可先設 HF_HOME 到大磁碟）
python -m pytest -m slow              # 用真實 1B checkpoint 驗證（EESD_TEST_DEVICE=cuda 可在 GPU 上跑）

# 1. 動機實驗（計畫階段 2–3）；最後一步產生 observation/results/<model>/REPORT.md 與 paper/（論文用表格、圖、數字）
tmux new -s obs 'bash observation/run_all.sh'
# 2. Baselines
tmux new -s eval 'K=<k*> bash evaluation/run_baselines.sh'
```

### 4090（24 GB）的記憶體限制
KV cache 是連續配置（B × (最長 prompt + max_new)）。`scripts/check_env.py` 會印出下表的實機版本：

| 模型 | 權重 | KV / token | max_seq 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| LayerSkip Llama3-8B | 15 GiB | 128 KiB | B≈58 | B≈28 | B≈13 |
| LayerSkip Llama3.2-1B | 2.5 GiB | 32 KiB | B≈640 | B≈318 | B≈157 |

所以 8B 上 B=64 只在 prompt+生成 ≲ 900 tokens 時可行；腳本的 `--batch-sizes ...,max` 會自動跳過放不下的 batch size。
需要大 batch 的實驗（O4 的 B 掃描、S3 的收益地圖）可在 1B 上做完整掃描、在 8B 上做短 context 掃描。

## 開發流程
* 改了 `common/` 之後先跑 `python -m pytest`（CPU，約 15 秒）。無損與模擬器一致性測試會擋下大部分錯誤。
* 新的解碼方法：實作一個 `DecodeEngine` 子類，在 `solutions/` 用 `common.methods.register(...)` 註冊，
  `evaluation/benchmark.py --method <name>` 就能用同一協定評測、`evaluation/compare.py` 自動算相對 AR 加速比。
* 新的排程策略（S2/S3）：實作 `common.policies.Policy`，先用 `common.simulator.AsyncTraceSimulator` 在 trace 上驗證，
  再接到真實 engine。
* 所有結果 JSON 都記錄完整參數、GPU / 版本與 git commit；建議 `git init` 後再開始跑正式實驗。

## EESD 實驗（solutions/，全部為實機量測）

先在 CPU 跑測試（不需要 GPU）：`python -m pytest`。以下每一步都會用到 GPU，請依序執行；全部都可以續跑。
長時間的工作建議用 `systemd-run --user --unit=<名字> --collect --working-directory=$PWD <指令>` 或 tmux 啟動，
避免登出時被中斷。

```bash
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false

# 0. 校準成本模型（每台 GPU、每個模型一次），並回報預測誤差 MAPE
for m in layerskip-llama3-8B layerskip-llama3-1B layerskip-llama2-7B; do
  python -m solutions.calibrate --model $m --validate
done

# 1. 無損檢查：fp32 的 1B 必須 100% 一致；bf16 的 8B 回報一致率（含 AR 自身在不同 B 下的一致率作為下限）
python -m solutions.check_lossless --model layerskip-llama3-1B --dtype fp32
python -m solutions.check_lossless --model layerskip-llama3-8B --dtype bf16

# 2. 階段 0–3（每階段一個短 benchmark；--dry-run 可先列出要跑的組合）
python -m solutions.run_ablation --preset smoke          # 幾分鐘，確認流程
python -m solutions.run_ablation --preset phase0         # AR / fixed γ / aspire（重現舊數據）
python -m solutions.run_ablation --preset phase1         # + M1、M1 + linear 成本模型
python -m solutions.run_ablation --preset ksweep         # fixed k = 2..8（M2 的收益上限）
python -m solutions.run_ablation --preset phase2         # + M2
python -m solutions.run_ablation --preset prefill-share  # prefill 時間占比（M3 的收益上限）
python -m solutions.run_ablation --preset phase3         # + M3

# 3. 完整消融：3 模型 × 4 任務 × prompt 256/512/1024 × B∈{1,4,8,16,32} × 15 組合 × 3 次（8100 次執行，需數天）
python -m solutions.run_ablation --preset full
python -m solutions.run_ablation --preset poisson        # 動態模式（Poisson 到達）

# 只重算 CSV 和表格（不用 GPU）
python -m solutions.run_ablation --preset full --summarize
```

輸出在 `results/eesd/`，說明見 `solutions/README.md`。
