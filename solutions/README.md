# solutions/ — EESD engine（三個可獨立開關的機制）

批次 early-exit self-speculative decoding（LayerSkip，greedy，無損）。draft 只跑淺層 `[0,k)`，驗證跑深層 `[k,L)`。
深層一次的成本高而且幾乎固定（膝點以下時間主要花在讀權重），所以三個機制的目標都是**讓深層少跑、整批共用**。

所有結果都必須在實機（GPU）上量測。成本模型只負責**決策**，以及用來跟實測比較誤差（S2），
不能拿它的預測值當作結果。

## 檔案

| 檔案 | 內容 |
|---|---|
| `engine.py` | 合併執行引擎 `EESDEngine`（shallow / deep step、KV 長度指標回滾、CPU 端 NumPy 狀態），以及三種 driver：cyclic replay（ASPIRE 協定）、static（每個 prompt 一次）、Poisson 到達 |
| `packed.py` | 不等長批次：token 接成一條，用 `qo_indptr = cumsum(d_i+1)` 分段。GEMM / MLP / head 不做 padding；attention 依 query 長度分桶（`exact`：同長度一桶，完全不 pad；`pad`：每段一桶） |
| `cost_model.py` | 可攜式 roofline 成本模型：`t_layer(n,C) = max(W/BW, 2·P·n/F_eff) + KV(C)/BW + n_op·t_op`。提供 `t_sh / t_deep / t_prefill / t_segment / knee`，GEMM 項啟動時先查表；`knee()` 回傳實測膝點。另有 `roofline-peak`（舊版：8192³ 峰值 F 與解析膝點，只用於比較 MAPE）和 ASPIRE 式 3 參數線性模型（消融用） |
| `schedulers.py` | `--sched fixed / aspire / m1`，以及比較對象 `fsm-sync / aspire-fsm / m1-oracle` |
| `exit_select.py` | `--exit m2`（整批共用、動態選擇退出層） |
| `prefill.py` | `--prefill separate / m3`（依深度拆開的 prefill 搭載） |
| `calibrate.py` | 每台 GPU、每個模型量一次 BW、峰值 F、**F_eff**（模型自己的 QKV/O/W13/W2 形狀在 n = 1..512 的最高有效算力）、**knee_measured**（一層 decoder 在 n = 1..512 的實測時間做 hinge 擬合）、t_op、n_op 和線性擬合，存成 JSON；`--validate` 量實際 step 時間，報告 roofline（新）、roofline_peak（舊）、linear 三者的 MAPE，並單獨報告 n = 64..256 這一段 |
| `make_ref.py` | 產生 m1-oracle 用的 AR 參考輸出（每個位置的 AR token，以及 exit 2..8 的 draft 是否等於它），存在 `results/eesd/ref/<model>/<task>_L<len>.npz` |
| `check_lossless.py` | 實機無損檢查：每種開關組合都和 AR 比對（fp32 要完全一致；bf16 回報一致率與 AR 自身的下限） |
| `run_ablation.py` | 量測與消融（presets、CSV、平均 ± 標準差摘要表，可續跑） |

CPU 測試在 `tests/test_eesd.py`（`python -m pytest`，不需要 GPU），涵蓋：
- 24 種開關組合加上 AR，都和 AR 逐 token 相同；
- 不等長分段；
- 回滾不搬動 KV；
- 搭載 prefill 寫出的 KV 和一般 prefill 相同；
- 換 k 後輸出不變；
- shallow step 不做 GPU→CPU 同步，每個 deep step 只同步一次；
- M1 規則的方向；
- 成本模型公式與 IO；
- 消融 driver。

## 共同基礎
- **shallow step**：所有 decoding slot 拿目前的輸入 token（d = 0 時是最後確認的 token，否則是最新的 draft）跑 `[0,k)` 和共用的 exit head，各產生 1 個 draft。第 k 層的 hidden state 存在 `exit_h[slot, d]`，KV 寫進前 k 層。
- **deep step**：要驗證的 slot 先把最新的 draft 跑 `[0,k)`，再把全部 d_i+1 個 token 跑 `[k,L)`（輸入是快取的第 k 層 hidden state），在 GPU 上做 greedy 接受判斷，接受 a_i 個並確認 a_i+1 個 token。不驗證的 slot 在同一步 draft。
- **回滾**：每個 request 的已確認位置 `pos` 就是 KV 長度指標。被拒 token 的 KV 留在指標之後，不會被讀到（query 只讀 ≤ 自己位置的 KV），下一輪會覆寫，不搬任何資料。
- **同步**：shallow step 和排程器都不做 GPU→CPU 同步。每個 deep step 只讀回一次，內容是接受數、驗證後的 token、M2 probe 計數、M3 首個 token。
- **持續補位**：照 ASPIRE 的 cyclic replay。完成即補位；前 max(8, B) 個完成不計；量測至少 max(16, 2B) 個完成，吞吐量變動連續 4 次 < 1% 就停止，最多 256 個。
- **計時**：每次執行回報三種吞吐量，speedup 也三種都算（相對同模型、任務、長度、B、重複次數的 AR）：
  - **decode-only**（`throughput_tok_s`）= 確認 token 數 / 非 prefill step 的 GPU 時間（CUDA events，ASPIRE 口徑）；
  - **端到端 GPU**（`e2e_throughput_tok_s`）= 確認 token 數 / (decode + separate prefill 的 GPU 時間)；
  - **牆鐘**（`wall_throughput_tok_s`）= 確認 token 數 / 量測區間的 perf_counter 時間（開始和結束前都 synchronize），包含 step 之間的 Python 處理、規劃與打包。`host_overhead_share = 1 − GPU 時間 / 牆鐘時間`。
  - 只看 GPU 時間會讓 host 端工作較多的設定（M2、M3）看起來比實際快，比較方法時以牆鐘為準。
  - M3 搭載的 prefill 成本算在 decode step 裡，所以 M3 的 decode-only 吞吐量是保守的。
  - 收斂判斷仍沿用 decode-only 吞吐量（ASPIRE 協定）。

## 三個機制
- **M1（`--sched m1`）**：每一步之後比較
  `R(s) = Σ E_i(d_i) / (s·t_sh + t_deep)` 和 `R(s+1)`，其中 `E_i(d) = (1−α_i^{d+1})/(1−α_i)`。
  - R(s+1) > R(s) 就繼續 draft，否則跑 deep step。
  - 成本的算法：淺層（每個 shallow step，以及 deep step 本身的淺段）用全部 decoding slot 的 token 和 context；
    深段的 token 數與 KV 讀取量只算會驗證的 slot：R(s) 用 d ≥ 1 的 slot，R(s+1) 用 d + 1 之後的全部 slot（context 各多 1）。
  - 強制驗證：任何 d_i ≥ d_max，或已等待 `--sla-steps` 步。
  - 所有 d_i ≥ 1 的 request 一定驗證；d_i = 0 的在同一步 draft。
  - α 用 ASPIRE 的 EMA 更新（ω = 0.8，α0 = 0.9）。
  - 規劃在 CPU 上向量化計算，每步的規劃耗時記在 `plan_ms_per_step`。
- **M2（`--exit m2`）**：
  - LayerSkip 的每個 exit 都共用 final norm 和 lm_head（`LayerSkipLlama.head`），所以可以在執行中換 k，不需要額外權重。
  - 每 `--probe-every` 個 deep step 做一次 probe：驗證 token 重算所有淺層，在候選層 `--k-cands` 套上共用的 head，和 Y 比對（只看前綴正確的位置），估計 α(k)。probe 的成本算在該步的實測時間裡。
  - 用成本模型算各 k 的 `R*(k)`；新 k 要好 `--switch-margin` 以上才換，而且只在 deep step 剛結束、沒有做到一半的 prefill 時換。
  - 換 k 時捨棄舊 k 產生的未驗證 draft（draft 只是猜測，所以仍然無損）。
  - k 隨時間的變化記在 `k_timeline`。
- **M3（`--prefill m3`）**：
  - 塊大小 `c = min(剩餘, knee·s − 這一步淺段的 token 數, knee·s − 預估下一個深段的 token 數)`。knee 是校準實測的膝點（不再用峰值 F 推算；舊算法約 169 tokens，實測約 104），s = `--knee-safety`（預設 0.9）。
  - 每塊在某一步的淺段跑 `[0,k)`，存下第 k 層的 hidden state；在下一個 deep step 的深段跑 `[k,L)`。每層的 KV 都會寫入，後面的塊可以 attend 到前面的塊。
  - 最後一塊做完時產生第一個 token，該 request 從 d = 0 開始 draft。
  - 退回一般 prefill 的條件（在 request 進入 slot 時判斷）：沒有低於膝點的空間、需要超過 `--max-wait` 個 deep step，或成本模型估計 Δt(c) ≥ 另外跑的時間。原因記在 `m3_fallback_reasons`。
  - 若所有 slot 都還在 prefill（沒有 decoding slot），會直接在同一步把整塊跑完深淺兩段，避免卡住。

## 比較對象（不是方法）
| config | 說明 |
|---|---|
| `fsm-sync` | 同步回合，整批共用一個 γ（起始 `--base-gamma`）。每次 deep step 後，超過一半的驗證者全部接受 → γ + 1，否則 γ − 1，限制在 [1, d_max] |
| `aspire-fsm` | ASPIRE 的 FeedbackPolicy：每個 request 各自的目標長度（起始 3），全部接受 +1、否則 −1，各自觸發驗證 |
| `m1-oracle` | 診斷用：決策規則和 M1 相同，但 E_i(d) 換成真實的確認 token 數 min(run_i, d) + 1。run_i 來自事先存好的 AR 參考輸出（`make_ref.py`，同一批 prompt），因此不需要額外的 GPU 同步。bf16 下輸出偏離參考時，該 request 改用 α 估計，比例記在 `oracle_coverage`。找不到參考輸出時直接報錯並印出產生指令。只能搭配固定 exit |

已知限制（會影響量測，請在論文中寫明）：
- 沒有 FlashInfer / flash-attn，attention 用 SDPA。
- 驗證列不連續時要 gather KV（動機實驗 observation B1 量到 +10～31%；observation/ 另外存放）。
- eager 模式，沒有 CUDA Graph。
- cyclic 量測一開始填滿 batch 時，一律用 separate prefill（此時還沒有可以搭載的 step）。

## 輸出
執行指令見 repo 根目錄 README 的「EESD 實驗」一節。輸出在 `results/eesd/<preset>/`：

| 檔案 | 內容 |
|---|---|
| `runs/<model>/<task>_L<len>/B<B>/<config>_rep<r>.json` | 單次執行的完整結果，含 env、seed、主機、校準檔（不進 git） |
| `runs.csv` | 每次執行一列（含 speedup vs AR、bf16 token 一致率） |
| `summary.csv` | 每組 3 次的平均與標準差 |
| `SUMMARY.md` | 表格 |
| `skipped/` | 放不下或 prompt 不夠而跳過的組合，以及原因 |
