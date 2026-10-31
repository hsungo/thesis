"""results/observe-4060/SUMMARY.md (+ C_*.png, G_cheats.csv, ranking.csv) from the CSVs of budget.py, outputs.py,
cheats.py.  No GPU.

    python observation/v3/summary.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "observe-4060"
SHORT = {"cnn_dm_summarization": "cnn_dm", "xsum_summarization": "xsum", "human_eval": "human_eval", "gsm8k": "gsm8k"}


def rd(name: str) -> list:
    p = OUT / name
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pc(x, d=1) -> str:
    return "–" if not np.isfinite(x) else f"{100 * x:.{d}f}%"


def med(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


# ------------------------------------------------------------------------------ cheats

def cheat_rows(A: list) -> list:
    base = {(r["task"], int(r["B"]), r["config"]): r for r in A if r["mode"] == "graph"}
    runs = {}
    for p in sorted((OUT / "cheats").glob("*/B*/*.json")):
        res = json.loads(p.read_text())
        r = res["run"]
        m = res["metrics"]
        runs.setdefault((r["dataset"], r["B"], r["cheat"], r["config"]), []).append(
            (m["wall_throughput_tok_s"], m.get("acceptance_rate"), m.get("ref_missing"), m.get("ref_lookups"),
             m.get("oracle_coverage"), m.get("prefill_time_share")))
    rows = []
    for (ds, B, ch, cname), v in sorted(runs.items()):
        bl = base.get((ds, B, cname))
        thr = [x[0] for x in v]
        row = {"task": ds, "B": B, "cheat": ch, "config": cname, "reps": len(v),
               "cheat_tok_s": med(thr), "cheat_tok_s_q1": float(np.percentile(thr, 25)), "cheat_tok_s_q3": float(np.percentile(thr, 75)),
               "base_tok_s": f(bl["wall_tok_s"]) if bl else float("nan"),
               "base_tok_s_q1": f(bl["wall_tok_s_q1"]) if bl else float("nan"), "base_tok_s_q3": f(bl["wall_tok_s_q3"]) if bl else float("nan"),
               "acceptance": med([f(x[1]) for x in v])}
        row["gain"] = row["cheat_tok_s"] / row["base_tok_s"] - 1
        if ch == "head":
            row["replay_missing_share"] = sum(x[2] or 0 for x in v) / max(sum(x[3] or 0 for x in v), 1)
        if ch == "oracle":
            row["oracle_coverage"] = med([f(x[4]) for x in v])
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------ plots

def plots(cal: list, pos: list, auto: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    models = sorted({r["model"] for r in cal})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for model in models:
        for ds in SHORT:
            rs = [r for r in cal if r["model"] == model and r["task"] == ds and r["signal"] == "top1_prob"]
            if rs:
                axes[0].plot([f(r["mean_signal"]) for r in rs], [f(r["accept_rate"]) for r in rs], "o-" if "8B" in model else "s--",
                             ms=3, label=f"{model[-2:]} {SHORT[ds]}")
            rp = [r for r in pos if r["model"] == model and r["task"] == ds]
            if rp:
                axes[1].plot([int(r["position"]) for r in rp], [f(r["accept_rate"]) for r in rp], "o-" if "8B" in model else "s--",
                             ms=3, label=f"{model[-2:]} {SHORT[ds]}")
            ra = [r for r in auto if r["model"] == model and r["task"] == ds]
            if ra:
                axes[2].plot([int(r["lag"]) for r in ra], [f(r["autocorr"]) for r in ra], "o-" if "8B" in model else "s--",
                             ms=3, label=f"{model[-2:]} {SHORT[ds]}")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=0.8)
    axes[0].set(xlabel="exit-3 top-1 機率", ylabel="接受率", title="校準曲線")
    axes[1].set(xlabel="草稿位置（前面都已接受）", ylabel="條件接受率", title="草稿位置")
    axes[2].set(xlabel="lag（token）", ylabel="自相關", title="請求內接受序列的自相關")
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "C_structure.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------ main

def main() -> None:
    from solutions.feasibility import write_csv
    A, W, D = rd("A_budget.csv"), rd("B_waste_1B.csv"), rd("D_slack.csv")
    Wr, E, Et, F = rd("B_waste_ref.csv"), rd("E_guess.csv"), rd("E_tail.csv"), rd("F_coverage.csv")
    Cp, Cc, Cs, Ct = rd("C_position.csv"), rd("C_calibration.csv"), rd("C_calibration_stats.csv"), rd("C_token_type.csv")
    Cr, Ca, Ce = rd("C_requests.csv"), rd("C_autocorr.csv"), rd("C_exit_layers.csv")
    env = json.loads((OUT / "run_environment.json").read_text(encoding="utf-8-sig"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    runs_A = rd("A_budget_runs.csv")
    tel_clk = [f(r.get("tel_sm_clock_mhz_median")) for r in runs_A]
    tel_tmax = [f(r.get("tel_temp_c_max")) for r in runs_A]
    tel_min = [f(r.get("tel_sm_clock_mhz_min")) for r in runs_A]

    # ---------------- problem shares (graph, best gamma, 10 cells)
    Wd = {(r["task"], r["B"], r["config"], r["mode"]): r for r in W}
    best = [r for r in A if r["mode"] == "graph" and r.get("best_gamma") == "True"]
    eag = {(r["task"], r["B"], r["config"]): r for r in A if r["mode"] == "eager"}
    cells = []
    for r in best:
        w = f(r["wall_ms_per_token"])
        wb = Wd[(r["task"], r["B"], r["config"], "graph")]
        e = eag[(r["task"], r["B"], r["config"])]
        sh = (f(r["sh_layers"]) + f(r["sh_attn_kv"]) + f(r["sh_head"])) / w
        dp = (f(r["dp_layers"]) + f(r["dp_attn_kv"]) + f(r["dp_head"])) / w
        cells.append({"task": r["task"], "B": int(r["B"]), "config": r["config"],
                      "prefill": f(r["prefill"]) / w, "sh_head": f(r["sh_head"]) / w,
                      "sh_rejected": sh * f(wb["shallow_time_on_rejected_drafts"]),
                      "dp_rejected": dp * f(wb["rejected_share_of_deep_time"]), "dp_all": dp,
                      "attn_kv": (f(r["sh_attn_kv"]) + f(r["dp_attn_kv"])) / w, "host": f(r["host_other"]) / w,
                      "metadata": f(r["metadata_upload"]) / w, "accept": f(r["accept_readback"]) / w,
                      "launch": 1 - w / f(e["wall_ms_per_token"]),
                      "sh_layers_k": (f(r["sh_layers"]) + f(r["sh_attn_kv"])) / w})
    write_csv(OUT / "ranking_cells.csv", cells)
    G = cheat_rows(A)
    write_csv(OUT / "G_cheats.csv", G)
    plots(Cc, Cp, Ca)

    def stat(key):
        v = np.array([c[key] for c in cells])
        return float(np.median(v)), float(v.min()), float(v.max())

    def ub(s):
        return 1 / (1 - s) - 1

    def cheat_gain(ch):
        g = [r["gain"] for r in G if r["cheat"] == ch]
        return (med(g), min(g), max(g)) if g else (float("nan"),) * 3

    # data quoted in the ranking table (C-F)
    def frow(model, B):
        """Setting with the best worst-task lower bound of alpha_V/alpha (ties: smaller max |V|):
        (worst lower, worst upper, max |V|, (R, K, F), max undetermined share)."""
        rs = [r for r in F if r["model"] == model and int(r["B"]) == B]
        best_ = None
        for key in {(r["R"], r["K"], r["F"]) for r in rs}:
            sel = [r for r in rs if (r["R"], r["K"], r["F"]) == key]
            if len(sel) < 4:
                continue
            c = (min(f(r["ratio_lower"]) for r in sel), min(f(r["ratio_upper"]) for r in sel),
                 max(int(r["V_max"]) for r in sel), key, max(f(r["undetermined_share"]) for r in sel))
            if best_ is None or (round(c[0], 3), -c[2]) > (round(best_[0], 3), -best_[2]):
                best_ = c
        return best_

    f8_16 = frow("layerskip-llama3-8B", 16)
    f8_32 = frow("layerskip-llama3-8B", 32)
    cs8 = [r for r in Cs if "8B" in r["model"]]
    acc_hi = (min(f(r["accept_p1_ge_05"]) for r in cs8), max(f(r["accept_p1_ge_05"]) for r in cs8)) if cs8 else (np.nan, np.nan)
    acc_lo = (min(f(r["accept_p1_lt_05"]) for r in cs8), max(f(r["accept_p1_lt_05"]) for r in cs8)) if cs8 else (np.nan, np.nan)
    lag1 = [f(r["autocorr"]) for r in Ca if r["lag"] == "1" and "8B" in r["model"]]
    het = [(f(r["p10"]), f(r["p90"])) for r in Cr if "8B" in r["model"]]
    dsl = [r for r in D if r["mode"] == "graph" and r["config"] == "fixed-g4" and r["segment"] == "deep_segment"]
    slack16 = [f(r["slack_median"]) for r in dsl if r["B"] == "16"]
    knee = f(dsl[0]["knee_measured"]) if dsl else float("nan")
    head_feas = [(r["model"], f(r["V_mean"])) for r in F]

    P = [  # key, name, why existing methods do not solve it, batch early-exit property (C-F), 4090 + 8B
        ("sh_head", "淺層 LM head（每個草稿都乘一次 128k 詞彙的 lm_head）",
         "LayerSkip 的 exit 共用完整的 lm_head；靜態的高頻子詞彙（如 FR-Spec）不隨批次內容改變，訓練小 head 則需要額外權重與訓練。",
         (f"F：8B 在 B = 16 時，聯集詞彙最大 {f8_16[2]} 個（R/K/F = {'/'.join(f8_16[3])}，約 128k 的 {f8_16[2] / 128256:.0%}），"
          f"exit-3 限制在其中後 α_V/α 在最差任務為 {f8_16[0]:.3f}–{f8_16[1]:.3f}（下界–上界，{f8_16[4]:.1%} 的位置 top-64 無法判定）；"
          f"B = 32 時最大 {f8_32[2]} 個、{f8_32[0]:.3f}–{f8_32[1]:.3f}。") if f8_16 and f8_32 else "F：見第 F 節。",
         "8B 的 lm_head（1.05 GB）約等於 2.4 層 decoder 的權重，k = 3 時仍占淺層 step 讀取量的四成左右；4090 是頻寬受限，占比預期相近，不會因 launch 開銷消失而變小。"),
        ("prefill", "prefill 暫停整批（separate prefill）",
         "chunked prefill 只把 prefill 切開與 decode 混合，計算量不變；prefix cache 只省得了共用的前綴（few-shot 範例）。",
         f"D：深層 step 的 token 數遠低於膝點（fixed-g4 B = 16 空位中位數 {slack16[0]:.0f} tokens，knee = {knee:.0f}），淺層段只有 B 個 token；同一步的空位可以容納 prompt token（舊 M3 的方向，在這台 GPU 上量到不划算）。" if slack16 else "D：見第 4 節空位。",
         "4090 上 8B 的 prefill 是 compute-bound（512 tokens 遠超膝點），decode 是頻寬受限；prefill 相對 decode 更貴，占比取決於輸出長度，短輸出任務（gsm8k）仍會很高。"),
        ("sh_rejected", "淺層花在之後被拒的草稿",
         "動態草稿長度（per-request）在批次中會讓深層幾乎每步都跑（觀察 O3）；同步批次只能整批共用 γ。",
         (f"C：exit-3 的信心值有校準力：8B top-1 機率 ≥ 0.5 時接受率 {pc(acc_hi[0], 0)}–{pc(acc_hi[1], 0)}，< 0.5 時 {pc(acc_lo[0], 0)}–{pc(acc_lo[1], 0)}；"
          f"請求內 lag-1 自相關 {min(lag1):.2f}–{max(lag1):.2f}，請求間平均接受率 p10–p90 約 {min(h[0] for h in het):.2f}–{max(h[1] for h in het):.2f}。"
          "但淺層 step 在膝點以下，單一 slot 少算一個 token 幾乎不省時間，只有整批都停才省。") if cs8 and lag1 and het else "C：見第 C 節。",
         "8B 的拒絕率更高（B_waste_ref：8B gsm8k γ = 4 有 71% 的草稿被拒），浪費比例預期更大。"),
        ("dp_rejected", "深層花在被拒 token 上（按 token 比例分攤）", "驗證本來就要跑；被拒 token 在膝點以下幾乎不增加時間，按比例分攤高估了可省的量。", "D：深層 token 數低於膝點。", "同左。"),
        ("launch", "kernel 啟動開銷（無 graph − 有 graph）", "CUDA Graph 已解決（本表其他列都是有 graph 的數字）。", "—", "4090 + Linux 的 launch 開銷較小，但 eager 仍有。"),
        ("attn_kv", "attention / KV 讀取（隨 context 增加的部分）", "—（context 512 時很小）", "—", "長 context 時會變大。"),
        ("metadata", "metadata 與上傳", "—", "—", "—"),
        ("host", "host 開銷（其他）", "—", "—", "—"),
        ("accept", "接受判斷與讀回", "—", "—", "—"),
    ]
    ranked = sorted(P, key=lambda p: -ub(stat(p[0])[0]))
    dpall_m = stat("dp_all")[0]
    write_csv(OUT / "ranking.csv", [{"problem": p[1], "share_median": stat(p[0])[0], "share_min": stat(p[0])[1],
                                     "share_max": stat(p[0])[2], "ideal_gain_median": ub(stat(p[0])[0]),
                                     "ideal_gain_max": ub(stat(p[0])[2])} for p in ranked])
    research = [p for p in ranked if p[0] not in ("launch", "dp_rejected")]
    top3 = research[:3]

    L = []
    w = L.append
    w("# 觀察 v3：時間花在哪裡、浪費在哪裡")
    w("")
    w(f"RTX 4060 Laptop（8 GB，Windows），LayerSkip Llama3.2-1B 實機計時；LayerSkip Llama3-8B 與 1B 的輸出型分析。k = 3，bf16，greedy。"
      f"程式在 commit `{commit}`；420 次計時執行都在 commit `0ece9aa`（乾淨工作樹，每個 run JSON 都記錄）。")
    w("")
    w("**雜訊條件**：插電、Turbo 電源計畫；依你的指示直接開始，**Chrome、Discord、Steam、LINE 執行中沒有關閉**"
      f"（`run_environment.json`）。每次執行用 nvidia-smi 每 500 ms 取樣：420 次執行各自的 SM 時脈中位數在 {np.nanmin(tel_clk):.0f}–{np.nanmax(tel_clk):.0f} MHz，"
      f"溫度最高 {np.nanmax(tel_tmax):.0f} °C（每次執行的時脈最低值是執行開頭的閒置取樣）。每個點重複 3 次，報告中位數與四分位距（CSV 有 q1/q3）。")
    w("")
    gh, gp, go = cheat_gain("head"), cheat_gain("prefill"), cheat_gain("oracle")
    w("## 0. 結論")
    w("")
    w(f"1. 有 CUDA Graph 後（launch 開銷占無 graph 時間的 {pc(stat('launch')[0])} 被消除），1B 每個確認 token 的時間主要是：深層驗證 {pc(dpall_m)}、"
      f"淺層 LM head {pc(stat('sh_head')[0])}、prefill {pc(stat('prefill')[0])}、淺層 3 層 {pc(med([c['sh_layers_k'] for c in cells]))}（10 格中位數）。")
    w(f"2. 前三名問題：{'、'.join(p[1].split('（')[0] for p in top3)}。")
    w(f"3. 上限實驗：淺層 LM head 幾乎免費時吞吐量 +{pc(gh[0])}（+{pc(gh[1])}～+{pc(gh[2])}）；prefill 幾乎免費時 +{pc(gp[0])}（+{pc(gp[1])}～+{pc(gp[2])}，"
      f"B 越大、輸出越短越高）；完美的驗證時機（m1-oracle）只有 {pc(go[0])}（{pc(go[1])}～{pc(go[2])}），低於 10%，**不值得做**——"
      "「淺層花在被拒草稿」的時間無法靠選擇驗證時機省下（淺層 step 在膝點以下，少算幾個 token 幾乎不省時間）。")
    w(f"4. 輸出型觀察（8B 與 1B）：草稿接受與 exit 信心值明顯相關、請求間差異大；批次聯集詞彙只需數千個 token（8B B = 32 最多 "
      + (str(f8_32[2]) if f8_32 else "–") + " 個）；免費猜測只有 n-gram 在摘要任務有量，跨請求與被拒尾巴幾乎沒有。")
    w("5. 限制：時間型數據只來自 1B 與這台筆電（launch-bound、Windows），背景程式未關閉；8B 只有輸出型數據。")
    w("")
    w("## 1. 問題排序（有 CUDA Graph；1B；cnn_dm、gsm8k × B = 1–32 共 10 格，每格取最佳同步 γ）")
    w("")
    w("占比 = 該塊時間 ÷ 每確認 1 個 token 的牆鐘時間；理想上限 = 這塊完全消失時的吞吐量提升 1/(1−占比) − 1（**上限估算**，不是方法）。")
    w("")
    w("| 排名 | 問題 | 占比 中位數（範圍） | 理想上限 中位數（最大） | 既有方法為何解決不了 | 批次化 early exit 的哪個特性有機會（C–F 數據） | 4090 + 8B 是否可能不同 |")
    w("|---|---|---|---|---|---|---|")
    for i, p in enumerate(ranked, 1):
        m_, lo, hi = stat(p[0])
        tag = {"launch": "（既有方法已解決，不列入前三）", "dp_rejected": "（按比例分攤高估，不列入前三）"}.get(p[0], "")
        w(f"| {i} | {p[1]}{tag} | {pc(m_)}（{pc(lo)}–{pc(hi)}） | +{pc(ub(m_))}（+{pc(ub(hi))}） | {p[2]} | {p[3]} | {p[4]} |")
    dpall = stat("dp_all")
    w("")
    w(f"不列入排序：**深層驗證本身**占 {pc(dpall[0])}（{pc(dpall[1])}–{pc(dpall[2])}），是無損的代價；其中可省的部分就是上表的「深層花在被拒 token」。"
      "「kernel 啟動開銷」已被 CUDA Graph（既有方法）解決，「深層被拒 token」按比例分攤會高估可省的量，所以兩者不列入前三名。")
    w("")
    w("重疊：「淺層 LM head」與「淺層花在被拒草稿」有交集（被拒草稿的 head 時間兩邊都算），兩者的上限不能相加。")
    w("")
    w("**前三名**：" + "；".join(f"{i}. {p[1]}（占比中位數 {pc(stat(p[0])[0])}，理想上限 +{pc(ub(stat(p[0])[0]))}）"
                                for i, p in enumerate(top3, 1)))
    w("")

    # ---------------- cheats
    w("## 2. 前三名的上限實驗（實機，作弊版：只量速度上限，**不是方法**）")
    w("")
    w("| 問題 | 作弊版 | 實測吞吐量提升 中位數（範圍，10 格 × 設定） | 判定 |")
    w("|---|---|---|---|")
    names = {"head": ("淺層 LM head", "淺層 head 只算 1024 列（argmax 丟棄），草稿從同一格的實際執行記錄重播；接受率與軌跡與真實執行相同"),
             "prefill": ("prefill", "完美 prefix cache：64 個 prompt 事先 prefill 一次存下 KV，執行中每次 prefill 只複製 KV"),
             "oracle": ("淺層花在被拒草稿（驗證時機）", "m1-oracle：驗證時機用真實的接受長度（AR 參考輸出）決定")}
    for ch in ("head", "prefill", "oracle"):
        g = cheat_gain(ch)
        if np.isfinite(g[0]):
            w(f"| {names[ch][0]} | {names[ch][1]} | {pc(g[0])}（{pc(g[1])} – {pc(g[2])}） | {'值得做（上限 ≥ 10%）' if g[0] >= 0.10 else '**不值得做**（上限 < 10%）'} |")
    w("")
    w("逐格結果（基準 = 同一格、同設定、有 graph 的 budget 執行，3 次中位數；括號為四分位距）：")
    w("")
    w("| 作弊版 | 任務 | B | 設定 | 基準 tok/s | 作弊 tok/s | 提升 | 接受率 | 備註 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in G:
        note = ""
        if r["cheat"] == "head":
            note = f"重播缺值 {pc(r.get('replay_missing_share', np.nan))}（只發生在已結束請求的最後一輪）"
        elif r["cheat"] == "oracle":
            note = f"oracle 涵蓋率 {pc(r.get('oracle_coverage', np.nan), 0)}"
        w(f"| {r['cheat']} | {SHORT[r['task']]} | {r['B']} | {r['config']} | {r['base_tok_s']:.0f}（{r['base_tok_s_q1']:.0f}–{r['base_tok_s_q3']:.0f}） | "
          f"{r['cheat_tok_s']:.0f}（{r['cheat_tok_s_q1']:.0f}–{r['cheat_tok_s_q3']:.0f}） | {pc(r['gain'])} | {r['acceptance']:.3f} | {note} |")
    w("")

    # ---------------- A
    w("## 3. 觀察")
    w("")
    w("### A. 時間預算（1B，實機）")
    w("")
    w("![有 graph](A_budget_graph.png)")
    w("")
    w("![無 graph](A_budget_eager.png)")
    w("")
    w("每確認 1 個 token 的牆鐘時間拆解（ms，3 次中位數）。最佳 γ = 同格 γ ∈ {2,3,4,6,8} 中牆鐘吞吐量最高者。")
    w("")
    comp = ["sh_layers", "sh_attn_kv", "sh_head", "dp_layers", "dp_attn_kv", "dp_head", "accept_readback", "metadata_upload", "host_other", "prefill"]
    hdr = ["淺層 k 層", "淺層 KV", "淺層 head", "深層 L−k 層", "深層 KV", "深層 head", "接受+讀回", "metadata", "host", "prefill"]
    for mode in ("graph", "eager"):
        w(f"**{'有 CUDA Graph' if mode == 'graph' else '沒有 graph'}**")
        w("")
        w("| 任務 | B | 設定 | tok/s（IQR） | ms/token | " + " | ".join(hdr) + " |")
        w("|---|---|---|---|---|" + "---|" * len(hdr))
        for r in A:
            if r["mode"] != mode or not (r["config"] in ("ar", "m1") or r.get("best_gamma") == "True"):
                continue
            wl = f(r["wall_ms_per_token"])
            w(f"| {SHORT[r['task']]} | {r['B']} | {r['config']} | {f(r['wall_tok_s']):.0f}（{f(r['wall_tok_s_q1']):.0f}–{f(r['wall_tok_s_q3']):.0f}） | {wl:.3f} | "
              + " | ".join(pc(f(r[c]) / wl, 0) for c in comp) + " |")
        w("")
    lc = stat("launch")
    w(f"這代表：有 graph 時，淺層 LM head 比淺層 3 層本身還貴（每個草稿都乘 128k × 2048 的 lm_head）；B ≥ 16 時 prefill 是最大的單一塊；"
      f"host 開銷只剩約 {pc(stat('host')[0])}。沒有 graph 時，每 token 時間中有 {pc(lc[0])}（{pc(lc[1])}–{pc(lc[2])}）在有 graph 時消失，這就是 kernel 啟動開銷。"
      "AR 的「深層」欄是全部 L 層。KV 讀取用同一 segment 在 L = 64 與最大 L 兩點實測的斜率推得；無 graph 時 launch-bound，斜率量到約 0。")
    w("")

    # ---------------- B
    w("### B. 浪費的拆解")
    w("")
    w("1B 實機回合記錄（有 graph，3 次中位數）：")
    w("")
    w("| 任務 | B | 設定 | 淺層時間花在被拒草稿 | 草稿全接受卻停下的輪次 | 深層被拒 token（時間） | 每輪淺層 step（每 slot） |")
    w("|---|---|---|---|---|---|---|")
    bestcfg = {(r["task"], r["B"]): r["config"] for r in best}
    for r in W:
        if r["mode"] == "graph" and (r["config"] == "m1" or bestcfg.get((r["task"], r["B"])) == r["config"]):
            w(f"| {SHORT[r['task']]} | {r['B']} | {r['config']} | {pc(f(r['shallow_time_on_rejected_drafts']))} | {pc(f(r['full_accept_round_share']))} | "
              f"{pc(f(r['rejected_share_of_deep_tokens']))}（{pc(f(r['rejected_share_of_deep_time']))}） | {f(r['shallow_steps_per_round_slot']):.2f} |")
    w("")
    w("AR 參考輸出上的同步 γ 回合（8B 與 1B，四個任務；輸出型）：")
    w("")
    w("| 模型 | 任務 | γ | 被拒草稿 | 全接受卻停下的輪次 | 深層被拒 token | 每輪確認 |")
    w("|---|---|---|---|---|---|---|")
    for r in Wr:
        if r["gamma"] in ("2", "4", "8"):
            w(f"| {r['model'][-2:]} | {SHORT[r['task']]} | {r['gamma']} | {pc(f(r['rejected_draft_share']))} | {pc(f(r['full_accept_round_share']))} | "
              f"{pc(f(r['rejected_share_of_deep_tokens']))} | {f(r['tokens_per_round']):.2f} |")
    w("")
    wg = [r for r in W if r["mode"] == "graph"]
    shw = [f(r["shallow_time_on_rejected_drafts"]) for r in wg]
    fa = [f(r["full_accept_round_share"]) for r in wg]
    r8 = [f(r["rejected_draft_share"]) for r in Wr if "8B" in r["model"] and r["gamma"] == "4"]
    r1 = [f(r["rejected_draft_share"]) for r in Wr if "1B" in r["model"] and r["gamma"] == "4"]
    w(f"這代表：同步 γ 同時有兩種浪費（範圍涵蓋所有 γ 與 M1、兩個任務、所有 B）——{pc(min(shw), 0)}–{pc(max(shw), 0)} 的淺層時間花在之後被拒的草稿上，同時又有 {pc(min(fa), 0)}–{pc(max(fa), 0)} 的輪次"
      f"在草稿全對時停下；γ 調小減少前者、增加後者，單一 γ 無法同時減少兩者（M1 也一樣）。γ = 4 時被拒草稿 8B 為 {pc(min(r8), 0)}–{pc(max(r8), 0)}，1B 為 {pc(min(r1), 0)}–{pc(max(r1), 0)}。")
    w("")

    # ---------------- C
    w("### C. 接受的結構（輸出型，exit 3）")
    w("")
    w("![C](C_structure.png)")
    w("")
    w("| 模型 | 任務 | α（exit 3） | 位置 1 / 2 / 4 / 8 的條件接受率 | top-1 ≥ 0.5 的接受率（占比） | < 0.5 的接受率 | ECE | lag-1 自相關 | 請求平均接受率 p10–p90 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in Cs:
        key = (r["model"], r["task"])
        a3 = next((f(x["alpha"]) for x in Ce if (x["model"], x["task"]) == key and x["k"] == "3"), np.nan)
        ps = {x["position"]: f(x["accept_rate"]) for x in Cp if (x["model"], x["task"]) == key}
        l1 = next((f(x["autocorr"]) for x in Ca if (x["model"], x["task"]) == key and x["lag"] == "1"), np.nan)
        rq = next((x for x in Cr if (x["model"], x["task"]) == key), None)
        w(f"| {r['model'][-2:]} | {SHORT[r['task']]} | {a3:.3f} | " + " / ".join(pc(ps.get(str(j), np.nan), 0) for j in (1, 2, 4, 8))
          + f" | {pc(f(r['accept_p1_ge_05']), 0)}（{pc(f(r['share_p1_ge_05']), 0)}） | {pc(f(r['accept_p1_lt_05']), 0)} | {f(r['ece_top1']):.3f} | {l1:.2f} | "
          + (f"{f(rq['p10']):.2f}–{f(rq['p90']):.2f}" if rq else "–") + " |")
    w("")
    w("token 類型（AR token 的類別 → 接受率；占比）：")
    w("")
    w("| 模型 | 任務 | 一般詞 | 標點 | 數字 | 空白/換行 | 出現在 prompt | 不在 prompt |")
    w("|---|---|---|---|---|---|---|---|")
    for key in sorted({(r["model"], r["task"]) for r in Ct}):
        d = {x["value"]: x for x in Ct if (x["model"], x["task"]) == key}
        cell = lambda v: f"{pc(f(d[v]['accept_rate']), 0)}（{pc(f(d[v]['share']), 0)}）" if v in d else "–"  # noqa: E731
        w(f"| {key[0][-2:]} | {SHORT[key[1]]} | {cell('word')} | {cell('punctuation')} | {cell('digit')} | {cell('whitespace')} | {cell('in_prompt')} | {cell('not_in_prompt')} |")
    w("")
    w("退出層 k = 2..8 的 α：")
    w("")
    w("| 模型 | 任務 | " + " | ".join(f"k={k}" for k in range(2, 9)) + " |")
    w("|---|---|" + "---|" * 7)
    for key in sorted({(r["model"], r["task"]) for r in Ce}):
        d = {x["k"]: f(x["alpha"]) for x in Ce if (x["model"], x["task"]) == key}
        w(f"| {key[0][-2:]} | {SHORT[key[1]]} | " + " | ".join(f"{d.get(str(k), np.nan):.3f}" for k in range(2, 9)) + " |")
    w("")
    l1all = [f(r["autocorr"]) for r in Ca if r["lag"] == "1"]
    hetall = [(f(r["p10"]), f(r["p90"])) for r in Cr]
    w(f"這代表：接受不是均勻的。前面都接受時，越後面的草稿條件接受率越高；exit 的 top-1 機率能把 token 分成高接受（≥ 0.5：{pc(acc_hi[0], 0)}–{pc(acc_hi[1], 0)}，8B）"
      f"與低接受（< 0.5：{pc(acc_lo[0], 0)}–{pc(acc_lo[1], 0)}）兩群；請求間差異大（請求平均接受率 p10–p90 跨 {min(h[0] for h in hetall):.2f}–{max(h[1] for h in hetall):.2f}），"
      f"請求內的自相關很弱（lag-1 為 {min(l1all):.2f}–{max(l1all):.2f}）；出現在 prompt 中的 token 在每個任務都較容易接受，8B 在 gsm8k 的數字 token 只有 "
      + next((pc(f(x["accept_rate"]), 0) for x in Ct if "8B" in x["model"] and x["task"] == "gsm8k" and x["value"] == "digit"), "–")
      + " 被接受。exit 6 的 α 比 exit 3 高 0.06–0.09，但 exit 2 很差。")
    w("")

    # ---------------- D
    w("### D. 空位（1B，實機記錄）")
    w("")
    w(f"knee_measured = {knee:.0f} tokens（這台 GPU 的校準）。")
    w("")
    w("| 任務 | B | 設定 | 淺層段 token 中位數 | 深層段 token 中位數（p10–p90） | 深層空位中位數（p10–p90） | 深層超過膝點 |")
    w("|---|---|---|---|---|---|---|")
    for r in D:
        if r["mode"] != "graph" or r["config"] not in ("fixed-g4", "m1") or r["segment"] != "deep_segment":
            continue
        shr = next((x for x in D if x["mode"] == "graph" and x["config"] == r["config"] and x["task"] == r["task"] and x["B"] == r["B"]
                    and x["segment"] == "shallow_segment"), None)
        w(f"| {SHORT[r['task']]} | {r['B']} | {r['config']} | {f(shr['n_median']) if shr else np.nan:.0f} | {f(r['n_median']):.0f}（{f(r['n_p10']):.0f}–{f(r['n_p90']):.0f}） | "
          f"{f(r['slack_median']):.0f}（{f(r['slack_p10']):.0f}–{f(r['slack_p90']):.0f}） | {pc(f(r['share_over_knee']), 0)} |")
    w("")
    w("這代表：淺層段永遠只有 B 個 token，離膝點很遠；深層段在 B ≤ 16 時也有幾十到一百多個 token 的空位，B = 32 + γ = 4 才超過膝點。")
    w("")

    # ---------------- E
    w("### E. 免費猜測的命中率（輸出型；只在草稿全接受的輪次後接猜測，同步 γ = 4）")
    w("")
    w("| 模型 | 任務 | 來源 | G | 全接受輪次 | 每輪多確認 | 排除重複迴圈請求後 |")
    w("|---|---|---|---|---|---|---|")
    for r in E:
        if r["G"] in ("4", "8", "16"):
            w(f"| {r['model'][-2:]} | {SHORT[r['task']]} | {r['source']} | {r['G']} | {pc(f(r['full_accept_share']), 0)} | {f(r['extra_per_round']):.3f} | "
              f"{'–' if not np.isfinite(f(r.get('extra_per_round_excl_repetitive'))) else format(f(r.get('extra_per_round_excl_repetitive')), '.3f')} |")
    for r in Et:
        w(f"| 1B | {SHORT[r['task']]} | 被拒尾巴（實機 fixed-g4 B = 16） | ≤ 3 | – | {f(r['extra_per_round']):.3f} | – |")
    w("")
    ng8 = {(r["model"][-2:], SHORT[r["task"]]): (f(r["extra_per_round"]), f(r["extra_per_round_excl_repetitive"]))
           for r in E if r["source"] == "ngram" and r["G"] == "8"}
    xr = [f(r["extra_per_round"]) for r in E if r["source"] != "ngram"]
    w("這代表：n-gram（prompt + 已生成文字）在 cnn_dm 最高（G = 8：8B " + f"{ng8[('8B', 'cnn_dm')][0]:.2f}" + " 個/輪，排除重複迴圈請求後 "
      + f"{ng8[('8B', 'cnn_dm')][1]:.2f}）；gsm8k 的命中大多來自重複迴圈（1B {ng8[('1B', 'gsm8k')][0]:.2f} → {ng8[('1B', 'gsm8k')][1]:.2f}）。"
      f"跨請求每輪最多 {max(xr):.2f} 個，被拒尾巴最多 {max(f(r['extra_per_round']) for r in Et):.3f} 個。注意只有草稿全接受的輪次才能用到猜測，8B 在 gsm8k 只占 "
      + next((pc(f(r["full_accept_share"]), 0) for r in E if "8B" in r["model"] and r["task"] == "gsm8k" and r["source"] == "ngram"), "–") + "。")
    w("")

    # ---------------- F
    w("### F. 批次詞彙覆蓋率（輸出型，exit 3）")
    w("")
    w("每個 B 列出四個任務中最差任務的 α_V/α 下界最高的設定（相同時取最大 |V| 較小者）。下界把 top-64 無法判定的位置算成不一致，上界算成一致；完整表在 F_coverage.csv。")
    w("")
    w("| 模型 | B | R / K / F | 四任務最大 |V| | 最差任務 α_V/α（下界–上界） | 無法判定比例（最大） |")
    w("|---|---|---|---|---|---|")
    fr = {}
    for model in sorted({r["model"] for r in F}):
        for B in sorted({int(r["B"]) for r in F}):
            b_ = frow(model, B)
            fr[(model, B)] = b_
            if b_:
                w(f"| {model[-2:]} | {B} | {'/'.join(b_[3])} | {b_[2]} | {b_[0]:.3f}–{b_[1]:.3f} | {pc(b_[4], 2)} |")
    w("")
    w("這代表：批次內容只用到詞彙表的一小部分（8B、B = 32 最多 "
      + (str(fr[("layerskip-llama3-8B", 32)][2]) if fr.get(("layerskip-llama3-8B", 32)) else "–")
      + " 個 token），exit head 限制在聯集詞彙上的接受率損失最多 "
      + pc(1 - min(v[0] for v in fr.values() if v), 1) + "（下界）；上界 ≥ 1 表示也可能完全不損失，差距來自 top-64 無法判定的位置（最多 "
      + pc(max(v[4] for v in fr.values() if v), 1) + "）。")
    w("")

    w("## 4. 重現與注意事項")
    w("")
    w("- 指令：`python observation/v3/budget.py run`（420 次）→ `budget.py tails` → `budget.py analyze`；`python observation/v3/outputs.py --models layerskip-llama3-8B layerskip-llama3-1B`；"
      "`python observation/v3/cheats.py run`；`python observation/v3/summary.py`。8B 參考輸出：`python -m solutions.make_ref --model layerskip-llama3-8B --stream-weights "
      "--topk-layers 2 3 4 5 6 7 8 --freq-prompts 64 --length-mode auto --batch-size 32 --out results/observe-4060/ref`（1B 同指令不加 --stream-weights）。")
    w("- 原始 CSV：A_budget(_runs).csv、B_waste_1B.csv、B_waste_ref.csv、C_*.csv、D_slack.csv、E_guess.csv、E_tail.csv、F_coverage.csv、G_cheats.csv、ranking(_cells).csv；"
      "每次執行的 JSON 在 runs/、cheats/、tails/（不進 git）。")
    w("- 無損：fp32 下 24 種開關組合 + AR 在有 graph 時都和 AR 100% 一致（lossless/layerskip-llama3-1B_fp32_graphs.md）；8B 的串流路徑在 1B 上與一般路徑逐 token 相同（check_streamed_layerskip-llama3-1B.json）。")
    w("- 時間型數據只有 1B（8 GB 放不下 8B）；8B 只做輸出型分析，串流執行的時間沒有使用。")
    w("- 各塊的時間是 CUDA event 在 GPU 時間軸上的區間，包含 GPU 等待 launch 的空檔；有 graph 時這個空檔很小。")
    w("- 「淺層花在被拒草稿」「深層被拒 token」按 token 數比例分攤 step 時間；step 在膝點以下時時間幾乎不隨 token 數變，所以這兩列的理想上限是高估，第 2 節的實驗才是實測上限。")
    w("- cyclic 協定會重複使用 64 個 prompt，prefix cache 作弊版因此能全部命中；真實服務只能命中共用前綴。")
    w("- 背景程式（Chrome、Discord、Steam、LINE）在計時期間執行中。")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[summary] wrote {OUT / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
