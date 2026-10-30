# EESD measurements (real GPU runs)

Three throughputs per run: **decode** = tokens / decode-step GPU time (CUDA events, prefill excluded, ASPIRE protocol); **e2e** = tokens / (decode + prefill GPU time); **wall** = tokens / perf_counter wall time of the measured window (includes host work between steps).  Each cell gives the speedup vs AR of the same model, task, prompt length, B and repetition as `decode / e2e / wall` (mean over repetitions; std in summary.csv).  host = 1 - GPU time / wall time.

## layerskip-llama3-1B · cnn_dm_summarization · prompt 512

| config | B=32 speedup d / e / w | decode tok/s (B=max) | e2e tok/s | wall tok/s | host share | deep/token | k | plan ms/step |
|---|---|---|---|---|---|---|---|---|
| all | 1.31 / 1.23 / 1.24 | 1735.4 | 1283.1 | 1211.6 | 0.056 | 0.0108 | 3 | 0.648 |
| ar | 1.00 / 1.00 / 1.00 | 1320.6 | 1039.9 | 980.0 | 0.058 | 0.0312 | 3 | 0.000 |
| aspire | 0.39 / 0.45 / 0.46 | 515.5 | 466.7 | 447.9 | 0.040 | 0.0405 | 3 | 0.460 |
| fixed-g2 | 1.42 / 1.31 / 1.33 | 1872.6 | 1357.5 | 1301.7 | 0.041 | 0.0127 | 3 | 0.044 |
| fixed-g3 | 1.40 / 1.29 / 1.31 | 1847.6 | 1342.6 | 1288.1 | 0.041 | 0.0105 | 3 | 0.054 |
| fixed-g4 | 1.28 / 1.21 / 1.24 | 1695.8 | 1262.5 | 1213.3 | 0.039 | 0.0093 | 3 | 0.056 |
| fixed-g6 | 1.31 / 1.21 / 1.25 | 1725.0 | 1262.4 | 1221.2 | 0.033 | 0.0079 | 3 | 0.045 |
| m1 | 1.29 / 1.20 / 1.22 | 1701.1 | 1244.6 | 1194.8 | 0.040 | 0.0106 | 3 | 0.355 |
| m3 | 1.39 / 1.29 / 1.31 | 1830.1 | 1340.0 | 1287.3 | 0.039 | 0.0093 | 3 | 0.053 |

## layerskip-llama3-1B · gsm8k · prompt 512

| config | B=32 speedup d / e / w | decode tok/s (B=max) | e2e tok/s | wall tok/s | host share | deep/token | k | plan ms/step |
|---|---|---|---|---|---|---|---|---|
| all | 0.71 / 0.89 / 0.89 | 852.0 | 507.9 | 489.6 | 0.036 | 0.0169 | 3 | 0.443 |
| ar | 1.00 / 1.00 / 1.00 | 1207.8 | 570.2 | 547.2 | 0.040 | 0.0312 | 3 | 0.000 |
| aspire | 0.41 / 0.62 / 0.62 | 496.0 | 352.1 | 339.9 | 0.035 | 0.0465 | 3 | 0.456 |
| fixed-g2 | 1.29 / 1.38 / 1.40 | 1561.4 | 788.5 | 763.4 | 0.032 | 0.0143 | 3 | 0.056 |
| fixed-g3 | 1.24 / 1.23 / 1.25 | 1503.1 | 703.2 | 683.0 | 0.029 | 0.0127 | 3 | 0.055 |
| fixed-g4 | 1.18 / 1.29 / 1.31 | 1423.8 | 736.9 | 715.1 | 0.030 | 0.0113 | 3 | 0.056 |
| fixed-g6 | 0.96 / 1.13 / 1.14 | 1160.6 | 643.1 | 625.6 | 0.027 | 0.0103 | 3 | 0.062 |
| m1 | 1.20 / 1.26 / 1.28 | 1445.5 | 721.0 | 700.4 | 0.029 | 0.0129 | 3 | 0.347 |
| m3 | 1.19 / 1.30 / 1.31 | 1441.3 | 741.7 | 719.2 | 0.030 | 0.0113 | 3 | 0.059 |
