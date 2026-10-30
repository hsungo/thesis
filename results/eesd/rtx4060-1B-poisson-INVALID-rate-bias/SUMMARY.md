# EESD measurements (real GPU runs)

Three throughputs per run: **decode** = tokens / decode-step GPU time (CUDA events, prefill excluded, ASPIRE protocol); **e2e** = tokens / (decode + prefill GPU time); **wall** = tokens / perf_counter wall time of the measured window (includes host work between steps).  Each cell gives the speedup vs AR of the same model, task, prompt length, B and repetition as `decode / e2e / wall` (mean over repetitions; std in summary.csv).  host = 1 - GPU time / wall time.

## layerskip-llama3-1B · cnn_dm_summarization · prompt 512

| config | B=32 speedup d / e / w | decode tok/s (B=max) | e2e tok/s | wall tok/s | host share | deep/token | k | plan ms/step |
|---|---|---|---|---|---|---|---|---|
| all | 1.51 / 1.37 / 1.39 | 1858.4 | 1351.9 | 1279.3 | 0.054 | 0.0108 | 3 | 0.592 |
| ar | 1.00 / 1.00 / 1.00 | 1234.3 | 984.2 | 922.4 | 0.063 | 0.0312 | 3 | 0.000 |
| aspire | 0.43 / 0.49 / 0.50 | 533.1 | 481.2 | 462.0 | 0.040 | 0.0405 | 3 | 0.456 |
| fixed-g2 | 1.36 / 1.27 / 1.30 | 1679.5 | 1250.9 | 1197.6 | 0.043 | 0.0127 | 3 | 0.056 |
| fixed-g3 | 1.49 / 1.36 / 1.39 | 1845.1 | 1339.8 | 1285.5 | 0.041 | 0.0105 | 3 | 0.053 |
| fixed-g4 | 1.39 / 1.29 / 1.32 | 1719.0 | 1273.1 | 1221.3 | 0.041 | 0.0093 | 3 | 0.056 |
| fixed-g6 | 1.31 / 1.22 / 1.26 | 1618.7 | 1201.2 | 1159.4 | 0.035 | 0.0079 | 3 | 0.056 |
| m1 | 1.46 / 1.34 / 1.37 | 1800.0 | 1318.7 | 1266.6 | 0.040 | 0.0106 | 3 | 0.327 |
| m3 | 1.56 / 1.42 / 1.45 | 1930.3 | 1394.5 | 1341.9 | 0.038 | 0.0093 | 3 | 0.044 |

## layerskip-llama3-1B · gsm8k · prompt 512

| config | B=32 speedup d / e / w | decode tok/s (B=max) | e2e tok/s | wall tok/s | host share | deep/token | k | plan ms/step |
|---|---|---|---|---|---|---|---|---|
| all | 0.92 / 1.02 / 1.03 | 1087.6 | 577.3 | 559.2 | 0.031 | 0.0168 | 3 | 0.359 |
| ar | 1.00 / 1.00 / 1.00 | 1177.2 | 567.1 | 544.1 | 0.040 | 0.0312 | 3 | 0.000 |
| aspire | 0.46 / 0.65 / 0.66 | 545.6 | 369.7 | 357.0 | 0.034 | 0.0464 | 3 | 0.425 |
| fixed-g2 | 1.40 / 1.43 / 1.45 | 1649.3 | 811.3 | 786.8 | 0.030 | 0.0143 | 3 | 0.048 |
| fixed-g3 | 1.25 / 1.30 / 1.31 | 1469.5 | 736.6 | 715.0 | 0.029 | 0.0125 | 3 | 0.055 |
| fixed-g4 | 1.23 / 1.31 / 1.32 | 1443.8 | 741.2 | 719.6 | 0.029 | 0.0113 | 3 | 0.057 |
| fixed-g6 | 0.96 / 1.12 / 1.13 | 1132.3 | 633.9 | 616.2 | 0.028 | 0.0103 | 3 | 0.060 |
| m1 | 1.20 / 1.26 / 1.27 | 1412.6 | 712.1 | 690.8 | 0.030 | 0.0129 | 3 | 0.349 |
| m3 | 1.35 / 1.38 / 1.40 | 1585.4 | 781.0 | 759.6 | 0.027 | 0.0113 | 3 | 0.045 |
