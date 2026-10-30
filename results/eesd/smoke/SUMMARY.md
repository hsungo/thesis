# EESD measurements (real GPU runs)

Three throughputs per run: **decode** = tokens / decode-step GPU time (CUDA events, prefill excluded, ASPIRE protocol); **e2e** = tokens / (decode + prefill GPU time); **wall** = tokens / perf_counter wall time of the measured window (includes host work between steps).  Each cell gives the speedup vs AR of the same model, task, prompt length, B and repetition as `decode / e2e / wall` (mean over repetitions; std in summary.csv).  host = 1 - GPU time / wall time.

## layerskip-llama3-1B · gsm8k · prompt 256

| config | B=4 speedup d / e / w | decode tok/s (B=max) | e2e tok/s | wall tok/s | host share | deep/token | k | plan ms/step |
|---|---|---|---|---|---|---|---|---|
| all | 1.18 / 1.27 / 1.25 | 177.4 | 177.4 | 167.5 | 0.056 | 0.1039 | 3 | 0.640 |
| ar | 1.00 / 1.00 / 1.00 | 150.0 | 139.8 | 133.5 | 0.045 | 0.2500 | 3 | 0.000 |
| aspire | 0.66 / 0.67 / 0.66 | 98.9 | 93.0 | 88.6 | 0.047 | 0.2834 | 3 | 0.465 |
| fixed-g4 | 1.36 / 1.32 / 1.34 | 203.5 | 184.6 | 178.5 | 0.033 | 0.0867 | 3 | 0.058 |
| m1 | 1.27 / 1.22 / 1.24 | 190.0 | 171.1 | 165.3 | 0.034 | 0.1070 | 3 | 0.351 |
| m2 | 1.37 / 1.36 / 1.35 | 206.0 | 190.7 | 180.5 | 0.054 | 0.0798 | 3 | 0.329 |
| m3 | 1.13 / 1.21 / 1.22 | 169.4 | 169.4 | 163.5 | 0.035 | 0.0953 | 3 | 0.072 |
