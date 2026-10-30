# Losslessness vs AR(B=1): layerskip-llama3-1B, fp32, 32 prompts, max_new 96

| config | identical outputs | mean prefix agreement | k switches | piggybacked prompt tokens |
|---|---|---|---|---|
| ar B=8 (floor) | 100.0% | 100.0% |  |  |
| fixed-g4-k3-separate [roofline] | 100.0% | 100.0% | 0 | 0 |
| fixed-g4-k3-separate [linear] | 100.0% | 100.0% | 0 | 0 |
| fixed-g4-k3-m3 [roofline] | 100.0% | 100.0% | 0 | 4096 |
| fixed-g4-k3-m3 [linear] | 100.0% | 100.0% | 0 | 7936 |
| fixed-g4-m2-separate [roofline] | 100.0% | 100.0% | 0 | 0 |
| fixed-g4-m2-separate [linear] | 100.0% | 100.0% | 0 | 0 |
| fixed-g4-m2-m3 [roofline] | 100.0% | 100.0% | 0 | 4096 |
| fixed-g4-m2-m3 [linear] | 100.0% | 100.0% | 0 | 7936 |
| aspire-k3-separate [roofline] | 100.0% | 100.0% | 0 | 0 |
| aspire-k3-separate [linear] | 100.0% | 100.0% | 0 | 0 |
| aspire-k3-m3 [roofline] | 100.0% | 100.0% | 0 | 6400 |
| aspire-k3-m3 [linear] | 100.0% | 100.0% | 0 | 7936 |
| aspire-m2-separate [roofline] | 100.0% | 100.0% | 11 | 0 |
| aspire-m2-separate [linear] | 100.0% | 100.0% | 7 | 0 |
| aspire-m2-m3 [roofline] | 100.0% | 100.0% | 5 | 5376 |
| aspire-m2-m3 [linear] | 100.0% | 100.0% | 11 | 7680 |
| m1-k3-separate [roofline] | 100.0% | 100.0% | 0 | 0 |
| m1-k3-separate [linear] | 100.0% | 100.0% | 0 | 0 |
| m1-k3-m3 [roofline] | 100.0% | 100.0% | 0 | 6144 |
| m1-k3-m3 [linear] | 100.0% | 100.0% | 0 | 7936 |
| m1-m2-separate [roofline] | 100.0% | 100.0% | 1 | 0 |
| m1-m2-separate [linear] | 100.0% | 100.0% | 1 | 0 |
| m1-m2-m3 [roofline] | 100.0% | 100.0% | 0 | 6144 |
| m1-m2-m3 [linear] | 100.0% | 100.0% | 2 | 7936 |
