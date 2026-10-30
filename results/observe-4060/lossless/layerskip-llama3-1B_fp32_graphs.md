# Losslessness vs AR(B=1): layerskip-llama3-1B, fp32, 32 prompts, max_new 96

| config | identical outputs | mean prefix agreement | k switches | piggybacked prompt tokens | graph steps |
|---|---|---|---|---|---|
| ar B=8 (floor) | 100.0% | 100.0% |  |  |  |
| fixed-g4-k3-separate [roofline] | 100.0% | 100.0% | 0 | 0 | 100% |
| fixed-g4-k3-separate [linear] | 100.0% | 100.0% | 0 | 0 | 100% |
| fixed-g4-k3-m3 [roofline] | 100.0% | 100.0% | 0 | 4096 | 59% |
| fixed-g4-k3-m3 [linear] | 100.0% | 100.0% | 0 | 7936 | 81% |
| fixed-g4-m2-separate [roofline] | 100.0% | 100.0% | 0 | 0 | 95% |
| fixed-g4-m2-separate [linear] | 100.0% | 100.0% | 0 | 0 | 95% |
| fixed-g4-m2-m3 [roofline] | 100.0% | 100.0% | 0 | 4096 | 56% |
| fixed-g4-m2-m3 [linear] | 100.0% | 100.0% | 0 | 7936 | 78% |
| aspire-k3-separate [roofline] | 100.0% | 100.0% | 0 | 0 | 100% |
| aspire-k3-separate [linear] | 100.0% | 100.0% | 0 | 0 | 100% |
| aspire-k3-m3 [roofline] | 100.0% | 100.0% | 0 | 6400 | 78% |
| aspire-k3-m3 [linear] | 100.0% | 100.0% | 0 | 7936 | 92% |
| aspire-m2-separate [roofline] | 100.0% | 100.0% | 11 | 0 | 83% |
| aspire-m2-separate [linear] | 100.0% | 100.0% | 7 | 0 | 81% |
| aspire-m2-m3 [roofline] | 100.0% | 100.0% | 5 | 5376 | 63% |
| aspire-m2-m3 [linear] | 100.0% | 100.0% | 11 | 7680 | 72% |
| m1-k3-separate [roofline] | 100.0% | 100.0% | 0 | 0 | 100% |
| m1-k3-separate [linear] | 100.0% | 100.0% | 0 | 0 | 100% |
| m1-k3-m3 [roofline] | 100.0% | 100.0% | 0 | 6144 | 56% |
| m1-k3-m3 [linear] | 100.0% | 100.0% | 0 | 7936 | 82% |
| m1-m2-separate [roofline] | 100.0% | 100.0% | 1 | 0 | 93% |
| m1-m2-separate [linear] | 100.0% | 100.0% | 1 | 0 | 93% |
| m1-m2-m3 [roofline] | 100.0% | 100.0% | 0 | 6144 | 52% |
| m1-m2-m3 [linear] | 100.0% | 100.0% | 2 | 7936 | 79% |
| ar [graphs] | 100.0% | 100.0% |  |  |  |
