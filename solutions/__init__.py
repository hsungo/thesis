"""EESD: batched early-exit self-speculative decoding with batch-level deep-layer scheduling.

    engine.py        merged shallow/deep engine (packed tokens, KV pointer rollback, cyclic/static/Poisson drivers)
    packed.py        ragged token packing (qo_indptr), per-bucket attention, no GEMM padding
    cost_model.py    portable roofline cost model + ASPIRE-style linear model
    schedulers.py    --sched fixed | aspire | m1
    exit_select.py   --exit m2   (batch-shared dynamic exit layer)
    prefill.py       --prefill separate | m3 (depth-split piggyback prefill)
    calibrate.py     GPU calibration of the cost model + MAPE against measured steps
    run_ablation.py  measurements / ablations (CSV + summary tables)
"""
