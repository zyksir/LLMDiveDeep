# debug/ — throwaway probes behind the Aug-5 campaign conclusions

Each script is single-GPU, self-contained, and referenced from
`kimi_k3_layer/moe_optimization.md` / `kda_optimization.md` as the
evidence for a specific decision. Safe to delete wholesale once the
docs stop citing them.

| script | question it answered | verdict it produced |
|---|---|---|
| `shared_gemm_roofline.py` | are the shared/input GEMMs at the HBM roofline? | narrow-N cuBLAS runs ~2 TB/s, wide-N ~8 → motivated merge3 |
| `route_warp_tune.py` | is route_pack's `num_warps=4` still optimal? | yes (6.0–6.3 us; 2/8/16 all worse) |
| `base_sweep.sh` | which baseline knobs make the BASELINE fastest? | AllReduce ONESHOT (honest baseline); TWOSHOT pathological |
| `kda_prefill_gate_debug.py` | is the 0.90 layer cosine an opt bug? | no — b10 kernel 0.999993 vs oracle; the TRT chunk pipeline carries the deviation |
| `kda_decode_gemm_swap.py` | can the CuTeDSL tall-GEMM beat cuBLAS on the KDA projections? | no (cuBLAS ~7.5 TB/s standalone; trace numbers were PDL-inflated) — only B=1 in_proj wins +2.7 us |
