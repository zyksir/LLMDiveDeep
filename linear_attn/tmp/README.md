# tmp — one-off analysis scripts, safe to delete

- `profile_kda_stage_kernels.py`: torch.profiler per-kernel timing comparison
  of FlashKDA vs SGLang prefill stages (findings absorbed into KDA.md).
- `bench_flashkda_k1k2.py`: FLOP/byte roofline classification of FlashKDA's
  K1 (prepare, memory-bound) and K2 (recurrence, latency/parallelism-bound)
  kernels (findings absorbed into KDA.md and kda_prefill_spec.md).
