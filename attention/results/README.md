# Generated attention results

All future benchmark output belongs below this directory:

```
results/
  dense/
  sliding_window/
  compressed_sparse/
  kda/
  linear/
  legacy/
```

No measurements were generated during tutorial preparation. New benchmark
drivers require `--run`; older harnesses still use their original invocation.
CSV rows in the new drivers identify shape, precision, backend, environment,
timing boundary, correctness and failure status. A skipped/failed backend has
no valid timing. `linear/` holds the retained GDN/general-linear harness output;
`legacy/` separates full-layer/projection studies from attention-core results.

Cleanup removed 46 CSV/log/PNG files and one local_results JSON trace, plus
the empty old result directories. They are recoverable from the temporary
archive `/tmp/attention-artifacts-backup-IjjWX9/attention-generated-artifacts.tar.gz`.
This archive is outside the repository and may be lost when temporary storage
is cleared. Historical timing claims inside older notes/source comments are
not a current ranking and should be regenerated before use.

Keep manually redirected logs, plots, profiler traces, and custom outputs in
the corresponding subdirectory too. New drivers enforce this for their CSV
paths; legacy scripts only have their defaults redirected.
