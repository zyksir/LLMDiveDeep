"""Benchmark the causal window pattern with the same kernel adapters."""

from .bench_dense_attention import main


if __name__ == "__main__":
    main(windowed=True)
