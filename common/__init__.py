"""Shared helpers for the LLMDiveDeep micro-benchmarks.

Small, dependency-light utilities that are reused across the `sparse_attn`
and `quantization` benchmark suites. Kept deliberately tiny so any bench
script can `from common.bench_utils import bench_function, print_section`
without pulling in heavy imports.
"""

from common.bench_utils import bench_function, print_section

__all__ = ["bench_function", "print_section"]
