"""Vendored sglang MoE routing/quant fronts (see package docstring).

The two ``covered`` predicates are re-exported under distinct names so both
kernels share one flat adapter namespace.
"""

from ..sgl_copied_kernels.ops.moe.moe_route_quant_fused import (
    covered as route_quant_fused_covered,
)
from ..sgl_copied_kernels.ops.moe.moe_route_quant_fused import route_quant_fused
from ..sgl_copied_kernels.ops.moe.moe_route_radix import (
    covered as route_radix_covered,
)
from ..sgl_copied_kernels.ops.moe.moe_route_radix import route_radix

__all__ = [
    "route_quant_fused",
    "route_quant_fused_covered",
    "route_radix",
    "route_radix_covered",
]
