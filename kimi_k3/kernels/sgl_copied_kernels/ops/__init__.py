"""Vendoring glue (NOT a copy of sglang's ``kernels/ops/__init__.py``).

The upstream ``ops`` package eagerly imports every operator group to populate
a kernel registry; the vendored subset has no registry, so this stays empty
and callers import the specific group modules they need.
"""
