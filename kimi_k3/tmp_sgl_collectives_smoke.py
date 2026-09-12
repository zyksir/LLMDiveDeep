#!/usr/bin/env python3
"""8-rank smoke test of the sgl Collectives backend (first HW exercise).

Explicitly exercises ``all_reduce`` with ``impl="sgl:push_res"`` and
``impl="sgl:pull_res"`` (H=7168) and ``allreduce_norm`` with
``impl="sgl:push_norm"`` (H=3584) on random bf16 tensors, comparing each
against an fp32-accumulated torch.distributed reference.  Tolerance is a
few bf16 ulps of the reference magnitude (the sgl kernels accumulate in
higher precision but round through bf16 at least once).

Run:  mpirun -n 8 --allow-run-as-root python3 kimi_k3/tmp_sgl_collectives_smoke.py
"""
import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    if world == 1:
        raise SystemExit("run under mpirun -n 8")
    torch.cuda.set_device(int(os.environ.get(
        "OMPI_COMM_WORLD_LOCAL_RANK", rank)) % torch.cuda.device_count())
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world,
        device_id=torch.device("cuda", torch.cuda.current_device()))

    from communication.collective import Collectives

    tokens, h_ar, h_norm = 4, 7168, 3584
    col = Collectives(
        dist.group.WORLD,
        max_numel=tokens * h_ar,
        max_hidden=h_ar,
        enable_flashinfer=False,
        enable_trt=False,
    )

    torch.manual_seed(1234 + rank)
    dev = torch.device("cuda", torch.cuda.current_device())
    failures = []

    def ref_allreduce_f32(x: torch.Tensor) -> torch.Tensor:
        r = x.float().clone()
        dist.all_reduce(r)
        return r

    def tol_of(ref_f32: torch.Tensor, ulps: float = 4.0) -> float:
        # bf16 has 8 mantissa bits -> ulp(|v|) ~ |v| * 2^-8
        return ulps * ref_f32.abs().max().item() * 2.0 ** -8

    # --- all_reduce: sgl:push_res / sgl:pull_res @ H=7168 ---
    for impl in ("sgl:push_res", "sgl:pull_res"):
        x = torch.randn(tokens, h_ar, device=dev, dtype=torch.bfloat16)
        ref = ref_allreduce_f32(x)
        try:
            out = col.all_reduce(x.clone(), impl=impl).clone()
            torch.cuda.synchronize()
            diff = (out.float() - ref).abs().max().item()
            tol = tol_of(ref)
            ok = diff <= tol
        except Exception as exc:  # report, don't hide
            diff, tol, ok = float("nan"), float("nan"), False
            if rank == 0:
                import traceback
                traceback.print_exc()
                print(f"[smoke] {impl}: EXCEPTION {exc!r}")
        if not ok:
            failures.append(impl)
        if rank == 0:
            print(f"[smoke] all_reduce {impl:14s} H={h_ar} "
                  f"max_abs={diff:.5f} tol={tol:.5f} "
                  f"{'PASS' if ok else 'FAIL'}")

    # --- allreduce_norm: sgl:push_norm @ H=3584, with residual ---
    x = torch.randn(tokens, h_norm, device=dev, dtype=torch.bfloat16)
    gamma = torch.randn(h_norm, device=dev, dtype=torch.bfloat16)
    residual = torch.randn(tokens, h_norm, device=dev, dtype=torch.bfloat16)
    eps = 1e-5
    red = ref_allreduce_f32(x)
    new_res_ref = red + residual.float()
    norm_ref = torch.nn.functional.rms_norm(
        new_res_ref, (h_norm,), gamma.float(), eps)
    try:
        norm, new_res = col.allreduce_norm(
            x.clone(), gamma, eps, residual=residual.clone(),
            impl="sgl:push_norm")
        norm, new_res = norm.clone(), new_res.clone()
        torch.cuda.synchronize()
        d_norm = (norm.float() - norm_ref).abs().max().item()
        d_res = (new_res.float() - new_res_ref).abs().max().item()
        # norm output passes through gamma-scaled normalization; a couple
        # extra roundings -> allow a few more ulps of its magnitude
        t_norm = tol_of(norm_ref, ulps=8.0)
        t_res = tol_of(new_res_ref)
        ok = d_norm <= t_norm and d_res <= t_res
    except Exception as exc:
        d_norm = d_res = t_norm = t_res = float("nan")
        ok = False
        if rank == 0:
            import traceback
            traceback.print_exc()
            print(f"[smoke] sgl:push_norm: EXCEPTION {exc!r}")
    if not ok:
        failures.append("sgl:push_norm")
    if rank == 0:
        print(f"[smoke] allreduce_norm sgl:push_norm H={h_norm} "
              f"norm max_abs={d_norm:.5f} tol={t_norm:.5f} | "
              f"residual max_abs={d_res:.5f} tol={t_res:.5f} "
              f"{'PASS' if ok else 'FAIL'}")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"[smoke] RESULT: "
              f"{'ALL PASS' if not failures else 'FAILED: ' + ','.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
