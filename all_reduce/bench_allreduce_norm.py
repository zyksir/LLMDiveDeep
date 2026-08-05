"""Allreduce + residual-add + RMSNorm sweep: TRT-LLM vs FlashInfer, Kimi K3 shape.

Kimi K3 decode does o_proj allreduce (hidden 7168, bf16) followed by a
residual add + RMSNorm. Today TRT-LLM runs them as two kernels
(AllReduceFusionOp.NONE, then norm); both TRT-LLM and FlashInfer also ship a
single fused kernel. The two fused kernels are the *same* kernel family --
FlashInfer's trtllm_allreduce_fusion is a port of TRT-LLM's
allReduceFusionKernels.cu (one-shot Lamport / two-shot) -- so this bench
answers (a) whose build of the one-shot kernel is faster and (b) what fusing
the norm is worth versus the unfused two-kernel pipeline.

Variants, grouped by what they compute:

  group "ar" (allreduce only; answers TRT one-shot vs FlashInfer one-shot):
    nccl          plain ncclAllReduce (baseline)
    trt_oneshot   torch.ops.trtllm.allreduce, strategy=ONESHOT, fusion NONE
    trt_twoshot   ... strategy=TWOSHOT (tokens >= tp)
    fi_oneshot    flashinfer trtllm_allreduce_fusion, kAllReduce, oneshot
    fi_twoshot    ... use_oneshot=False (tokens > tp)

  group "ar_norm" (allreduce + residual add + RMSNorm):
    nccl+norm         NCCL AR, then flashinfer fused_add_rmsnorm
    trt_oneshot+norm  TRT one-shot AR, then fused_add_rmsnorm (today's layout)
    trt_oneshot_fused torch.ops.trtllm.allreduce, ONESHOT, RESIDUAL_RMS_NORM
    trt_minlat_fused  ... strategy=MIN_LATENCY, RESIDUAL_RMS_NORM
    trt_twoshot_fused ... strategy=TWOSHOT, RESIDUAL_RMS_NORM (tokens >= tp)
    fi_oneshot_fused  flashinfer trtllm_allreduce_fusion, kARResidualRMSNorm
    fi_twoshot_fused  ... use_oneshot=False (tokens > tp)

Correctness: every variant is checked against an exact fp32 reference (gloo
CPU sum, then fp32 residual+RMSNorm); bf16 rounding puts the expected error
at ~1e-1 for hidden 7168.

Launch (one TP size per mpirun; needs the tensorrt_llm venv, see README):

  mpirun -n 8 --allow-run-as-root \
      .venv-trtllm/bin/python all_reduce/bench_allreduce_norm.py
  mpirun -n 4 --allow-run-as-root \
      .venv-trtllm/bin/python all_reduce/bench_allreduce_norm.py
  mpirun -n 2 --allow-run-as-root \
      .venv-trtllm/bin/python all_reduce/bench_allreduce_norm.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.kernel_bench import print_section, print_table, report  # noqa: E402
from bench_allreduce import bench_collective  # noqa: E402

TOKEN_COUNTS = [1, 4, 8, 16, 32]
HIDDEN = 7168  # Kimi K3 attn o_proj
RMS_EPS = 1e-6
FI_MAX_TOKENS = 128  # flashinfer lamport workspace sizing


def build_variants(
    *,
    rank: int,
    world: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    trt_workspace: torch.Tensor,
    tp_group: list[int],
    fi_workspace: torch.Tensor,
    pdl: bool,
):
    """Return {name: (group, fn, outputs)} for one token count.

    fn() runs the variant end to end (capturable in a CUDA graph);
    outputs() reruns it on fresh buffers and returns (norm_out, residual_out)
    -- or (ar_out, None) for the "ar" group -- for the correctness check.
    """
    from flashinfer.comm import AllReduceFusionPattern, trtllm_allreduce_fusion
    from flashinfer.norm import fused_add_rmsnorm
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

    tokens = x.shape[0]

    def trt_ar(strategy, op, res, gam):
        return torch.ops.trtllm.allreduce(
            x, res, gam, None, None, trt_workspace, tp_group,
            int(strategy), int(op), RMS_EPS, True,
        )

    def fi_ar(pattern, use_oneshot, res_in, res_out, norm_out, ar_out, gam):
        trtllm_allreduce_fusion(
            allreduce_in=x,
            world_size=world,
            world_rank=rank,
            token_num=tokens,
            hidden_dim=HIDDEN,
            workspace_ptrs=fi_workspace,
            launch_with_pdl=pdl,
            trigger_completion_at_end=True,
            fp32_acc=False,
            pattern_code=pattern,
            use_oneshot=use_oneshot,
            allreduce_out=ar_out,
            residual_in=res_in,
            residual_out=res_out,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            rms_gamma=gam,
            rms_eps=RMS_EPS,
            scale_factor=None,
            layout_code=None,
        )

    variants: dict[str, tuple[str, callable, callable]] = {}

    # ---- group "ar": allreduce only -------------------------------------
    def add_trt_ar(name, strategy):
        fn = lambda: trt_ar(strategy, AllReduceFusionOp.NONE, None, None)
        variants[name] = ("ar", fn, lambda: (fn()[0], None))

    add_trt_ar("nccl", AllReduceStrategy.NCCL)
    add_trt_ar("trt_oneshot", AllReduceStrategy.ONESHOT)
    if tokens >= world:
        add_trt_ar("trt_twoshot", AllReduceStrategy.TWOSHOT)

    def add_fi_ar(name, use_oneshot):
        ar_out = torch.empty_like(x)
        fn = lambda: fi_ar(
            AllReduceFusionPattern.kAllReduce, use_oneshot,
            None, None, None, ar_out, None,
        )
        def outputs():
            fn()
            return ar_out.clone(), None
        variants[name] = ("ar", fn, outputs)

    add_fi_ar("fi_oneshot", True)
    if tokens > world:
        add_fi_ar("fi_twoshot", False)

    # ---- group "ar_norm": allreduce + residual + RMSNorm ----------------
    def add_unfused(name, strategy):
        res_buf = residual.clone()
        def fn():
            out = trt_ar(strategy, AllReduceFusionOp.NONE, None, None)[0]
            fused_add_rmsnorm(out, res_buf, gamma, RMS_EPS, enable_pdl=pdl)
        def outputs():
            out = trt_ar(strategy, AllReduceFusionOp.NONE, None, None)[0]
            res = residual.clone()
            fused_add_rmsnorm(out, res, gamma, RMS_EPS, enable_pdl=pdl)
            return out, res
        variants[name] = ("ar_norm", fn, outputs)

    add_unfused("nccl+norm", AllReduceStrategy.NCCL)
    add_unfused("trt_oneshot+norm", AllReduceStrategy.ONESHOT)

    def add_trt_fused(name, strategy):
        fn = lambda: trt_ar(strategy, AllReduceFusionOp.RESIDUAL_RMS_NORM,
                            residual, gamma)
        def outputs():
            out = fn()
            return out[0], out[1]
        variants[name] = ("ar_norm", fn, outputs)

    add_trt_fused("trt_oneshot_fused", AllReduceStrategy.ONESHOT)
    add_trt_fused("trt_minlat_fused", AllReduceStrategy.MIN_LATENCY)
    if tokens >= world:
        add_trt_fused("trt_twoshot_fused", AllReduceStrategy.TWOSHOT)

    def add_fi_fused(name, use_oneshot):
        norm_out = torch.empty_like(x)
        res_out = torch.empty_like(x)
        fn = lambda: fi_ar(
            AllReduceFusionPattern.kARResidualRMSNorm, use_oneshot,
            residual, res_out, norm_out, None, gamma,
        )
        def outputs():
            fn()
            return norm_out.clone(), res_out.clone()
        variants[name] = ("ar_norm", fn, outputs)

    add_fi_fused("fi_oneshot_fused", True)
    if tokens > world:
        add_fi_fused("fi_twoshot_fused", False)

    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-pdl", action="store_true",
                        help="disable PDL launches (flashinfer + norm kernel)")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
    torch.cuda.set_device(rank)
    pdl = not args.no_pdl

    csv_path = args.csv or str(
        Path(__file__).parent / "results" / f"bench_allreduce_norm_tp{world}.csv"
    )

    # torch.distributed (gloo) is only used to exchange flashinfer IPC handles.
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    from flashinfer.comm import (
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
    )
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, tp_size=world, rank=rank,
                      gpus_per_node=world)
    tp_group = list(mapping.tp_group)
    trt_workspace = get_allreduce_workspace(mapping)
    fi_ipc, fi_workspace = trtllm_create_ipc_workspace_for_all_reduce_fusion(
        tp_rank=rank, tp_size=world, max_token_num=FI_MAX_TOKENS,
        hidden_dim=HIDDEN, use_fp32_lamport=False,
    )

    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    gamma = torch.rand(HIDDEN, device="cuda", dtype=torch.bfloat16,
                       generator=torch.Generator(device="cuda").manual_seed(7))

    bench_rows, check_rows = [], []
    for tokens in TOKEN_COUNTS:
        x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16,
                        generator=gen)
        residual = torch.randn(tokens, HIDDEN, device="cuda",
                               dtype=torch.bfloat16, generator=gen)

        # Exact fp32 reference: gloo CPU sum, then fp32 residual + RMSNorm.
        ref_sum = x.float().cpu()
        dist.all_reduce(ref_sum)
        ref_sum = ref_sum.cuda()
        ref_res = ref_sum + residual.float()
        ref_norm = (
            ref_res
            * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + RMS_EPS)
            * gamma.float()
        )

        variants = build_variants(
            rank=rank, world=world, x=x, residual=residual, gamma=gamma,
            trt_workspace=trt_workspace, tp_group=tp_group,
            fi_workspace=fi_workspace, pdl=pdl,
        )

        msg_kb = tokens * HIDDEN * 2 / 1024
        for name, (group, fn, outputs) in variants.items():
            row = {"impl": name, "group": group, "tokens": tokens,
                   "msg_kb": round(msg_kb, 1)}
            check = {"impl": name, "group": group, "tokens": tokens}
            try:
                out, res_out = outputs()
                ref = ref_norm if group == "ar_norm" else ref_sum
                check["max_abs_vs_fp32"] = (out.float() - ref).abs().max().item()
                if res_out is not None:
                    check["res_max_abs"] = (
                        (res_out.float() - ref_res).abs().max().item()
                    )
                torch.cuda.synchronize()
                comm.Barrier()
                row["latency_us"] = bench_collective(
                    comm, fn, args.warmup, args.iters, args.repeats
                )
            except Exception as exc:  # noqa: BLE001
                torch.cuda.synchronize()
                row["error"] = f"{type(exc).__name__}: {exc}"
                check["error"] = row["error"]
            bench_rows.append(row)
            check_rows.append(check)

    # Max latency over ranks (collectives finish together, but launch skew
    # and clock skew make per-rank medians differ).
    all_rows = comm.gather(bench_rows, root=0)
    if rank == 0:
        merged = []
        for i, row in enumerate(bench_rows):
            row = dict(row)
            lat = [r[i]["latency_us"] for r in all_rows if "latency_us" in r[i]]
            if lat:
                row["latency_us"] = max(lat)
            merged.append(row)
        for row in merged:
            base_impl = "nccl+norm" if row["group"] == "ar_norm" else "nccl"
            base = next(
                (r["latency_us"] for r in merged
                 if r["impl"] == base_impl and r["tokens"] == row["tokens"]
                 and "latency_us" in r),
                None,
            )
            if base and "latency_us" in row:
                row["speedup_vs_nccl"] = base / row["latency_us"]

        print_section(
            f"TP={world} allreduce(+RMSNorm) TRT vs FlashInfer, "
            f"hidden {HIDDEN} bf16, PDL={'on' if pdl else 'off'}"
        )
        print_table(
            check_rows,
            title="correctness vs fp32 reference (bf16 rounding ~1e-1 expected)",
        )
        report(
            merged,
            title="latency (max over ranks, median of repeats)",
            csv_path=csv_path,
            plot=dict(
                x="tokens",
                y="latency_us",
                panel="group",
                suptitle=f"TP={world} allreduce + RMSNorm fusion, "
                f"Kimi K3 h{HIDDEN} (B200)",
            ),
        )

    comm.Barrier()
    try:
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion(fi_ipc)
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
