"""Cache manager + attention metadata + input builders for the 4-block demo.

Mirrors the unchanged TRT-LLM unittest driving pattern
(``tests/unittest/_torch/models/test_kimi_linear.py``,
``_run_kimi_linear_mixed_prefill_and_decode``): a
``MixedMambaHybridCacheManager`` covers the 3 KDA blocks (conv+SSM state
pools) and the 1 MLA block (paged latent KV, ``num_kv_heads=1``,
``head_dim=kv_lora_rank+qk_rope_head_dim=576``); the TRTLLM attention backend
metadata is built directly and ``prepare()``-d, and ``Mamba2Metadata`` is
derived from it exactly as ``KimiLinearModel.forward`` does.

Decode note: decode-phase runs start from ZERO-filled caches with
``num_cached_tokens_per_seq=[isl]`` rather than executing a real prefill
first. Kernel shapes, cache layouts, and memory traffic are identical to a
real ISL-token history; only the cache VALUES are synthetic (this is a
perf/comm demo with random attention weights anyway). This keeps the routed
MoE workspace sized for the benchmark tokens instead of a one-off giant
prefill.
"""

from __future__ import annotations

from typing import Any

import torch

from stack import NUM_BLOCKS

TOKENS_PER_BLOCK = 64
MLA_LAYER_IDX = 3  # 0-based; blocks 0..2 are KDA
_REQUEST_ID_BASE = 1


class DemoRuntime:
    """Per-(mode, phase, batch) cache manager + metadata bundle."""

    def __init__(
        self,
        *,
        attn_model_config: Any,
        batch_size: int,
        isl: int,
        phase: str,
        device: torch.device,
    ) -> None:
        from tensorrt_llm._torch import metadata as metadata_lib
        from tensorrt_llm._torch.attention_backend import utils as attention_utils
        from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
        from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
            MixedMambaHybridCacheManager,
        )
        from tensorrt_llm._torch.pyexecutor.resource_manager import (
            CacheTypeCpp,
            DataType,
        )
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        if phase not in ("prefill", "decode"):
            raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}")
        config = attn_model_config.pretrained_config
        linear_cfg = config.linear_attn_config
        self.phase = phase
        self.batch_size = batch_size
        self.isl = isl
        self.device = device

        max_seq_len = isl + TOKENS_PER_BLOCK
        num_kda = len(linear_cfg["kda_layers"])
        self.cache_manager = MixedMambaHybridCacheManager(
            mamba_d_state=linear_cfg["head_dim"],
            mamba_d_conv=linear_cfg["short_conv_kernel_size"],
            mamba_num_heads=linear_cfg["num_heads"],
            mamba_n_groups=linear_cfg["num_heads"],
            mamba_head_dim=linear_cfg["head_dim"],
            mamba_num_layers=num_kda,
            mamba_layer_mask=[True] * num_kda + [False],
            mamba_cache_dtype=torch.bfloat16,
            mamba_ssm_cache_dtype=torch.float32,
            kv_cache_config=KvCacheConfig(
                max_tokens=batch_size * max_seq_len,
                enable_block_reuse=False,
            ),
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=NUM_BLOCKS - num_kda,
            layer_mask=[False] * num_kda + [True],
            num_kv_heads=1,
            head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=attn_model_config.mapping,
            dtype=DataType.BF16,
            model_type="qwen3_next",
        )
        self.request_ids = list(range(_REQUEST_ID_BASE, _REQUEST_ID_BASE + batch_size))
        self.cache_manager.add_dummy_requests(
            self.request_ids, token_nums=[isl] * batch_size
        )
        # This harness bypasses the scheduler's cache lifecycle; zero the pools
        # so repeated runs never inherit allocator garbage (same as the test).
        self.cache_manager.get_buffers(MLA_LAYER_IDX).zero_()
        for kda_layer in range(num_kda):
            self.cache_manager.get_conv_states(kda_layer).zero_()
            self.cache_manager.get_ssm_states(kda_layer).zero_()

        if phase == "prefill":
            seq_lens = [isl] * batch_size
            num_contexts = batch_size
            cached = [0] * batch_size
            self.num_tokens = batch_size * isl
            position_ids = torch.cat(
                [torch.arange(isl, dtype=torch.long, device=device)] * batch_size
            ).unsqueeze(0)
        else:
            seq_lens = [1] * batch_size
            num_contexts = 0
            cached = [isl] * batch_size
            self.num_tokens = batch_size
            position_ids = torch.full(
                (1, batch_size), isl, dtype=torch.long, device=device
            )
        self.position_ids = position_ids

        metadata_cls = attention_utils.get_attention_backend("TRTLLM").Metadata
        self.attn_metadata = metadata_cls(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            num_contexts=num_contexts,
            kv_cache_params=metadata_lib.KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=cached
            ),
            kv_cache_manager=self.cache_manager,
            request_ids=self.request_ids,
            prompt_lens=[isl] * batch_size,
            max_num_requests=batch_size,
            max_num_tokens=max(32, self.num_tokens),
        )
        self.attn_metadata.prepare()
        self.mamba_metadata = Mamba2Metadata(
            self.attn_metadata.max_num_requests, chunk_size=128
        )
        self.mamba_metadata.prepare(self.attn_metadata)

    def make_hidden_states(self, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return (
            torch.randn(
                (self.num_tokens, 7168),
                dtype=torch.bfloat16,
                device=self.device,
                generator=generator,
            )
            * 0.5
        )

    def shutdown(self) -> None:
        self.cache_manager.shutdown()
