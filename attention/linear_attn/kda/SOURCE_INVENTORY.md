# KDA source inventory

Static AST inventory of every Python module in this directory, including wrappers,
implementation files, registries, and legacy layer code. No modules were imported
or kernels executed. Use [KERNELS.md](KERNELS.md) for the mathematical reading order.
Source line numbers are navigation hints for this snapshot, not permanent IDs.
Functions nested inside launch factories are not individually enumerated here;
read their containing factory when following a kernel. A registry entry is not
proof that its optional dependency is installed or that its historical speed
claim remains true.

## [__init__.py](__init__.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `kda.kda_attention`.

## [attention_modules.py](attention_modules.py)

`AttentionShape` (L15), `SGLangStyleKDAAttention` (L26), `get_attention_runners` (L158)

Relevant import targets: `fla.layers.kda`, `fla.models.utils`, `sglang.srt.layers.attention.fla.fused_norm_gate`, `sglang.srt.layers.attention.linear.kernels.kda_triton`, `sglang.srt.layers.attention.mamba.causal_conv1d_triton`.

## [b10/__init__.py](b10/__init__.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

## [b10/_b10_kda_decode_impl_cutedsl.py](b10/_b10_kda_decode_impl_cutedsl.py)

`kda_decode_kernel` (L170), `kda_decode_kernel_vec` (L278), `make_vsplit_launcher` (L396), `make_ptr_launcher` (L783), `kda_decode_launch` (L816), `_pick_cs` (L840), `_ep` (L1027), `_num_sms` (L1038), `make_launcher` (L1051), `spill_bytes` (L1817), `chosen_config` (L1867), `_Launch` (L1878), `KdaDecode` (L1929), `KdaDecodeGatedBody` (L2011), `kda_decode_step` (L2226), `kda_decode_gated` (L2235), `kda_decode_conv_step` (L2249), `kda_decode_conv_gated` (L2266), `_ref_decode` (L2292), `_ref_gated_norm` (L2307), `_cos` (L2314), `_self_test` (L2319)

Relevant import targets: `sglang.srt.layers.attention.mamba.causal_conv1d_triton`.

## [b10/_b10_kda_replay_ssm_conv_gated_impl_cutedsl.py](b10/_b10_kda_replay_ssm_conv_gated_impl_cutedsl.py)

`_sigmoid` (L158), `_conv_gather` (L164), `_conv_acc4` (L189), `make_launcher` (L248), `_Engine` (L1062), `_run` (L1103), `kda_replay_ssm_conv_gated` (L1126), `kda_replay_ssm_gated_noconv` (L1156)

## [b10/_b10_kda_replay_ssm_impl_cutedsl.py](b10/_b10_kda_replay_ssm_impl_cutedsl.py)

`_sigmoid` (L97), `_conv_gather` (L103), `_conv_fold` (L120), `make_launcher` (L190), `_make_bare_launcher` (L217), `_make_onecta_launcher` (L787), `_Engine` (L1704), `kda_replay_ssm_fused` (L1741), `_run_onecta` (L1778), `kda_replay_ssm_fused_gated` (L1797), `kda_replay_ssm_fused_prenorm` (L1805), `_check_conv_args` (L1815), `kda_replay_ssm_fused_conv` (L1832), `kda_replay_ssm_fused_conv_gated` (L1863), `fused_recurrent_gated_delta_rule_cached_replay_update` (L1884)

## [b10/_b10_kda_save_ssm_conv_gated_impl_cutedsl.py](b10/_b10_kda_save_ssm_conv_gated_impl_cutedsl.py)

`make_launcher` (L199), `_dummy_ptr` (L1052), `_Engine` (L1064), `_launch` (L1111), `kda_save_ssm_conv_gated` (L1129), `kda_save_ssm_gated_noconv` (L1152)

## [b10/_b10_kda_save_ssm_impl_cutedsl.py](b10/_b10_kda_save_ssm_impl_cutedsl.py)

`make_launcher` (L227), `_ptr_spec` (L1124), `_Engine` (L1130), `_bind_and_launch` (L1181), `kda_save_ssm` (L1187), `_launch_gated` (L1223), `kda_save_ssm_gated` (L1258), `kda_save_ssm_prenorm` (L1264), `_check_conv_args` (L1270), `kda_save_ssm_conv` (L1291), `kda_save_ssm_conv_gated` (L1321), `_ref_conv_silu` (L1349), `_sglang_conv_closure` (L1375), `_time_fn` (L1406), `_selftest` (L1420)

Relevant import targets: `sglang.srt.layers.attention.mamba.causal_conv1d_triton`.

## [b10/b10_kda_chunk_prefill_cutedsl.py](b10/b10_kda_chunk_prefill_cutedsl.py)

`_set_nwc` (L127), `_swz_k` (L168), `_wv_k` (L177), `_flat8` (L184), `frag_B` (L189), `acc_to_a` (L205), `sub_to_a` (L217), `sub_frag_to_a` (L230), `wgemm` (L247), `acc_kt_to_a` (L258), `wgemm_2a` (L271), `wgemm_s` (L301), `_decay` (L328), `wgemm_t` (L340), `wgemm_kt` (L353), `SmemP` (L367), `_kern_prep16` (L378), `_kern_carry16` (L736), `_cw_issue` (L886), `_cw_step` (L938), `_pick_rpw` (L1050), `_pick_nwc` (L1089), `_pick_mbpc` (L1131), `_compile` (L1154), `_Lazy` (L1208), `_workspace` (L1228), `_pick_ncb` (L1241), `kda_chunk_prefill` (L1250)

## [b10/b10_kda_decode_conv_cutedsl.py](b10/b10_kda_decode_conv_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_decode_impl_cutedsl`.

## [b10/b10_kda_decode_conv_gated_cutedsl.py](b10/b10_kda_decode_conv_gated_cutedsl.py)

`candidate_ladder` (L303), `_ep` (L380), `_num_sms` (L391), `make_launcher` (L404), `_ptr_dtypes` (L1097), `func_attrs` (L1116), `ctas_per_sm` (L1160), `spill_bytes` (L1169), `chosen_config` (L1219), `_Launch` (L1230), `KdaDecodeConvGated` (L1295), `kda_decode_conv_gated_packed` (L1504), `kda_decode_conv_gated` (L1525), `kda_decode_conv_gated_raw` (L1536), `kda_decode_gated_raw_strided` (L1571), `ref_conv` (L1603), `ref_conv_triton_emul` (L1625), `ref_pre_norm` (L1648), `ref_gate` (L1660), `ref_step` (L1668), `_venv_src_candidates` (L1689), `_load_real_sglang_leaf` (L1705), `sglang_conv_update` (L1748), `_triton_norm_gate_fn` (L1774), `epilogue_oracle` (L1814), `_make_inputs` (L1871), `_randn` (L1887), `_run_both` (L1891), `_check_one` (L1909), `run_check` (L1914), `_time_block` (L2132), `_bench` (L2144), `_bench_many` (L2150), `_graph_of` (L2181), `_time_graph` (L2208), `_graph_reps` (L2226), `bench_shape` (L2233), `run_suite` (L2283), `run_bench` (L2320), `main` (L2416)

Relevant import targets: `fla.modules.fused_norm_gate`, `sglang.srt.layers.attention.fla.fused_norm_gate`, `sglang.srt.layers.attention.mamba.causal_conv1d_triton`.

## [b10/b10_kda_decode_conv_gated_fusefb_cutedsl.py](b10/b10_kda_decode_conv_gated_fusefb_cutedsl.py)

`candidate_ladder` (L310), `_ep` (L387), `_num_sms` (L398), `make_launcher` (L411), `_ptr_dtypes` (L1132), `func_attrs` (L1153), `ctas_per_sm` (L1197), `spill_bytes` (L1206), `chosen_config` (L1256), `_Launch` (L1267), `KdaDecodeConvGated` (L1332), `kda_decode_conv_gated_raw_fusefb` (L1541)

## [b10/b10_kda_decode_cutedsl.py](b10/b10_kda_decode_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_decode_impl_cutedsl`.

## [b10/b10_kda_decode_gated_cutedsl.py](b10/b10_kda_decode_gated_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_decode_impl_cutedsl`.

## [b10/b10_kda_gated_rmsnorm_cutedsl.py](b10/b10_kda_gated_rmsnorm_cutedsl.py)

`pick_rpt` (L111), `_selp_lt0` (L123), `_tree_sum` (L138), `_sigmoid_group` (L154), `make_launcher` (L231), `_empty_kernel` (L375), `_empty_launch` (L380), `empty_node_ms` (L387), `_Launch` (L414), `GatedRMSNorm` (L460), `gated_rmsnorm` (L546)

## [b10/b10_kda_prefill_cutedsl.py](b10/b10_kda_prefill_cutedsl.py)

`make_launcher` (L87), `_Engine` (L402), `kda_spec_decode` (L449)

## [b10/b10_kda_prefill_triton.py](b10/b10_kda_prefill_triton.py)

`_prep_kernel` (L65), `_carry_kernel` (L179), `kda_chunk_prefill` (L241)

## [b10/b10_kda_replay_ssm_conv_cutedsl.py](b10/b10_kda_replay_ssm_conv_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_replay_ssm_impl_cutedsl`.

## [b10/b10_kda_replay_ssm_conv_gated_cutedsl.py](b10/b10_kda_replay_ssm_conv_gated_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_replay_ssm_conv_gated_impl_cutedsl`.

## [b10/b10_kda_replay_ssm_conv_gated_wychunk_cutedsl.py](b10/b10_kda_replay_ssm_conv_gated_wychunk_cutedsl.py)

`_kda_replay_ssm_conv_gated_wychunk_kernel` (L59), `_launch` (L800), `_t` (L843), `kda_replay_ssm_conv_gated_wychunk` (L847), `bytes_moved` (L925)

## [b10/b10_kda_replay_ssm_cutedsl.py](b10/b10_kda_replay_ssm_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_replay_ssm_impl_cutedsl`.

## [b10/b10_kda_replay_ssm_gated_cutedsl.py](b10/b10_kda_replay_ssm_gated_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_replay_ssm_conv_gated_impl_cutedsl`.

## [b10/b10_kda_save_ssm_conv_cutedsl.py](b10/b10_kda_save_ssm_conv_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_save_ssm_impl_cutedsl`.

## [b10/b10_kda_save_ssm_conv_gated_cutedsl.py](b10/b10_kda_save_ssm_conv_gated_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_save_ssm_conv_gated_impl_cutedsl`.

## [b10/b10_kda_save_ssm_cutedsl.py](b10/b10_kda_save_ssm_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_save_ssm_impl_cutedsl`.

## [b10/b10_kda_save_ssm_gated_cutedsl.py](b10/b10_kda_save_ssm_gated_cutedsl.py)

Package marker or re-export-only wrapper; no top-level function/class definitions.

Relevant import targets: `_b10_kda_save_ssm_conv_gated_impl_cutedsl`.

## [b10/bench_b10_kda_fusefb.py](b10/bench_b10_kda_fusefb.py)

`graph_time_us` (L29), `main` (L47)

Relevant import targets: `kda.b10.b10_kda_decode_conv_gated_cutedsl`, `kda.b10.b10_kda_decode_conv_gated_fusefb_cutedsl`.

## [inputs.py](inputs.py)

`DecodeInputs` (L26), `ConvNormInputs` (L38), `make_conv_norm_inputs` (L52), `PrefillInputs` (L73), `make_decode_inputs` (L86), `make_prefill_inputs` (L132), `split_qkv` (L180), `beta_logit_of` (L199)

## [kda_attention.py](kda_attention.py)

`activate_kda_gate` (L33), `conv4_silu_reference` (L60), `gated_rmsnorm_reference` (L83), `kda_recurrent_reference` (L100)

## [kda_chunk_verify_triton.py](kda_chunk_verify_triton.py)

`_chunk_prep_kernel` (L54), `_chunk_state_kernel` (L164), `_chunk_fold_kernel` (L220), `_scratch` (L263), `kda_chunk_verify` (L274)

## [kda_decode_register.py](kda_decode_register.py)

`_torch_kda_reference` (L48), `_b10_kda_decode` (L85), `_b10_kda_decode_gated` (L114), `_b10_kda_decode_conv_gated` (L145), `_fla_kda_recurrent` (L196), `_sglang_kda_packed` (L231), `_sglang_kda_split` (L259), `_sglang_kda_fused_decode` (L290), `_sglang_kda_cutedsl` (L362), `_sglang_kda_flashinfer` (L397), `_vllm_kda_recurrent` (L426), `_trtllm_kda_split` (L467), `_trtllm_kda_packed` (L503), `_trtllm_kda_fused_decode` (L532), `get_kda_decode_backends` (L591)

Relevant import targets: `fla.ops.kda.fused_recurrent`, `kda.b10.b10_kda_decode_conv_gated_cutedsl`, `kda.b10.b10_kda_decode_cutedsl`, `kda.b10.b10_kda_decode_gated_cutedsl`, `kda.inputs`, `kda.kda_attention`, `sglang.srt.layers.attention.linear.kernels.kda_cutedsl`, `sglang.srt.layers.attention.linear.kernels.kda_flashinfer`, `sglang.srt.layers.attention.linear.kernels.kda_triton`.

Active registrations:

- `KDA_DECODE:torch_kda_reference` → `_torch_kda_reference`
- `KDA_DECODE:b10_kda_decode` → `_b10_kda_decode`
- `KDA_DECODE:b10_kda_decode_gated` → `_b10_kda_decode_gated`
- `KDA_DECODE:b10_kda_decode_conv_gated` → `_b10_kda_decode_conv_gated`
- `KDA_DECODE:fla_kda_recurrent` → `_fla_kda_recurrent`
- `KDA_DECODE:sglang_kda_packed` → `_sglang_kda_packed`
- `KDA_DECODE:sglang_kda_split` → `_sglang_kda_split`
- `KDA_DECODE:sglang_kda_fused_decode` → `_sglang_kda_fused_decode`
- `KDA_DECODE:sglang_kda_flashinfer` → `_sglang_kda_flashinfer`
- `KDA_DECODE:vllm_kda_recurrent` → `_vllm_kda_recurrent`
- `KDA_DECODE:trtllm_kda_split` → `_trtllm_kda_split`
- `KDA_DECODE:trtllm_kda_packed` → `_trtllm_kda_packed`
- `KDA_DECODE:trtllm_kda_fused_decode` → `_trtllm_kda_fused_decode`

## [kda_prefill_register.py](kda_prefill_register.py)

`_fla_kda_chunk` (L44), `_fla_kda_safe_triton` (L72), `_flashkda_fwd_builder` (L113), `_flash_kda` (L148), `_import_flash_kda_ptx` (L171), `_flashkda_ptx_int21` (L198), `_fi_recurrent_kda` (L214), `_sglang_kda_chunk` (L241), `_vllm_kda_chunk` (L267), `_trtllm_kda_chunk` (L293), `_trtllm_kda_cute` (L323), `_b10_flashkda_triton_builder` (L364), `_b10_flashkda_triton_c16` (L397), `_b10_flashkda_triton_c64` (L405), `_b10_kda_chunk_prefill` (L415), `_b10_kda_recurrent` (L446), `_flashinfer_cake_kda` (L500), `get_kda_prefill_backends` (L604), `_sglang_extend_builder` (L620), `_sglang_extend_triton` (L653), `_sglang_extend_cutedsl` (L666), `_sglang_extend_flashkda` (L679), `get_kda_sglang_extend_backends` (L694)

Relevant import targets: `fla.ops.kda`, `flashinfer.kda_decode`, `flashinfer.kda_prefill`, `kda.b10.b10_kda_chunk_prefill_cutedsl`, `kda.b10.b10_kda_prefill_cutedsl`, `kda.b10.b10_kda_prefill_triton`, `kda.inputs`, `kda.kda_attention`, `sglang.srt.layers.attention.fla.kda`.

Active registrations:

- `KDA_PREFILL:fla_kda_chunk` → `_fla_kda_chunk`
- `KDA_PREFILL:fla_kda_safe_triton` → `_fla_kda_safe_triton`
- `KDA_PREFILL:flash_kda` → `_flash_kda`
- `KDA_PREFILL:flashkda_ptx_int21` → `_flashkda_ptx_int21`
- `KDA_PREFILL:fi_recurrent_kda` → `_fi_recurrent_kda`
- `KDA_PREFILL:sglang_kda_chunk` → `_sglang_kda_chunk`
- `KDA_PREFILL:vllm_kda_chunk` → `_vllm_kda_chunk`
- `KDA_PREFILL:trtllm_kda_chunk` → `_trtllm_kda_chunk`
- `KDA_PREFILL:trtllm_kda_cute` → `_trtllm_kda_cute`
- `KDA_PREFILL:b10_flashkda_triton_c16` → `_b10_flashkda_triton_c16`
- `KDA_PREFILL:b10_flashkda_triton_c64` → `_b10_flashkda_triton_c64`
- `KDA_PREFILL:b10_kda_chunk_prefill` → `_b10_kda_chunk_prefill`
- `KDA_PREFILL:b10_kda_recurrent` → `_b10_kda_recurrent`
- `KDA_PREFILL:flashinfer_cake_kda` → `_flashinfer_cake_kda`
- `KDA_SGLANG_EXTEND:sglang_triton` → `_sglang_extend_triton`
- `KDA_SGLANG_EXTEND:sglang_cutedsl` → `_sglang_extend_cutedsl`
- `KDA_SGLANG_EXTEND:sglang_flashkda` → `_sglang_extend_flashkda`

## [kda_replayssm_fold.py](kda_replayssm_fold.py)

`kda_replayssm_exact_fold_kernel` (L46), `commit_kda_replayssm_spec` (L185), `commit_kda_replayssm_spec_all_layers` (L263)

## [kda_verify_register.py](kda_verify_register.py)

`set_heads` (L71), `_safe_gate` (L77), `verify_tensors` (L85), `verify_reference` (L146), `snapshot_oracle` (L155), `_load_trt_cached_replay` (L175), `sglang_verify_closure` (L189), `commit_gather_closure` (L230), `ring_buffers` (L249), `ring_store_closure` (L258), `fold_replay_closure` (L275), `trt_closure` (L303), `replay_ssm_closure` (L331), `save_ssm_closure` (L345), `_conv_tensors` (L369), `save_ssm_conv_closure` (L392), `replay_ssm_conv_closure` (L416), `conv_silu_reference` (L447), `gated_norm_reference` (L463), `_canonical_gate` (L471), `activate_gate` (L478), `verify_conv_tensors` (L485), `e2e_oracle` (L520), `logical_s_from_ring` (L556), `b10_save_ssm_conv_gated_closure` (L575), `b10_save_ssm_gated_closure` (L603), `b10_replay_ssm_conv_gated_closure` (L629), `b10_replay_ssm_conv_gated_wychunk_closure` (L659), `sglang_conv_closure` (L710), `sglang_norm_closure` (L732), `sglang_save_ssm_chain_closure` (L750), `sglang_save_ssm_stitched` (L774), `trt_replay_chain_closure` (L801), `trt_replay_stitched` (L823), `triton_chunk_closure` (L851), `triton_chunk_chain_closure` (L873), `triton_chunk_stitched` (L893)

Relevant import targets: `kda.b10.b10_kda_replay_ssm_conv_gated_cutedsl`, `kda.b10.b10_kda_replay_ssm_conv_gated_wychunk_cutedsl`, `kda.b10.b10_kda_replay_ssm_gated_cutedsl`, `kda.b10.b10_kda_save_ssm_conv_gated_cutedsl`, `kda.b10.b10_kda_save_ssm_gated_cutedsl`, `kda.kda_attention`, `kda.kda_chunk_verify_triton`, `kda.kda_replayssm_fold`, `sglang.srt.layers.attention.fla.fused_norm_gate`, `sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent`, `sglang.srt.layers.attention.mamba.causal_conv1d_triton`.


