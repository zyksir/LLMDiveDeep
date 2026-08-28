# Provenance: import-vs-copy inventory

Generated: 2026-08-28

All TRT-LLM kernels and Python modules are used UNCHANGED. Nothing is
copied-and-edited: every piece is either imported directly (IMPORTED)
or is a small, documented in-package adaptation (ADAPTED) where direct
import is impractical. There are no COPIED files.
`harness.verify_provenance()` re-hashes the imported harness files at
every benchmark run and records the result in each JSON receipt.

- Reference checkout: `/node-storage/trt-llm` branch
  `yikai/optimized-k3` @ `56c0dc32f5641ad48b74a6c2536ffb293589ca98`
- Runtime container: trt-k3-prod image (baseten/dynamo-cache-aware-routing:trtllm-d9fa74fd94-d9d1a655f-c08257485a, installed tensorrt_llm 1.3.0rc19 fork with NVLinkOneSided/MegaMoE)

## Inventory

| kind | what | source / reason |
|---|---|---|
| IMPORTED (installed package) | tensorrt_llm runtime: torch.ops.trtllm compiled ops, ConfigurableMoE/TRTLLMGenFusedMoE, NVLinkOneSided, GatedMLP, AllReduce, Mapping, ModelConfig, AutoTuner | container-installed tensorrt_llm (build of fork commit d9fa74fd94); the reference checkout's tensorrt_llm/ python diverges from the installed compiled bindings by ~129 files, so overlaying it via PYTHONPATH is not binary-safe |
| IMPORTED (reference checkout) | bench_moe microbenchmark harness (tests/microbenchmarks/bench_moe/**): module build, mapping, quant params, routing plan + logits projection, specs, device utils | PYTHONPATH import from the checkout below; byte-identical to the container-build checkout (hashes pinned in this manifest) |
| IMPORTED (reference checkout) | unittest MoE fixtures (tests/unittest/_torch/modules/moe/quantize_utils.py, moe_test_utils.py): MXFP4MXFP8QuantizeUtil weight generation (routed + shared experts), prepare_weights_from_backend, MXFP4MXFP8RefGatedMLPFusedMoE sequential reference | PYTHONPATH import from the checkout below; byte-identical to the container-build checkout |
| IMPORTED (LLMDiveDeep) | bench_a2a_megamoe_pipeline._build_module/_correctness, bench_a2a_megamoe_sweep._per_rank_tokens/_prepare_local_backend_weights (cached local weights) | kimi_k3_layer/ in this repository (the proven MegaMoE receipt harness) |
| ADAPTED (in-package) | bench_layer._build_routed_with_reference | bench_moe.build._build_moe_module — same calls, same order; direct import impractical because the stock builder discards the reference weights/module needed for the correctness gate |
| ADAPTED (in-package) | bench_layer._run_layer_autotune | bench_moe.timing.autotune._run_autotune — same tuner settings; direct import impractical because the stock helper's signature is bound to a bare MoE forward, not a full-layer callable |
| ADAPTED (in-package) | layer.KimiK3MoELayerBaseline / KimiK3SharedExperts wiring | tensorrt_llm/_torch/models/modeling_deepseekv3.py Deepseekv3MoE — shared GatedMLP(reduce_output=False) + AllReduce + routed MoE composition; direct import impractical because the model class pulls attention/pipeline dependencies irrelevant to a single-layer benchmark |

## Pinned harness file hashes

| file (relative to checkout) | sha256 |
|---|---|
| `tests/microbenchmarks/bench_moe/__init__.py` | `f57008199ffc20c6d9a42175491618a4630e1f9b5761faadaa697ee6d06b85f5` |
| `tests/microbenchmarks/bench_moe/__main__.py` | `b8f5c8c64ce0015dcd8694317ac6075a4bc80fcad6f8c563b57b4739258d5bc9` |
| `tests/microbenchmarks/bench_moe/backend.py` | `e6c6adc4fa30efdea374d5b8099e9ea6e4b74c12250aee9e508a4a18ff7c34f5` |
| `tests/microbenchmarks/bench_moe/build.py` | `132ed6872864b8c1427f7e829458869f28c4bb0f44ce075281bb2a444e52c8c2` |
| `tests/microbenchmarks/bench_moe/case_runner.py` | `a8e419c73bc9df2723d21550717621afae18f90fee1a4eef65a365a544f461cd` |
| `tests/microbenchmarks/bench_moe/cli.py` | `0932755199c5fa293be7e524cc861c94f2be3eb2528b97eeadff642823a6a583` |
| `tests/microbenchmarks/bench_moe/mapping.py` | `e62b48e2089821b10166e65c53ad4378ac34d5146ae10cb6a367713ba62205d1` |
| `tests/microbenchmarks/bench_moe/quantize.py` | `abec94c8f2d712aa4d0a5632d2eb97f399bb51277d857668b5cbfa0169e491b8` |
| `tests/microbenchmarks/bench_moe/reporting.py` | `a4de1c007d57bf1222d5fdf3e7158913ef1c20425319bf6a9cc7cca64ba138c9` |
| `tests/microbenchmarks/bench_moe/results.py` | `bea5bc955e43c60a5e62e90430a661b53e06aaada2d233f7739cab344012910e` |
| `tests/microbenchmarks/bench_moe/routing/__init__.py` | `0ad3e43c7987c50540ca38155e747540008665b08d480a52a47d8675fe8060db` |
| `tests/microbenchmarks/bench_moe/routing/builders.py` | `54a678af4e1e971686cca5f6a3cb44d1a56fa362df9dc3bb24401dc338571250` |
| `tests/microbenchmarks/bench_moe/routing/materialize.py` | `ba8075108854a90f102b5bd2a83d30be00da45f9a548aa2ab10b684713a3dfd7` |
| `tests/microbenchmarks/bench_moe/routing/native_logits.py` | `c11c8d5deba36bb56316101ce26cb55a2568bfd9ddde42711b161d7f5bc14627` |
| `tests/microbenchmarks/bench_moe/routing/parsing.py` | `8f99c1210ab4ca0d26e6cc7d3ecf01941c40d6d05390c49380c89318328f0610` |
| `tests/microbenchmarks/bench_moe/search.py` | `0043274af92753af070cf6fc53eb99f5d7e4f2f7e1764582039ecb38d646be41` |
| `tests/microbenchmarks/bench_moe/specs.py` | `8514f81f1c2e8a362c04d8e7f3f9a61c67546809442748693c8cb6c81203e353` |
| `tests/microbenchmarks/bench_moe/timing/__init__.py` | `2953112c221a9b0610afe8e86f9e27aacaaab830f9244b86912a2484bd012b3c` |
| `tests/microbenchmarks/bench_moe/timing/autotune.py` | `b5fd64812bdfc008e46ad27691b23f14eab637c04be00e3d607e8f6364b9aa31` |
| `tests/microbenchmarks/bench_moe/timing/cuda_graph.py` | `86ddf9670a0708514c3b1ec93d62e3f478ef0b0ffad41bb890eb352e8b42dd5d` |
| `tests/microbenchmarks/bench_moe/timing/cupti.py` | `fce4db253f6b611cc33ea42d9277d2d0c884e60775d353cb4a4844843345a06a` |
| `tests/microbenchmarks/bench_moe/timing/eager.py` | `8239d3b8d7e950cf539ca3f0327ea53d1e3227cc49673d638f36d8983bbb5dd8` |
| `tests/microbenchmarks/bench_moe/utils.py` | `2b09120c0aa425465c3873d543609ac1996f10c69afd70f2f18a1f2e6d40ad09` |
| `tests/microbenchmarks/bench_moe/worker.py` | `883893ec662f7ce408c77d2166eb48863bfff9c9c59f5b0649fbb2c2f82e9433` |
| `tests/unittest/_torch/helpers.py` | `24c42a7e5ac395fb5e4840b3a652092ae11646f2dc7bd941aa478595919d6003` |
| `tests/unittest/_torch/modules/moe/moe_test_utils.py` | `3e039738f09f2768fc6fd8f5c902693d5772e52537ceb2965a51713f359e5014` |
| `tests/unittest/_torch/modules/moe/quantize_utils.py` | `0d4e2cc7df00afe8a52084b774166929d87c8aca51e6bb0a3420e060f18ad30f` |
| `tests/unittest/utils/util.py` | `3826a2f73faa92bf183166e0b3b5f06cbf5b7140b31a6fe363cc5a40f6dd35c5` |
