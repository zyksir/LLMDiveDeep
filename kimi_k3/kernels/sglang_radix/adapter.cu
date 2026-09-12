// SPDX-License-Identifier: Apache-2.0
// Local standalone export adapter. The vendored SGLang kernel and headers are
// separate, byte-for-byte unchanged files under csrc/ and include/.

#include "moe/route_radix.cuh"

#include <tvm/ffi/function.h>

static void run(
    const tvm::ffi::TensorView scores,
    const tvm::ffi::TensorView bias,
    const tvm::ffi::TensorView out_w,
    const tvm::ffi::TensorView out_i,
    int64_t topk,
    double routed_scaling_factor,
    bool renormalize,
    bool apply_scale,
    bool sorted) {
  sglang::RouteRadixKernel<true>::run(
      scores,
      bias,
      out_w,
      out_i,
      topk,
      routed_scaling_factor,
      renormalize,
      apply_scale,
      sorted);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
