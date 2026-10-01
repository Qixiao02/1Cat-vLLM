// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Checkpoint-FP16 Qwen3.8 TP4 batch projections. No quantization or K-tree
// change.
//
// SX port of upstream 1Cat main@d30469863's packed m8n8k4 batch kernels for
// the native-MTP M5/M10 verify (sm70_fp16_gemv.py, "SX MTP batch routes"):
// the E512 router projection (3b7365925) and the shared-expert gate/up
// projection with its SiLU-and-multiply (69eac6d8e). Upstream's no-MTP dense
// batch kernel (BATCH_FASTPATH) is not part of this build, nor is its
// shared-gate sigmoid/multiply epilogue: the fork's exact multi-row shared
// gate (SX_OPT_SHARED_GATE_ROWS) already serves those rows.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <torch/types.h>

namespace {
#define GDN_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// The high-precision small-batch cuBLAS router projection uses four
// contiguous FP32 K640 partitions. Each independent m8n8k4 quad pair owns one
// partition, with an ordered warp-shuffle reduction; N8 tiles distribute work
// over more SMs without changing the dot-product tree.
template <int Unroll>
__global__ __launch_bounds__(32, 8) void router_split_quad_kernel(const half* x,
                                                                  const half* w,
                                                                  half* out,
                                                                  int m) {
  const int lane = threadIdx.x, split = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0), row = blockIdx.y * 8 + r;
  float acc[8] = {};
#pragma unroll Unroll
  for (int g = 0; g < 40; ++g) {
    const half* weights =
        w + (blockIdx.x * 40 * 64 + g * 64 + split * 8 + r) * 8;
    const uint4 lo = *reinterpret_cast<const uint4*>(weights);
    const uint4 hi = *reinterpret_cast<const uint4*>(weights + 32 * 8);
    uint4 a = {}, b = {};
    if (row < m) {
      const half* input = x + row * 2560 + split * 640 + g * 16;
      a = *reinterpret_cast<const uint4*>(input);
      b = *reinterpret_cast<const uint4*>(input + 8);
    }
    GDN_MMA(acc, a.x, a.y, lo.x, lo.y);
    GDN_MMA(acc, a.z, a.w, lo.z, lo.w);
    GDN_MMA(acc, b.x, b.y, hi.x, hi.y);
    GDN_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr =
        blockIdx.y * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
    const int cc =
        blockIdx.x * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    float value = __shfl_sync(0xffffffff, acc[i], lane & ~12);
#pragma unroll
    for (int s = 1; s < 4; ++s)
      value = __fadd_rn(
          value, __shfl_sync(0xffffffff, acc[i], (lane & ~12) | (s << 2)));
    if (split == 0 && rr < m) out[rr * 512 + cc] = __float2half_rn(value);
  }
}

// The shared expert's original cuBLASLt projection uses eight K320
// partitions. Preserve each FP16 partial before the ordered FP32 reduction.
__global__ __launch_bounds__(32, 8) void shared_up_batch_kernel(const half* x,
                                                                const half* w,
                                                                half* partial,
                                                                int m) {
  const int lane = threadIdx.x, quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int row = blockIdx.y * 8 + r, col = quad * 8 + r;
  const int split = blockIdx.z;
  float acc[8] = {};
#pragma unroll 20
  for (int g = split * 20; g < (split + 1) * 20; ++g) {
    const int offset = (blockIdx.x * 160 + g) * 512 + col * 8;
    const uint4 lo = *reinterpret_cast<const uint4*>(w + offset);
    const uint4 hi = *reinterpret_cast<const uint4*>(w + offset + 256);
    uint4 a = {}, b = {};
    if (row < m) {
      a = *reinterpret_cast<const uint4*>(x + row * 2560 + g * 16);
      b = *reinterpret_cast<const uint4*>(x + row * 2560 + g * 16 + 8);
    }
    GDN_MMA(acc, a.x, a.y, lo.x, lo.y);
    GDN_MMA(acc, a.z, a.w, lo.z, lo.w);
    GDN_MMA(acc, b.x, b.y, hi.x, hi.y);
    GDN_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int rr =
        blockIdx.y * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
    const int cc = blockIdx.x * 32 + quad * 8 +
                   ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
    if (rr < m) partial[(split * m + rr) * 320 + cc] = __float2half_rn(acc[i]);
  }
}

__global__ void shared_up_reduce_silu_kernel(const half* partial, half* out,
                                             int m) {
  const int k = blockIdx.x * 128 + threadIdx.x;
  if (k >= m * 160) return;
  const int r = k / 160, c = k % 160, index = r * 320 + c;
  float gate = __half2float(partial[index]);
  float up = __half2float(partial[index + 160]);
#pragma unroll
  for (int s = 1; s < 8; ++s) {
    gate = __fadd_rn(gate, __half2float(partial[s * m * 320 + index]));
    up = __fadd_rn(up, __half2float(partial[s * m * 320 + index + 160]));
  }
  const half g = __float2half_rn(gate), u = __float2half_rn(up);
  const float value = __half2float(g);
  // Match the native packed SiLU: full expf, FP16 activation, then FP16 mul.
  const half activated = __float2half_rn(value / __fadd_rn(1.f, expf(-value)));
  out[k] = __hmul(activated, u);
}

#undef GDN_MMA

void router_batch(torch::Tensor output, torch::Tensor x, torch::Tensor packed) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) >= 2 &&
                  x.size(0) <= 16 && x.size(1) == 2560,
              "SM70 batch router requires M2..16, K2560");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, x, packed})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                    t.scalar_type() == at::kHalf &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Batch router requires aligned contiguous FP16 storage");
  TORCH_CHECK(packed.sizes() == at::IntArrayRef({64, 40, 2, 4, 8, 8}) &&
                  output.sizes() == at::IntArrayRef({x.size(0), 512}),
              "Invalid batch router geometry");
  router_split_quad_kernel<40><<<dim3(64, (x.size(0) + 7) / 8), 32, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(packed.data_ptr()),
      reinterpret_cast<half*>(output.data_ptr()), x.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shared_up_batch(torch::Tensor output, torch::Tensor partial,
                     torch::Tensor x, torch::Tensor packed) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 &&
                  (x.size(0) == 5 || x.size(0) == 10) && x.size(1) == 2560,
              "Shared expert batch projection requires M5/M10, K2560");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  for (const auto& t : {output, partial, x, packed})
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() &&
                    t.scalar_type() == at::kHalf &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "Shared expert batch requires aligned contiguous FP16 storage");
  const int m = x.size(0);
  TORCH_CHECK(packed.sizes() == at::IntArrayRef({10, 160, 2, 32, 8}) &&
                  partial.sizes() == at::IntArrayRef({8, m, 320}) &&
                  output.sizes() == at::IntArrayRef({m, 160}),
              "Invalid shared expert batch geometry");
  const auto stream = at::cuda::getCurrentCUDAStream();
  shared_up_batch_kernel<<<dim3(10, (m + 7) / 8, 8), 32, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(packed.data_ptr()),
      reinterpret_cast<half*>(partial.data_ptr()), m);
  shared_up_reduce_silu_kernel<<<(m * 160 + 127) / 128, 128, 0, stream>>>(
      reinterpret_cast<const half*>(partial.data_ptr()),
      reinterpret_cast<half*>(output.data_ptr()), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "qwen38_shared_up_batch_sm70_out(Tensor(a!) out, Tensor(b!) partial, "
      "Tensor x, Tensor packed) -> ()");
  m.def(
      "qwen38_router_batch_sm70_out(Tensor(a!) out, Tensor x, Tensor packed) "
      "-> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("qwen38_shared_up_batch_sm70_out", &shared_up_batch);
  m.impl("qwen38_router_batch_sm70_out", &router_batch);
}
