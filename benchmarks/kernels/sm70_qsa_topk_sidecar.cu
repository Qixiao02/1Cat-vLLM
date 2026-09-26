// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "../../csrc/qsa_lexicographic_topk.cuh"

namespace {
// kCandidate: the production launcher (decode kernel when num_rows == 1,
// generic kernel otherwise). kBaseline: always the generic kernel.
// kDecodeRows: the decode-specialised kernel for every row (grid = num_rows).
enum class Mode { kCandidate, kBaseline, kDecodeRows };

void topk(torch::Tensor logits, torch::Tensor lengths, torch::Tensor output,
          int64_t k, Mode mode) {
  TORCH_CHECK(logits.is_cuda() && lengths.is_cuda() && output.is_cuda(),
              "QSA tensors must be CUDA");
  TORCH_CHECK(
      logits.device() == lengths.device() && logits.device() == output.device(),
      "QSA device mismatch");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 &&
                  lengths.scalar_type() == torch::kInt32 &&
                  output.scalar_type() == torch::kInt32,
              "QSA dtype mismatch");
  TORCH_CHECK(k == 512 && logits.dim() == 2 && lengths.dim() == 1 &&
                  output.dim() == 2 && lengths.numel() == logits.size(0) &&
                  output.size(0) == logits.size(0) && output.size(1) == k &&
                  logits.stride(1) == 1 && lengths.is_contiguous() &&
                  output.is_contiguous(),
              "QSA shape mismatch");
  if (!logits.size(0)) return;
  const c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (mode == Mode::kDecodeRows) {
    vllm::qsa::launch_qsa_lexicographic_decode_topk_rows<512>(
        logits.data_ptr<float>(), lengths.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(), logits.size(0), logits.size(1),
        logits.stride(0), stream);
  } else if (mode == Mode::kBaseline) {
    vllm::qsa::qsa_lexicographic_topk_kernel<512>
        <<<logits.size(0), vllm::qsa::kLexicographicTopKThreads, 0, stream>>>(
            logits.data_ptr<float>(), lengths.data_ptr<int32_t>(),
            output.data_ptr<int32_t>(), logits.size(0), logits.size(1),
            logits.stride(0));
  } else {
    vllm::qsa::launch_qsa_lexicographic_topk<512>(
        logits.data_ptr<float>(), lengths.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(), logits.size(0), logits.size(1),
        logits.stride(0), stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void candidate(torch::Tensor x, torch::Tensor n, torch::Tensor y, int64_t k) {
  topk(x, n, y, k, Mode::kCandidate);
}
void baseline(torch::Tensor x, torch::Tensor n, torch::Tensor y, int64_t k) {
  topk(x, n, y, k, Mode::kBaseline);
}
void decode_rows(torch::Tensor x, torch::Tensor n, torch::Tensor y, int64_t k) {
  topk(x, n, y, k, Mode::kDecodeRows);
}
// Version of the qsa_lexicographic_topk candidate (unchanged). The optional
// qsa_lexicographic_topk_decode_rows op (design_4 [MR5]) is detected by its
// presence, not by this number.
int64_t version() { return 1; }
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_qsa_sm70, ops) {
  ops.def(
      "qsa_lexicographic_topk(Tensor logits, Tensor lengths, "
      "Tensor(a!) output, int top_k) -> ()");
  ops.impl("qsa_lexicographic_topk", torch::kCUDA, &candidate);
  ops.def(
      "qsa_lexicographic_topk_decode_rows(Tensor logits, Tensor lengths, "
      "Tensor(a!) output, int top_k) -> ()");
  ops.impl("qsa_lexicographic_topk_decode_rows", torch::kCUDA, &decode_rows);
  ops.def("decode_specialization_version() -> int", &version);
}
TORCH_LIBRARY_FRAGMENT(_C_qsa_verify, ops) {
  ops.def(
      "baseline(Tensor logits, Tensor lengths, Tensor(a!) output, int k) -> "
      "()");
  ops.impl("baseline", torch::kCUDA, &baseline);
}
