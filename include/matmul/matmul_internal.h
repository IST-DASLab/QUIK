#pragma once
#include <torch/extension.h>

namespace QUIK::matmul {
torch::Tensor int4MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);
torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);

torch::Tensor int4SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E);

torch::Tensor reorderMeta(const torch::Tensor &E);

torch::Tensor genRandomSparseMeta(int M, int K);

torch::Tensor uncompress(const torch::Tensor &A, const torch::Tensor &E, int M,
                         int K);
}  // namespace QUIK::matmul