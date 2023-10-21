#pragma once

#include <torch/extension.h>

#ifdef QUIK_WITH_CUSPARSELT
#include <cusparseLt.h>
#endif

namespace QUIK::matmul {
torch::Tensor int4MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);

torch::Tensor int4OutputInt8MatmulCUDA(const torch::Tensor &A,
                                       const torch::Tensor &B);

torch::Tensor int4SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E);

torch::Tensor int4OutputInt8SpMatmulCUDA(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &E);

torch::Tensor int4ReorderMeta(const torch::Tensor &E);

torch::Tensor int4GenRandomSparseMeta(int M, int K);

torch::Tensor int4Uncompress(const torch::Tensor &A, const torch::Tensor &E,
                             int M, int K);

torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);

torch::Tensor int8OutputInt8MatmulCUDA(const torch::Tensor &A,
                                       const torch::Tensor &B);

torch::Tensor int8SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E);

torch::Tensor int8OutputInt8SpMatmulCUDA(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &E);

torch::Tensor int8ReorderMeta(const torch::Tensor &E);

torch::Tensor int8GenRandomSparseMeta(int M, int K);

torch::Tensor int8Uncompress(const torch::Tensor &A, const torch::Tensor &E,
                             int M, int K);

#ifdef QUIK_WITH_CUSPARSELT

class CusparseLtInt8SpMatmul {
 private:
  using pytorchIndex = torch::IntArrayRef::value_type;
  const torch::Tensor &A_, &B_;
  torch::Tensor A_compressed_;
  int8_t *dA_ = nullptr;
  int8_t *dB_ = nullptr;
  int8_t *dA_compressed_ = nullptr;

  int64_t M_, N_, K_;

  cusparseOperation_t opA_, opB_;

  cusparseLtHandle_t *handle_;
  cusparseLtMatDescriptor_t *matA_, *matB_, *matC_;
  cusparseLtMatmulDescriptor_t *matmul_;
  cusparseLtMatmulAlgSelection_t *alg_sel_;
  cusparseLtMatmulPlan_t *plan_;
  cudaStream_t stream_ = nullptr;
  cudaStream_t *streams_ = nullptr;
  int num_streams_ = 0;

  float alpha_, beta_;

 public:
  CusparseLtInt8SpMatmul() = delete;
  CusparseLtInt8SpMatmul(const torch::Tensor &A, const torch::Tensor &B,
                         const int alg);
  ~CusparseLtInt8SpMatmul();

  void compress();
  torch::Tensor matmulBy(const torch::Tensor &B);
  torch::Tensor matmulDefault();
};

#endif
}  // namespace QUIK::matmul
