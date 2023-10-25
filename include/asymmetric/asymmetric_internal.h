#pragma once
#include <torch/extension.h>

namespace QUIK::asymmetric {
torch::Tensor quantizeCUDAOld(const torch::Tensor &src,
                              const torch::Tensor &meta, const int bits);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantizeCUDA(
    const torch::Tensor &src, const torch::Tensor &int_indices,
    const torch::Tensor &fp_indices, const int bits);

torch::Tensor dequantizeCUDA(const torch::Tensor &x, const torch::Tensor &meta,
                             const torch::Tensor &scaleCol,
                             const torch::Tensor &wReduced,
                             const torch::Tensor &y, const int bits);

torch::Tensor findMaxMinMetaCUDA(const torch::Tensor &src, const unsigned bits);

torch::Tensor int4FusedDequantizeCUDA(
    const torch::Tensor &A, const torch::Tensor &B,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y);

torch::Tensor int8FusedDequantizeCUDA(
    const torch::Tensor &A, const torch::Tensor &B,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y);

torch::Tensor int4SpFusedDequantizeCUDA(
    const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &E,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y);

torch::Tensor int8SpFusedDequantizeCUDA(
    const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &E,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y);
}  // namespace QUIK::asymmetric