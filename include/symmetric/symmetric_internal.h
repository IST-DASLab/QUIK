#pragma once
#include <torch/extension.h>

namespace QUIK::symmetric {
torch::Tensor int4QuantizationCUDA(const torch::Tensor &src,
                                   const torch::Tensor &scale);

torch::Tensor int8QuantizationCUDA(const torch::Tensor &src,
                                   const torch::Tensor &scale);

template <typename T>
torch::Tensor dequantizationCUDA(const torch::Tensor &x,
                                 const torch::Tensor &scaleRow,
                                 const torch::Tensor &scaleCol,
                                 const torch::Tensor &y);

extern template torch::Tensor dequantizationCUDA<int8_t>(
    const torch::Tensor &x, const torch::Tensor &scaleRow,
    const torch::Tensor &scaleCol, const torch::Tensor &y);

extern template torch::Tensor dequantizationCUDA<torch::Half>(
    const torch::Tensor &x, const torch::Tensor &scaleRow,
    const torch::Tensor &scaleCol, const torch::Tensor &y);

extern template torch::Tensor dequantizationCUDA<int>(
    const torch::Tensor &x, const torch::Tensor &scaleRow,
    const torch::Tensor &scaleCol, const torch::Tensor &y);

torch::Tensor int4FusedDequantizeCUDA(const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      const torch::Tensor &scale_row,
                                      const torch::Tensor &scale_col,
                                      const torch::Tensor &y);

torch::Tensor int8FusedDequantizeCUDA(const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      const torch::Tensor &scale_row,
                                      const torch::Tensor &scale_col,
                                      const torch::Tensor &y);

torch::Tensor int4SpFusedDequantizeCUDA(const torch::Tensor &A,
                                        const torch::Tensor &B,
                                        const torch::Tensor &E,
                                        const torch::Tensor &scale_row,
                                        const torch::Tensor &scale_col,
                                        const torch::Tensor &y);

torch::Tensor int8SpFusedDequantizeCUDA(const torch::Tensor &A,
                                        const torch::Tensor &B,
                                        const torch::Tensor &E,
                                        const torch::Tensor &scale_row,
                                        const torch::Tensor &scale_col,
                                        const torch::Tensor &y);

}  // namespace QUIK::symmetric
