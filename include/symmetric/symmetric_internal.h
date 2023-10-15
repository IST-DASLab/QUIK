#pragma once
#include <torch/extension.h>

namespace QUIK::symmetric {
torch::Tensor quantizeCUDA(const torch::Tensor &src,
                               const torch::Tensor &scale);

torch::Tensor dequantizeCUDA(const torch::Tensor &x,
                                 const torch::Tensor &scaleRow,
                                 const torch::Tensor &scaleCol,
                                 const torch::Tensor &y);
}  // namespace QUIK::symmetric