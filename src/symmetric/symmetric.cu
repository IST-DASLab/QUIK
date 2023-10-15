#include "int4.h"
#include "symmetric/symmetric_internal.h"
#include "util.h"

namespace QUIK::symmetric {
__global__ void quantizeCUDAKernel(Int4Storage *__restrict__ dst,
                                       const torch::Half *__restrict__ scale,
                                       const torch::Half *__restrict__ src,
                                       const unsigned rows,
                                       const unsigned colsSrc,
                                       unsigned colsDst) {
  const unsigned row = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned colDst = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= rows || colDst * kElementsPerVector >= colsSrc) {
    return;
  }
  Int4Storage storage;
  memset(&storage, 0, sizeof(storage));
  const unsigned id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
  for (int i = 0; i < kElementsPerVector; ++i) {
    bool safe = (colDst * kElementsPerVector + i) < colsSrc;
    if (safe) {
      __half data = __hdiv(src[id + i], scale[row]);
      Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(
          __half2int_rn(data));
    }
  }
  dst[colDst + row * colsDst] = storage;
}

torch::Tensor quantizeCUDA(const torch::Tensor &src,
                               const torch::Tensor &scale) {
  torch::checkSameGPU("quantize", {src, "src", 0}, {scale, "scale", 1});
  torch::checkSize("quantize", torch::TensorArg{scale, "scale", 1}, 0,
                   src.size(0));
  unsigned rows = src.size(0);
  unsigned colsSrc = src.size(1);
  unsigned colsDst = (colsSrc - 1) / kElementsPerVector + 1;
  auto dst =
      torch::empty({rows, colsDst},
                   torch::dtype(util::TorchDtypeDispatcher<Int4Storage>::value)
                       .device(src.device()));
  dim3 block{std::min<unsigned>(colsDst, 32), std::min<unsigned>(rows, 16)};
  dim3 grid{(colsDst - 1) / block.x + 1, (rows - 1) / block.y + 1};
  quantizeCUDAKernel<<<grid, block>>>(
      dst.data_ptr<Int4Storage>(), scale.data_ptr<torch::Half>(),
      src.data_ptr<torch::Half>(), rows, colsSrc, colsDst);
  return dst;
}

__global__ void dequantizationKernel(torch::Half *__restrict__ out,
                                     const int *__restrict__ x,
                                     const torch::Half *__restrict__ scaleRow,
                                     const torch::Half *__restrict__ scaleCol,
                                     const torch::Half *__restrict__ y,
                                     const unsigned rows, const unsigned cols) {
  const unsigned row = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned col = threadIdx.x + blockIdx.x * blockDim.x;
  if (col >= cols) {
    return;
  }

  if (row >= rows) {
    return;
  }

  __half xElement = __int2half_rn(x[col + row * cols]);

  out[col + row * cols] = __hfma(__hmul(xElement, scaleRow[row]), scaleCol[col],
                                 y[col + row * cols]);
}

torch::Tensor dequantizeCUDA(const torch::Tensor &x,
                                 const torch::Tensor &scaleRow,
                                 const torch::Tensor &scaleCol,
                                 const torch::Tensor &y) {
  torch::checkAllSameGPU("dequantize", {{x, "x", 0},
                                          {scaleRow, "scaleRow", 1},
                                          {scaleCol, "scaleCol", 2},
                                          {y, "y", 3}});
//  torch::checkSameNumel("dequantize", torch::TensorArg{x, "x", 0}, torch::TensorArg{y, "y", 1});
  unsigned rows = x.size(0);
  unsigned cols = x.size(1);
  torch::checkSize("dequantize", torch::TensorArg{scaleRow, "scaleRow", 1},
                   0, rows);
  torch::checkSize("dequantize", torch::TensorArg{scaleCol, "scaleCol", 2}, 0,
                   cols);
  auto out = torch::empty_like(y);
  dim3 block{std::min<unsigned>(cols, 16),
             std::min<unsigned>((rows - 1) + 1, 16)};
  dim3 grid{(cols - 1) / block.x + 1, (rows - 1) / block.y + 1};
  dequantizationKernel<<<grid, block>>>(
      out.data_ptr<torch::Half>(), x.data_ptr<int>(),
      scaleRow.data_ptr<torch::Half>(), scaleCol.data_ptr<torch::Half>(),
      y.data_ptr<torch::Half>(), rows, cols);
  return out;
}

}  // namespace QUIK::symmetric