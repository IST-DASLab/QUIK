#include "symmetric/symmetric.h"

#include <torch/extension.h>

#include "symmetric/symmetric_internal.h"

namespace QUIK::symmetric {
torch::Tensor quantize(const torch::Tensor &src, const torch::Tensor &scale) {
  torch::checkAllContiguous("quantize", {{src, "src", 0}, {scale, "scale", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("quantize", {src, scale}, at::DeviceType::CUDA);
  return quantizeCUDA(src, scale);
}

torch::Tensor dequantize(const torch::Tensor &x, const torch::Tensor &scaleRow,
                         const torch::Tensor &scaleCol,
                         const torch::Tensor &y) {
  torch::checkAllContiguous("dequantize", {{x, "x", 0},
                                           {scaleRow, "scaleRow", 1},
                                           {scaleCol, "scaleCol", 2},
                                           {y, "y", 3}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("dequantize", {x, scaleRow, scaleCol, y},
                         at::DeviceType::CUDA);
  return dequantizeCUDA(x, scaleRow, scaleCol, y);
}

void buildSubmodule(py::module &mod) {
  py::module m =
      mod.def_submodule("symmetric", "Symmetric Quantization Functions");
  m.def("quantize", &quantize,
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
        "torch.Tensor(M x 1, FP16, CUDA))\n"
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale))",
        py::arg("src"), py::arg("scale"));

  m.def("dequantize", &dequantize,
        "input (x: torch.Tensor(M x N, INT32), scale_row: torch.Tensor(M x 1, "
        "FP16), scale_col: torch.Tensor(1 x N, FP16), y: torch.Tensor(M x N, "
        "FP16))\n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = x * scale_row * scale_col + y",
        py::arg("x"), py::arg("scale_row"), py::arg("scale_col"), py::arg("y"));
}
}  // namespace QUIK::symmetric