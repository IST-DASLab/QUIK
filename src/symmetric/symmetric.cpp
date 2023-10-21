#include "symmetric/symmetric.h"

#include <torch/extension.h>

#include "symmetric/symmetric_internal.h"

namespace QUIK::symmetric {

torch::Tensor quantize(const torch::Tensor &src, const torch::Tensor &scale,
                       const int bits) {
  torch::checkAllContiguous("quantize", {{src, "src", 0}, {scale, "scale", 1}});
  torch::checkDeviceType("quantize", {src, scale}, at::DeviceType::CUDA);
  switch (bits) {
    case 4:
      return int4QuantizationCUDA(src, scale);
    case 8:
      return int8QuantizationCUDA(src, scale);
    default:
      TORCH_CHECK(false, "Unsupported data type")
  }
}

torch::Tensor dequantize(const torch::Tensor &x, const torch::Tensor &scaleRow,
                         const torch::Tensor &scaleCol, const torch::Tensor &y,
                         const int bits) {
  torch::checkAllContiguous("dequantize", {{x, "x", 0},
                                           {scaleRow, "scaleRow", 1},
                                           {scaleCol, "scaleCol", 2},
                                           {y, "y", 3}});
  torch::checkDeviceType("dequantize", {x, scaleRow, scaleCol, y},
                         at::DeviceType::CUDA);
  switch (bits) {
    case 8:
      return dequantizationCUDA<int8_t>(x, scaleRow, scaleCol, y);
    case 16:
      return dequantizationCUDA<torch::Half>(x, scaleRow, scaleCol, y);
    case 32:
      return dequantizationCUDA<int>(x, scaleRow, scaleCol, y);
    default:
      TORCH_CHECK(false, "Unsupported data type")
  }
}

void buildSubmodule(py::module &mod) {
  py::module m = mod.def_submodule("symmetric", "Symmetric Functions");
  m.def("quantize", &quantize,
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
        "torch.Tensor(M x 1, FP16, CUDA))"
        "bits: int\n"
        "when bits equal 4: "
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n"
        "when bits equal 8: "
        "output: torch.Tensor(M x N, INT8, CUDA)\n"
        "output = int8Rounding(source / scale))\n",
        py::arg("src"), py::arg("scale"), py::arg("bits"));

  m.def("dequantize", &dequantize,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
        "FP16), scale_col: torch.Tensor(1 x N, FP16), y: torch.Tensor(M x N, "
        "FP16))\n"
        "bits: int\n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = x * scale_row * scale_col + y"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is FP16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("x"), py::arg("scale_row"), py::arg("scale_col"), py::arg("y"),
        py::arg("bits"));
}
}  // namespace QUIK::symmetric
