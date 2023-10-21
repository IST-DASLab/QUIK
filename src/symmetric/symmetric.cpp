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

torch::Tensor int4FusedDequantize(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y) {
  torch::checkAllContiguous("int4FusedDequantize", {{A, "A", 0},
                                                    {B, "B", 1},
                                                    {scale_row, "scale_row", 2},
                                                    {scale_col, "scale_col", 3},
                                                    {y, "y", 4}});
  torch::checkDeviceType("int4FusedDequantize", {A, B, scale_row, scale_col, y},
                         at::DeviceType::CUDA);
  return int4FusedDequantizeCUDA(A, B, scale_row, scale_col, y);
}

torch::Tensor int8FusedDequantize(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y) {
  torch::checkAllContiguous("int8FusedDequantize", {{A, "A", 0},
                                                    {B, "B", 1},
                                                    {scale_row, "scale_row", 2},
                                                    {scale_col, "scale_col", 3},
                                                    {y, "y", 4}});
  torch::checkDeviceType("int8FusedDequantize", {A, B, scale_row, scale_col, y},
                         at::DeviceType::CUDA);
  return int8FusedDequantizeCUDA(A, B, scale_row, scale_col, y);
}

torch::Tensor int4SpFusedDequantize(const torch::Tensor &A,
                                    const torch::Tensor &B,
                                    const torch::Tensor &E,
                                    const torch::Tensor &scale_row,
                                    const torch::Tensor &scale_col,
                                    const torch::Tensor &y) {
  torch::checkAllContiguous("int4SpFusedDequantize",
                            {{A, "A", 0},
                             {B, "B", 1},
                             {scale_row, "scale_row", 2},
                             {scale_col, "scale_col", 3},
                             {y, "y", 4}});
  torch::checkDeviceType("int4SpFusedDequantize",
                         {A, B, scale_row, scale_col, y}, at::DeviceType::CUDA);
  return int4SpFusedDequantizeCUDA(A, B, E, scale_row, scale_col, y);
}

torch::Tensor int8SpFusedDequantize(const torch::Tensor &A,
                                    const torch::Tensor &B,
                                    const torch::Tensor &E,
                                    const torch::Tensor &scale_row,
                                    const torch::Tensor &scale_col,
                                    const torch::Tensor &y) {
  torch::checkAllContiguous("int8SpFusedDequantize",
                            {{A, "A", 0},
                             {B, "B", 1},
                             {scale_row, "scale_row", 2},
                             {scale_col, "scale_col", 3},
                             {y, "y", 4}});
  torch::checkDeviceType("int8SpFusedDequantize",
                         {A, B, scale_row, scale_col, y}, at::DeviceType::CUDA);
  return int8SpFusedDequantizeCUDA(A, B, E, scale_row, scale_col, y);
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

  m.def(
      "int4FusedDequantize", &int4FusedDequantize,
      "input: (A: torch.Tensor(M x K/2, UINT8, CUDA), B: torch.Tensor(N x K/2, "
      "UINT8, CUDA)\n"
      "scale_row: torch.Tensor(M x 1, FP16, CUDA), scale_col: torch.Tensor(1 x "
      "N, FP16, CUDA)"
      "y: torch.Tensor(M x N, FP16, CUDA))"
      "output: torch.Tensor(M x N, INT32, CUDA)\n"
      "output = int4Unpacking(A) @ int4Unpacking(B)^T * scale_row * scale_cal "
      "+ y",
      py::arg("A"), py::arg("B"), py::arg("scale_row"), py::arg("scale_col"),
      py::arg("y"));

  m.def("int8FusedDequantize", &int8FusedDequantize,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA)\n"
        "scale_row: torch.Tensor(M x 1, FP16, CUDA), scale_col: torch.Tensor(1 "
        "x N, FP16, CUDA)"
        "y: torch.Tensor(M x N, FP16, CUDA))"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T * scale_row * scale_cal + y",
        py::arg("A"), py::arg("B"), py::arg("scale_row"), py::arg("scale_col"),
        py::arg("y"));

  m.def(
      "int4SpFusedDequantize", &int4SpFusedDequantize,
      "input: (A: torch.Tensor(M x K/4, UINT8, CUDA), B: torch.Tensor(N x K/2, "
      "UINT8, CUDA)\n"
      "E: torch.Tensor(M x 32, UINT8, CUDA)"
      "scale_row: torch.Tensor(M x 1, FP16, CUDA), scale_col: torch.Tensor(1 x "
      "N, FP16, CUDA)"
      "y: torch.Tensor(M x N, FP16, CUDA))"
      "output: torch.Tensor(M x N, INT32, CUDA)\n"
      "output = A @ B^T * scale_row * scale_cal + y",
      py::arg("A"), py::arg("B"), py::arg("E"), py::arg("scale_row"),
      py::arg("scale_col"), py::arg("y"));

  m.def("int8SpFusedDequantize", &int8SpFusedDequantize,
        "input: (A: torch.Tensor(M x K/2, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "E: torch.Tensor(M x 32, UINT8, CUDA)"
        "scale_row: torch.Tensor(M x 1, FP16, CUDA), scale_col: torch.Tensor(1 "
        "x N, FP16, CUDA)"
        "y: torch.Tensor(M x N, FP16, CUDA))"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T * scale_row * scale_cal + y",
        py::arg("A"), py::arg("B"), py::arg("E"), py::arg("scale_row"),
        py::arg("scale_col"), py::arg("y"));
}

}  // namespace QUIK::symmetric
