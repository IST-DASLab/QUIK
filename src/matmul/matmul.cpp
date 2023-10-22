#include "matmul/matmul.h"

#include <torch/extension.h>

#include "matmul/matmul_internal.h"

namespace QUIK::matmul {
torch::Tensor int4Matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("int4Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4Matmul", {A, B}, at::DeviceType::CUDA);
  return int4MatmulCUDA(A, B);
}

torch::Tensor int4OutputInt8Matmul(const torch::Tensor &A,
                                   const torch::Tensor &B) {
  torch::checkAllContiguous("int4OutputInt8Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4OutputInt8Matmul", {A, B}, at::DeviceType::CUDA);
  return int4OutputInt8MatmulCUDA(A, B);
}

torch::Tensor int4SpMatmul(const torch::Tensor &A, const torch::Tensor &B,
                           const torch::Tensor &E) {
  torch::checkAllContiguous("int4SpMatmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4SpMatmul", {A, B, E}, at::DeviceType::CUDA);
  return int4SpMatmulCUDA(A, B, E);
}

torch::Tensor int4OutputInt8SpMatmul(const torch::Tensor &A,
                                     const torch::Tensor &B,
                                     const torch::Tensor &E) {
  torch::checkAllContiguous("int4OutputInt8SpMatmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4OutputInt8SpMatmul", {A, B, E},
                         at::DeviceType::CUDA);
  return int4OutputInt8SpMatmulCUDA(A, B, E);
}

torch::Tensor int8Matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8Matmul", {A, B}, at::DeviceType::CUDA);
  return int8MatmulCUDA(A, B);
}

torch::Tensor int8OutputInt8Matmul(const torch::Tensor &A,
                                   const torch::Tensor &B) {
  torch::checkAllContiguous("int8OutputInt8Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8OutputInt8Matmul", {A, B}, at::DeviceType::CUDA);
  return int8OutputInt8MatmulCUDA(A, B);
}

torch::Tensor int8SpMatmul(const torch::Tensor &A, const torch::Tensor &B,
                           const torch::Tensor &E) {
  torch::checkAllContiguous("int8SpMatmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8SpMatmul", {A, B, E}, at::DeviceType::CUDA);
  return int8SpMatmulCUDA(A, B, E);
}

torch::Tensor int8OutputInt8SpMatmul(const torch::Tensor &A,
                                     const torch::Tensor &B,
                                     const torch::Tensor &E) {
  torch::checkAllContiguous("int8OutputInt8Matmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8OutputInt8Matmul", {A, B, E},
                         at::DeviceType::CUDA);
  return int8OutputInt8SpMatmulCUDA(A, B, E);
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
  py::module m = mod.def_submodule("matmul", "Matmul Functions");
  m.def("int4Matmul", &int4Matmul,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));

  m.def(
      "int4SpMatmul", &int4SpMatmul,
      "input: (A: torch.Tensor(M x K / 2 / 2, UINT8, CUDA), B: torch.Tensor(N "
      "x K / 2, UINT8, CUDA), E: torch.Tensor(M x K / 2 / 2, INT32, CUDA)\n"
      "output: torch.Tensor(M x N, INT32, CUDA)\n"
      "output = sp2dn(int4Unpacking(A), E) @ int4Unpacking(B)^T",
      py::arg("A"), py::arg("B"), py::arg("E"));

  m.def("int4ReorderMeta", &int4ReorderMeta,
        "input: (E: torch.Tensor(M x ?, INT32, CPU), N)\n"
        "output: torch.Tensor(INT32, CPU)\n"
        "output = cutlass::reorder_meta(E)",
        py::arg("E"));

  m.def("int4GenRandomSparseMeta", &int4GenRandomSparseMeta, "");
  m.def("int4Uncompress", &int4Uncompress, "");

  m.def("int8Matmul", &int8Matmul,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"));

  m.def("int8SpMatmul", &int8SpMatmul,
        "input: (A: torch.Tensor(M x K / 2, UINT8, CUDA), B: torch.Tensor(N x "
        "K / 2, UINT8, CUDA), E: torch.Tensor(M x K / 2 / 2, INT32, CUDA)\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = sp2dn(A, E) @ B^T",
        py::arg("A"), py::arg("B"), py::arg("E"));

  m.def("int8ReorderMeta", &int8ReorderMeta,
        "input: (E: torch.Tensor(M x ?, INT32, CPU), N)\n"
        "output: torch.Tensor(INT32, CPU)\n"
        "output = cutlass::reorder_meta(E)",
        py::arg("E"));

  m.def("int8GenRandomSparseMeta", &int8GenRandomSparseMeta, "");
  m.def("int8Uncompress", &int8Uncompress, "");

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

  py::class_<CusparseLtInt8SpMatmul>(m, "CusparseLtInt8SpMatmul")
      .def(py::init<const torch::Tensor &, const torch::Tensor &, const int>(),
           "", py::arg("A"), py::arg("B"), py::arg("alg") = 0)
      .def("compress", &CusparseLtInt8SpMatmul::compress, "")
      .def("matmul_by", &CusparseLtInt8SpMatmul::matmulBy, "", py::arg("B"));
}
}  // namespace QUIK::matmul
