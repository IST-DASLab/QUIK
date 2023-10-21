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

#ifdef QUIK_WITH_CUSPARSELT
  py::class_<CusparseLtInt8SpMatmul>(m, "CusparseLtInt8SpMatmul")
      .def(py::init<const torch::Tensor &, const torch::Tensor &, const int>(),
           "", py::arg("A"), py::arg("B"), py::arg("alg") = 0)
      .def("compress", &CusparseLtInt8SpMatmul::compress, "")
      .def("matmul_by", &CusparseLtInt8SpMatmul::matmulBy, "", py::arg("B"));
#endif
}
}  // namespace QUIK::matmul
