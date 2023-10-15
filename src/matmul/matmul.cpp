#include "matmul/matmul.h"

#include <torch/extension.h>

#include "matmul/matmul_internal.h"

namespace QUIK::matmul {
torch::Tensor int4Matmul(const torch::Tensor& A, const torch::Tensor& B) {
  torch::checkAllContiguous("int4Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4Matmul", {A, B}, at::DeviceType::CUDA);
  torch::checkScalarType("int4Matmul", {A, "A", 0}, at::ScalarType::Byte);
  torch::checkScalarType("int4Matmul", {B, "B", 1}, at::ScalarType::Byte);

  return int4MatmulCUDA(A, B);
}

torch::Tensor int4SpMatmul(const torch::Tensor& A, const torch::Tensor& B,
                           const torch::Tensor& E) {
  torch::checkAllContiguous("int4Matmul",
                            {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int4Matmul", {A, B, E}, at::DeviceType::CUDA);
  torch::checkScalarType("int4Matmul", {A, "A", 0}, at::ScalarType::Byte);
  torch::checkScalarType("int4Matmul", {B, "B", 1}, at::ScalarType::Byte);
  return int4SpMatmulCUDA(A, B, E);
};

torch::Tensor int8Matmul(const torch::Tensor& A, const torch::Tensor& B) {
  torch::checkAllContiguous("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("int8Matmul", {A, B}, at::DeviceType::CUDA);
  torch::checkScalarType("int8Matmul", {A, "A", 0}, at::ScalarType::Char);
  torch::checkScalarType("int8Matmul", {B, "B", 1}, at::ScalarType::Char);
  return int8MatmulCUDA(A, B);
}


void buildSubmodule(py::module& mod) {
  py::module m = mod.def_submodule("matmul", "Matmul Functions");
  m.def("int4Matmul", &int4Matmul,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int4Unpacking(A) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"));

  m.def("int8Matmul", &int8Matmul,
        "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
        "UINT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = int8Unpacking(A) @ int8Unpacking(B)^T",
        py::arg("A"), py::arg("B"));

  m.def("int4SpMatmul", &int4SpMatmul,
        "input: (A: torch.Tensor(M x K / 2, UINT8, CUDA), B: torch.Tensor(N x "
        "K / 2, UINT8, CUDA), E: torch.Tensor(M x K / 2 / 2, INT32, CUDA)\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = sp2dn(int4Unpacking(A), E) @ int4Unpacking(B)^T",
        py::arg("A"), py::arg("B"), py::arg("E"));

//  m.def("reorderMeta", &reorderMeta,
//        "input: (E: torch.Tensor(M x ?, INT32, CPU), N)\n"
//        "output: torch.Tensor(INT32, CPU)\n"
//        "output = cutlass::reorder_meta(E)",
//        py::arg("E"));

//  m.def("genRandomSparseMeta", &genRandomSparseMeta, "");
//  m.def("uncompress", &uncompress, "");
}
}  // namespace QUIK::matmul