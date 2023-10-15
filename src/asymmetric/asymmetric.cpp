#include "asymmetric/asymmetric.h"

#include <torch/extension.h>

#include "asymmetric/asymmetric_internal.h"

namespace QUIK::asymmetric {
torch::Tensor quantizeOld(const torch::Tensor &src, const torch::Tensor &meta,
                       int bits) {
  torch::checkAllContiguous("quantize", {{src, "src", 0}, {meta, "meta", 1}});
  TORCH_CHECK(bits == 4 or bits == 8, "bits argument must be either 4 or 8");
  torch::checkDeviceType("quantize", {src, meta}, at::DeviceType::CUDA);
  return quantizeCUDAOld(src, meta, bits);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize(
    const torch::Tensor &src, const torch::Tensor &int_indices,
    const torch::Tensor &fp_indices, int bits) {
  torch::checkContiguous("quantize", {{src, "src", 0}});
  TORCH_CHECK(bits == 4 or bits == 8, "bits argument must be either 4 or 8");
  torch::checkDeviceType("quantize", {src}, at::DeviceType::CUDA);
  torch::checkDeviceType("quantize", {int_indices}, at::DeviceType::CUDA);
  torch::checkDeviceType("quantize", {fp_indices}, at::DeviceType::CUDA);
  return quantizeCUDA(src, int_indices, fp_indices, bits);
}

torch::Tensor dequantize(const torch::Tensor &x, const torch::Tensor &meta,
                         const torch::Tensor &scaleCol,
                         const torch::Tensor &wReduced, const torch::Tensor &y,
                         int bits) {
  torch::checkAllContiguous(
      "dequantize",
      {{x, "x", 0}, {meta, "meta", 1}, {scaleCol, "scaleCol", 2}, {y, "y", 3}});
  torch::checkDeviceType("dequantize", {x, meta, scaleCol, y},
                         at::DeviceType::CUDA);
  TORCH_CHECK(bits == 4 or bits == 8, "bits argument must be either 4 or 8");
  return dequantizeCUDA(x, meta, scaleCol, wReduced, y, bits);
}

torch::Tensor find_meta(const torch::Tensor &src, const unsigned bits) {
  torch::checkDeviceType("find_meta", {src}, at::DeviceType::CUDA);
  TORCH_CHECK(bits == 4 or bits == 8, "bits argument must be either 4 or 8");
  return findMaxMinMetaCUDA(src, bits);
}

void buildSubmodule(py::module &mod) {
  py::module m =
      mod.def_submodule("asymmetric", "Asymmetric Quantization Functions");
  m.def("quantizeOld", &quantizeOld,
        "input: (src: torch.Tensor(M x N, FP16, CUDA),\n"
        "meta: torch.Tensor(2M x 1, FP16, CUDA)\n"
        "bits: int 4 or 8 \n"
        "output: torch.Tensor(M x ceil(N * bits / 8), UINT8, CUDA)\n"
        "output = int{bits}Packing(int{bits}Rounding((source - meta.zero) / "
        "meta.scale - 2^{bits - 1}))",
        py::arg("src"), py::arg("meta"), py::arg("bits"));

  m.def("quantize", &quantize,
        "input: (src: torch.Tensor(M x N, FP16, CUDA),\n"
        "int_indices: (torch.Tensor(NUM_INTs x 1, LONG, CUDA)),\n"
        "fp_indices: (torch.Tensor(NUM_FPs x 1, LONG, CUDA)),\n"
        "bits: int 4 or 8 \n"
        "output: tuple: dst: torch.Tensor(M x ceil(N * bits / 8), UINT8, CUDA)\n"
        "meta: torch.Tensor(2M x 1, FP16, CUDA)\n"
        "fp_x: torch.Tensor(M x NUM_FPs, FP16, CUDA)\n"
        "dst = int{bits}Packing(int{bits}Rounding((source - meta.zero) / "
        "meta.scale - 2^{bits - 1}))",
        "meta: M * [scale, zero]",
        "meta: full precision features",
        py::arg("src"), py::arg("int_indices"), py::arg("fp_indices"), py::arg("bits"));

  m.def("dequantize", &dequantize,
        "input (x: torch.Tensor(M x N, INT32),\n"
        "meta: torch.Tensor(2M x 1, FP16),\n"
        "scale_col: torch.Tensor(1 x N, FP16),\n"
        "wReduced: torch.Tensor(1 x N, FP16))\n"
        "y: torch.Tensor(M x N, FP16))\n"
        "bits: 4 or 8 \n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = (x + (meta.zero / scale_row + 2^{bits - 1}) * wReduced) * "
        "meta.scale * "
        "scale_col  + y",
        py::arg("x"), py::arg("meta"), py::arg("scale_col"),
        py::arg("wReduced"), py::arg("y"), py::arg("bits"));

  m.def("find_meta", &find_meta,
        "input: (src: torch.Tensor(M x N, FP16, CUDA),\n"
        "bits: int 4 or 8 \n"
        "output: torch.Tensor(2M), FP16, CUDA)\n"
        "output = M * [scale, zero]",
        py::arg("src"), py::arg("bits"));
}
}  // namespace QUIK::asymmetric