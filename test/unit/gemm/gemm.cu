/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <fstream>

#include "../common/cutlass_unit_test.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "symmetric/gemm/device/gemm_dequant.h"

template <typename Element, typename Layout>
static bool verify_footprint(cutlass::TensorView<Element, Layout> result_view,
                             cutlass::TensorView<Element, Layout> ref_view) {
  for (int r = 0; r < result_view.extent().row(); ++r) {
    for (int c = 0; c < result_view.extent().column(); ++c) {
      cutlass::MatrixCoord source_coord{r, c};
      if (result_view.at(source_coord) != ref_view.at(source_coord)) {
        return false;
      }
    }
  }

  return true;
}

template <typename ElementOutput, typename ElementAccumulator, typename Layout>
__global__ void dequantizationKernel(
    cutlass::TensorRef<ElementOutput, Layout> out,
    cutlass::TensorRef<ElementAccumulator, Layout> x,
    cutlass::TensorRef<ElementOutput, Layout> row_vec,
    cutlass::TensorRef<ElementOutput, Layout> col_vec,
    cutlass::TensorRef<ElementOutput, Layout> y, const int rows,
    const int cols) {
  const int row = threadIdx.y + blockIdx.y * blockDim.y;
  const int col = threadIdx.x + blockIdx.x * blockDim.x;
  if (col >= cols) {
    return;
  }

  if (row >= rows) {
    return;
  }
  cutlass::MatrixCoord output_coord{row, col};
  cutlass::MatrixCoord col_vec_coord{0, row};
  cutlass::MatrixCoord row_vec_coord{0, col};
  ElementOutput xElement = ElementOutput(x.at(output_coord));

  out.at(output_coord) =
      xElement * row_vec.at(row_vec_coord) * col_vec.at(col_vec_coord) +
      y.at(output_coord);
}
TEST(gemm_s8_s8_f16_f16, tensor_op_128x256x64_64x64x64) {
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = cutlass::half_t;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = int;
  using ElementD = ElementC;
  using LayoutD = LayoutC;

  using GemmDequant = cutlass::gemm::device::symmetric::GemmDequant<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;

  GemmDequant gemm_dequant_op;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;

  Gemm gemm_op;

  static const int block_size_x = 3;
  static const int block_size_y = 4;
  static const int M = GemmDequant::ThreadblockShape::kM * block_size_y;
  static const int N = GemmDequant::ThreadblockShape::kN * block_size_x;
  static const int K = GemmDequant::ThreadblockShape::kK * 2;
  cutlass::MatrixCoord A_extent{M, K};
  cutlass::MatrixCoord B_extent{N, K};
  cutlass::MatrixCoord C_extent{M, N};
  cutlass::MatrixCoord row_vec_extent{1, N};
  cutlass::MatrixCoord col_vec_extent{1, M};
  cutlass::HostTensor<ElementA, LayoutA> A(A_extent);
  cutlass::HostTensor<ElementB, LayoutB> B(B_extent);
  cutlass::HostTensor<ElementC, LayoutC> C(C_extent);
  cutlass::HostTensor<ElementD, LayoutD> D(C_extent);
  cutlass::HostTensor<ElementD, LayoutD> D_ref(C_extent);
  cutlass::HostTensor<ElementC, LayoutC> col_vec(col_vec_extent);
  cutlass::HostTensor<ElementC, LayoutC> row_vec(row_vec_extent);
  cutlass::HostTensor<ElementAccumulator, LayoutC> accumulator(C_extent);

  int max = -3;
  int min = 3;
  uint64_t seed = 1;

  cutlass::reference::host::TensorFill(D.host_view(), ElementC(0));
  cutlass::reference::host::TensorFill(D_ref.host_view(), ElementC(0));
  cutlass::reference::host::TensorFillRandomUniform(A.host_view(), seed + 0,
                                                    max, min);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(), seed + 1,
                                                    max, min);
  cutlass::reference::host::TensorFillRandomUniform(C.host_view(), seed + 2,
                                                    max, min);
  cutlass::reference::host::TensorFillRandomUniform(row_vec.host_view(),
                                                    seed + 3, max, min);
  cutlass::reference::host::TensorFillRandomUniform(col_vec.host_view(),
                                                    seed + 4, max, min);

  A.sync_device();
  B.sync_device();
  C.sync_device();
  D.sync_device();
  D_ref.sync_device();
  row_vec.sync_device();
  col_vec.sync_device();
  typename GemmDequant::Arguments arguments_dequant{{M, N, K},
                                                    A.device_ref(),
                                                    B.device_ref(),
                                                    C.device_ref(),
                                                    D.device_ref(),
                                                    row_vec.device_ref(),
                                                    col_vec.device_ref(),
                                                    ElementC(1)};

  auto status_dequant = gemm_dequant_op(arguments_dequant);

  ASSERT_EQ(status_dequant, cutlass::Status::kSuccess)
      << cutlassGetStatusString(status_dequant);

  typename Gemm::Arguments arguments{{M, N, K},
                                     A.device_ref(),
                                     B.device_ref(),
                                     accumulator.device_ref(),
                                     accumulator.device_ref(),
                                     {1, 0}};

  auto status = gemm_op(arguments);

  ASSERT_EQ(status, cutlass::Status::kSuccess)
      << cutlassGetStatusString(status);

  D.sync_host();
  accumulator.sync_host();

  dim3 block(32, 32);
  dim3 grid((N - 1) / block.x + 1, (M - 1) / block.y + 1);
  dequantizationKernel<ElementC, ElementAccumulator, LayoutC><<<grid, block>>>(
      D_ref.device_ref(), accumulator.device_ref(), row_vec.device_ref(),
      col_vec.device_ref(), C.device_ref(), M, N);
  D_ref.sync_host();
  bool passed =
      verify_footprint<ElementC, LayoutC>(D.host_view(), D_ref.host_view());

  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream accumulator_output(
        "accumulator_tensor_op_128x256x64_64x64x64.csv");
    accumulator_output << accumulator.host_view();
    std::ofstream C_output("C_tensor_op_128x256x64_64x64x64.csv");
    C_output << C.host_view();
    std::ofstream row_vec_output("row_vec_tensor_op_128x256x64_64x64x64.csv");
    row_vec_output << row_vec.host_view();
    std::ofstream col_vec_output("col_vec_tensor_op_128x256x64_64x64x64.csv");
    col_vec_output << col_vec.host_view();
    std::ofstream D_output("D_tensor_op_128x256x64_64x64x64.csv");
    D_output << D.host_view();
    std::ofstream D_ref_output("D_ref_tensor_op_128x256x64_64x64x64.csv");
    D_ref_output << D_ref.host_view();
  }
}