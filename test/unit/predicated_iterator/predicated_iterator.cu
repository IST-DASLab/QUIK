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
#include "cutlass/aligned_buffer.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_row_broadcast.h"
#include "cutlass/half.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "symmetric/epilogue/thread/linear_combination_dequant.h"
#include "symmetric/epilogue/threadblock/predicated_vcol_iterator.h"
#include "symmetric/epilogue/threadblock/predicated_vrow_iterator.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename SourceIterator, typename RowVecIterator,
          typename ColVecIterator>
__global__ void kernel_load_iterator(
    typename SourceIterator::Params source_params,
    typename SourceIterator::TensorRef source_ref,
    cutlass::MatrixCoord source_extent,
    typename RowVecIterator::Params row_vec_params,
    typename RowVecIterator::TensorRef row_vec_ref,
    cutlass::MatrixCoord row_vec_extent,
    typename ColVecIterator::Params col_vec_params,
    typename ColVecIterator::TensorRef col_vec_ref,
    cutlass::MatrixCoord col_vec_extent) {
  typename SourceIterator::Fragment source_fragment;
  SourceIterator source_iterator(source_params, source_ref.data(),
                                 source_extent, threadIdx.x);
  typename RowVecIterator::Fragment row_vec_fragment;
  RowVecIterator row_vec_iterator(row_vec_params, row_vec_ref.data(),
                                  row_vec_extent, threadIdx.x);
  typename ColVecIterator::Fragment col_vec_fragment;
  ColVecIterator col_vec_iterator(col_vec_params, col_vec_ref.data(),
                                  col_vec_extent, threadIdx.x);

  source_fragment.clear();
  row_vec_fragment.clear();
  col_vec_fragment.clear();

  CUTLASS_PRAGMA_UNROLL
  for (int iter = 0; iter < SourceIterator::kIterations; ++iter) {
    row_vec_iterator.load(row_vec_fragment);
    col_vec_iterator.load(col_vec_fragment);
    source_iterator.load(source_fragment);
    __syncthreads();

    int const kOutputIterations = SourceIterator::Fragment::kElements /
                                  SourceIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int op_iter = 0; op_iter < kOutputIterations; ++op_iter) {
      int offset = op_iter * SourceIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < SourceIterator::kElementsPerAccess; ++i) {
        source_fragment[offset + i] =
            source_fragment[offset + i] +
            col_vec_fragment[offset + i] * row_vec_fragment[offset + i];
      }
    }

    __syncthreads();
    source_iterator.store(source_fragment);
    ++source_iterator;
    ++row_vec_iterator;
    ++col_vec_iterator;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename Layout>
static bool verify_footprint(cutlass::TensorView<Element, Layout> source_view,
                             cutlass::TensorView<Element, Layout> ref_view) {
  for (int r = 0; r < source_view.extent().row(); ++r) {
    for (int c = 0; c < source_view.extent().column(); ++c) {
      cutlass::MatrixCoord source_coord{r, c};
      if (source_view.at(source_coord) != ref_view.at(source_coord)) {
        return false;
      }
    }
  }

  return true;
}

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/threadblock/default_mma.h"

TEST(PredicatedIterator, tensor_op_128x256x64_64x64x64) {
  using Layout = cutlass::layout::RowMajor;
  using Element = cutlass::half_t;
  // using Element = float;

  static int const kElementsPerAccess =
      128 / cutlass::sizeof_bits<Element>::value;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int32_t;
  using ElementAccumulator = int32_t;
  using DefaultGemmConfig = cutlass::gemm::device::DefaultGemmConfiguration<
      OperatorClass, ArchTag, ElementA, ElementB, ElementC, ElementAccumulator>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = DefaultGemmConfig::ThreadblockShape;
  using WarpShape = DefaultGemmConfig::WarpShape;
  using InstructionShape = DefaultGemmConfig::InstructionShape;
  static const int kStages = DefaultGemmConfig::kStages;
  static const int kAlignmentA = DefaultGemmConfig::kAlignmentA;
  static const int kAlignmentB = DefaultGemmConfig::kAlignmentB;
  using Operator = DefaultGemmConfig::Operator;

  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, OperatorClass, ArchTag, ThreadblockShape,
      WarpShape, InstructionShape, kStages, Operator, false,
      cutlass::gemm::SharedMemoryClearOption::kNone, false, false,
      cutlass::layout::NoPermute, cutlass::layout::NoPermute>::ThreadblockMma;

  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;
  static const int block_size_x = 1;
  static const int block_size_y = 1;
  static const int rows = ThreadblockShape::kM * block_size_y;
  static const int cols = ThreadblockShape::kN * block_size_x;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using SourceThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
          ThreadblockShape, typename Mma::Operator::Shape, kPartitionsK,
          Element, kElementsPerAccess>::Type;
  using SourceIterator =
      cutlass::epilogue::threadblock::PredicatedTileIterator<SourceThreadMap,
                                                             Element>;

  using RowVecIterator =
      cutlass::epilogue::threadblock::symmetric::PredicatedVRowIterator<
          SourceThreadMap, Element>;

  using ColVecIterator =
      cutlass::epilogue::threadblock::symmetric::PredicatedVColIterator<
          SourceThreadMap, Element>;

  cutlass::MatrixCoord source_extent{rows, cols};
  cutlass::MatrixCoord row_vec_extent{1, cols};
  cutlass::MatrixCoord col_vec_extent{1, rows};

  cutlass::HostTensor<Element, Layout> source(source_extent);
  cutlass::HostTensor<Element, Layout> row_vec(row_vec_extent);
  cutlass::HostTensor<Element, Layout> col_vec(col_vec_extent);
  cutlass::HostTensor<Element, Layout> ref(source_extent);

  typename SourceIterator::Params source_iterator_params(source.layout());
  typename RowVecIterator::Params row_vec_iterator_params(source.layout());
  typename ColVecIterator::Params col_vec_iterator_params(source.layout());

  //
  // Fill data
  //

  // cutlass::reference::host::TensorFill(source.host_view(), Element(0));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      source.host_view().at({r, c}) = Element((c + 1) * (r + 1));
    }
  }

  for (int c = 0; c < cols; ++c) {
    row_vec.host_view().at({0, c}) = Element(c + 1);
  }

  for (int r = 0; r < rows; ++r) {
    col_vec.host_view().at({0, r}) = Element(r + 1);
  }

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      ref.host_view().at({r, c}) =
          source.host_view().at({r, c}) +
          col_vec.host_view().at({0, r}) * row_vec.host_view().at({0, c});
    }
  }

  source.sync_device();
  row_vec.sync_device();
  col_vec.sync_device();
  ref.sync_device();

  //
  // Launch kernel
  //

  dim3 grid(block_size_y, block_size_x);
  dim3 block(kThreadCount, 1);

  test::epilogue::threadblock::kernel_load_iterator<
      SourceIterator, RowVecIterator, ColVecIterator><<<grid, block>>>(
      source_iterator_params, source.device_ref(), source_extent,
      row_vec_iterator_params, row_vec.device_ref(), source_extent,
      col_vec_iterator_params, col_vec.device_ref(), source_extent);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

  //
  // Verify results
  //

  source.sync_host();

  bool passed = verify_footprint(source.host_view(), ref.host_view());
  EXPECT_TRUE(passed);

  if (!passed) {
    std::ofstream source_output("source_tensor_op_128x256x64_64x64x64.csv");
    source_output << source.host_view();
    std::ofstream row_vec_output("row_vec_tensor_op_128x256x64_64x64x64.csv");
    row_vec_output << row_vec.host_view();
    std::ofstream col_vec_output("col_vec_tensor_op_128x256x64_64x64x64.csv");
    col_vec_output << col_vec.host_view();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
