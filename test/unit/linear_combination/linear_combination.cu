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
/*! \file
    \brief Unit tests for thread-level GEMM
*/

#include "../common/cutlass_unit_test.h"
#include "asymmetric/epilogue/thread/linear_combination_dequant.h"
#include "cutlass/half.h"
#include "cutlass/util/reference/host/tensor_fill.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_linear_combination, int32_f16_value) {
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = int;
  using ElementCompute = cutlass::half_t;
  using ElementSource = cutlass::half_t;
  using Layout = cutlass::layout::RowMajor;
  static int const kCount = 4;

  using LinearCombination =
      cutlass::epilogue::thread::asymmetric::LinearCombinationDequant<
          ElementOutput, kCount, ElementAccumulator, ElementCompute,
          cutlass::epilogue::thread::symmetric::MyScaleType::Dequantize,
          cutlass::FloatRoundStyle::round_to_nearest, ElementSource>;

  ElementCompute beta = ElementCompute(1);

  typename LinearCombination::Params params(beta);

  LinearCombination linear_combination_op(params);

  cutlass::Array<ElementSource, kCount> source;
  cutlass::Array<ElementSource, kCount> row_vec;
  cutlass::Array<ElementSource, kCount> col_vec;
  cutlass::Array<ElementAccumulator, kCount> accum;
  cutlass::Array<ElementSource, kCount> zero_vec;
  cutlass::Array<ElementSource, kCount> w_reduce;

  for (int i = 0; i < kCount; ++i) {
    source[i] = ElementSource((i * 7 % 9) - 4);
    accum[i] = ElementAccumulator(i * 2);
    row_vec[i] = ElementSource(i);
    col_vec[i] = ElementSource(kCount + i);
    zero_vec[i] = ElementSource(kCount + i);
  }

  cutlass::Array<ElementOutput, kCount> destination =
      linear_combination_op(accum, source, row_vec, col_vec);

  for (int i = 0; i < kCount; ++i) {
    ElementOutput expected = ElementCompute(row_vec[i] * col_vec[i] * accum[i] +
                                            beta * ElementCompute(source[i]));

    ElementOutput got = destination[i];

    EXPECT_TRUE(expected == got);
  }
}
