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
  \brief Functor performing linear combination operations used by dequantize
  epilogues.
*/
#pragma once

#include <torch/extension.h>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {
namespace asymmetric {

struct MyScaleType {
  enum Kind {
    Dequantize,
  };
};
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_ = cutlass::half_t,
          MyScaleType::Kind Scale = MyScaleType::Dequantize,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          typename ElementSource_ = cutlass::half_t>
class LinearCombinationDequant {
 public:
  using ElementOutput = ElementOutput_;
  using ElementSource = ElementSource_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementC = ElementSource_;
  using ElementD = ElementOutput_;

  static int const kCount = Count;
  static const MyScaleType::Kind kScale = MyScaleType::Dequantize;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentSource = Array<ElementSource, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  struct Params {
    ElementCompute shift_value;

    CUTLASS_HOST_DEVICE
    Params() : shift_value(ElementCompute(0)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute shift_value) : shift_value(shift_value) {}
  };

 private:
  //
  // Data members
  //

  ElementCompute shift_value_ = ElementCompute(0);

 public:
  /// Constructs the function object
  CUTLASS_HOST_DEVICE
  LinearCombinationDequant(Params const &params) {
    shift_value_ = params.shift_value;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentSource const &row_vec_alpha,
                            FragmentSource const &col_vec_alpha,
                            FragmentSource const &zero_row_scaling_frag,
                            FragmentSource const &w_reduced_frag,
                            FragmentSource const &y_frag) const {
    NumericArrayConverter<ElementCompute, ElementSource, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    FragmentCompute converted_row_vec_alpha = source_converter(row_vec_alpha);
    FragmentCompute converted_col_vec_alpha = source_converter(col_vec_alpha);
    FragmentCompute converted_accumulator = accumulator_converter(accumulator);
    FragmentCompute converted_zero_row_scaling =
        source_converter(zero_row_scaling_frag);
    FragmentCompute converted_w_reduced = source_converter(w_reduced_frag);
    FragmentCompute converted_y = source_converter(y_frag);

    FragmentCompute result;
    torch::Half *result_ptr = reinterpret_cast<torch::Half *>(&result);

    const torch::Half *acc_ptr =
        reinterpret_cast<const torch::Half *>(&converted_accumulator);
    const torch::Half *row_vec_ptr =
        reinterpret_cast<const torch::Half *>(&converted_row_vec_alpha);
    const torch::Half *col_vec_ptr =
        reinterpret_cast<const torch::Half *>(&converted_col_vec_alpha);
    const torch::Half *zero_row_scaling_ptr =
        reinterpret_cast<const torch::Half *>(&converted_zero_row_scaling);
    const torch::Half *w_reduced_ptr =
        reinterpret_cast<const torch::Half *>(&converted_w_reduced);
    const torch::Half *y_ptr =
        reinterpret_cast<const torch::Half *>(&converted_y);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      result_ptr[i] = acc_ptr[i] * row_vec_ptr[i] * col_vec_ptr[i] +
                      (zero_row_scaling_ptr[i] +
                       (torch::Half)shift_value_ * col_vec_ptr[i]) *
                          w_reduced_ptr[i] +
                      y_ptr[i];
    }
    return destination_converter(result);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace asymmetric
}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
