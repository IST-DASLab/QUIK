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
#pragma once

#include "asymmetric/epilogue/threadblock/epilogue_dequant.h"
#include "asymmetric/epilogue/threadblock/predicated_vcol_iterator.h"
#include "asymmetric/epilogue/threadblock/predicated_vrow_iterator.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {
namespace asymmetric {
////////////////////////////////////////////////////////////////////////////////
template <typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
          typename OutputOp_, int ElementsPerAccess, bool ScatterD = false,
          typename PermuteDLayout = layout::NoPermute>
struct DefaultEpilogueTensorOpDequant
    : public DefaultEpilogueTensorOp<Shape_, WarpMmaTensorOp_, PartitionsK,
                                     OutputOp_, ElementsPerAccess, ScatterD,
                                     PermuteDLayout> {
  using OutputOp = OutputOp_;
  using DefaultEpilogueTensorOp =
      DefaultEpilogueTensorOp<Shape_, WarpMmaTensorOp_, PartitionsK, OutputOp_,
                              ElementsPerAccess, ScatterD, PermuteDLayout>;
  using RowVecIterator =
      cutlass::epilogue::threadblock::asymmetric::PredicatedVRowIterator<
          typename DefaultEpilogueTensorOp::OutputTileThreadMap,
          typename DefaultEpilogueTensorOp::ElementOutput, ScatterD,
          PermuteDLayout, DefaultEpilogueTensorOp::UseCUDAStore>;
  using ColVecIterator =
      cutlass::epilogue::threadblock::asymmetric::PredicatedVColIterator<
          typename DefaultEpilogueTensorOp::OutputTileThreadMap,
          typename DefaultEpilogueTensorOp::ElementOutput, ScatterD,
          PermuteDLayout, DefaultEpilogueTensorOp::UseCUDAStore>;

  using Epilogue = cutlass::epilogue::threadblock::asymmetric::EpilogueDequant<
      typename DefaultEpilogueTensorOp::Shape,
      typename DefaultEpilogueTensorOp::WarpMmaTensorOp,
      DefaultEpilogueTensorOp::kPartitionsK,
      typename DefaultEpilogueTensorOp::OutputTileIterator, RowVecIterator,
      ColVecIterator,
      typename DefaultEpilogueTensorOp::AccumulatorFragmentIterator,
      typename DefaultEpilogueTensorOp::WarpTileIterator,
      typename DefaultEpilogueTensorOp::SharedLoadIterator, OutputOp,
      typename DefaultEpilogueTensorOp::Padding,
      DefaultEpilogueTensorOp::kFragmentsPerIteration>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace asymmetric
}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
