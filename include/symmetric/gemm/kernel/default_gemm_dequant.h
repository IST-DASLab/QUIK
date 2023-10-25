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

#include "cutlass/gemm/kernel/default_gemm.h"
#include "symmetric/epilogue/threadblock/default_epilogue_tensor_op_dequant.h"
#include "symmetric/gemm/kernel/gemm_dequant.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
namespace symmetric {

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute,
    ///
    typename Enable = void>
struct DefaultGemmDequant
    : public DefaultGemm<ElementA_, LayoutA_, kAlignmentA, ElementB_, LayoutB_,
                         kAlignmentB, ElementC_, LayoutC_, ElementAccumulator,
                         arch::OpClassTensorOp, arch::Sm80, ThreadblockShape,
                         WarpShape, InstructionShape, EpilogueOutputOp,
                         ThreadblockSwizzle, Stages, SplitKSerial, Operator,
                         SharedMemoryClear, GatherA, GatherB, ScatterD,
                         PermuteDLayout, PermuteALayout, PermuteBLayout> {
  static_assert((platform::is_same<LayoutC_, layout::RowMajor>::value ||
                 platform::is_same<LayoutC_, layout::AffineRankN<2>>::value),
                "Epilogue in the kernel level must be row major");

  using DefaultGemm =
      DefaultGemm<ElementA_, LayoutA_, kAlignmentA, ElementB_, LayoutB_,
                  kAlignmentB, ElementC_, LayoutC_, ElementAccumulator,
                  arch::OpClassTensorOp, arch::Sm80, ThreadblockShape,
                  WarpShape, InstructionShape, EpilogueOutputOp,
                  ThreadblockSwizzle, Stages, SplitKSerial, Operator,
                  SharedMemoryClear, GatherA, GatherB, ScatterD, PermuteDLayout,
                  PermuteALayout, PermuteBLayout>;

  using Epilogue = typename cutlass::epilogue::threadblock::symmetric::
      DefaultEpilogueTensorOpDequant<
          ThreadblockShape, typename DefaultGemm::Mma::Operator,
          DefaultGemm::kPartitionsK, EpilogueOutputOp, EpilogueOutputOp::kCount,
          ScatterD, PermuteDLayout>::Epilogue;

  using GemmKernel =
      kernel::symmetric::GemmDequant<typename DefaultGemm::Mma, Epilogue,
                                     ThreadblockSwizzle, SplitKSerial>;
};
////////////////////////////////////////////////////////////////////////////////

}  // namespace symmetric
}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
