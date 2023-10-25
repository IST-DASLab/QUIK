#include "asymmetric/asymmetric_internal.h"
#include "asymmetric/gemm/device/gemm_dequant.h"
#include "asymmetric/gemm/device/gemm_sparse_dequant.h"
#include "int4.h"
#include "util.h"

namespace QUIK::asymmetric {

torch::Tensor int4FusedDequantizeCUDA(
    const torch::Tensor &A, const torch::Tensor &B,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y) {
  torch::checkAllSameGPU("int4FusedDequantize", {
                                                    {A, "A", 0},
                                                    {B, "B", 1},
                                                    {scale_row, "scale_row", 2},
                                                    {scale_col, "scale_col", 3},
                                                    {zero_row, "zero_row", 4},
                                                    {w_reduced, "w_reduced", 5},
                                                    {y, "y", 5},
                                                });
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1) * kElementsPerVector;
  auto D = torch::empty({M, N}, torch::dtype(torch::kF16).device(A.device()));

  using Gemm = cutlass::gemm::device::asymmetric::GemmDequant<
      cutlass::int4b_t,                // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      cutlass::int4b_t,                // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      cutlass::half_t,                 // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {(cutlass::int4b_t *)A.data_ptr<uint8_t>(), K},
      {(cutlass::int4b_t *)B.data_ptr<uint8_t>(), K},
      {(cutlass::half_t *)y.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)D.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_col.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_row.data_ptr<torch::Half>(), M},
      {(cutlass::half_t *)zero_row.data_ptr<torch::Half>(), M},
      {(cutlass::half_t *)w_reduced.data_ptr<torch::Half>(), N},
      Gemm::ElementC(shift_value)};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return D;
}

torch::Tensor int8FusedDequantizeCUDA(
    const torch::Tensor &A, const torch::Tensor &B,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y) {
  torch::checkAllSameGPU("int8FusedDequantize", {
                                                    {A, "A", 0},
                                                    {B, "B", 1},
                                                    {scale_row, "scale_row", 2},
                                                    {scale_col, "scale_col", 3},
                                                    {zero_row, "zero_row", 4},
                                                    {w_reduced, "w_reduced", 5},
                                                    {y, "y", 5},
                                                });
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);
  auto D = torch::empty({M, N}, torch::dtype(torch::kF16).device(A.device()));

  using Gemm = cutlass::gemm::device::asymmetric::GemmDequant<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      cutlass::half_t,                 // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {(cutlass::half_t *)y.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)D.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_col.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_row.data_ptr<torch::Half>(), M},
      {(cutlass::half_t *)zero_row.data_ptr<torch::Half>(), M},
      {(cutlass::half_t *)w_reduced.data_ptr<torch::Half>(), N},
      Gemm::ElementC(shift_value)};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return D;
}

namespace {
template <typename Gemm>
struct sparseFusedDequantize {
  using ElementComputing = typename Gemm::ElementA;
  using Storage =
      std::conditional_t <
      cutlass::sizeof_bits<ElementComputing>::value<8, uint8_t,
                                                    ElementComputing>;
  static_assert(
      std::is_same<typename Gemm::ElementA, typename Gemm::ElementB>::value);

  static constexpr int kSparse = Gemm::kSparse;
  static constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  using ElementInputE = typename Gemm::ElementE;
  using ElementInputESigned = typename std::make_signed<ElementInputE>::type;
  static_assert(std::is_same<ElementInputE, uint32_t>::value);
  using ReorderedLayoutInputE = typename Gemm::LayoutE;

  static torch::Tensor fusedDequantize(
      const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &E,
      const torch::Tensor &scale_row, const torch::Tensor &scale_col,
      const float shift_value, const torch::Tensor &zero_row,
      const torch::Tensor &w_reduced, const torch::Tensor &y) {
    torch::checkAllSameGPU("fusedDequantize", {
                                                  {A, "A", 0},
                                                  {B, "B", 1},
                                                  {scale_row, "scale_row", 2},
                                                  {scale_col, "scale_col", 3},
                                                  {zero_row, "zero_row", 4},
                                                  {w_reduced, "w_reduced", 5},
                                                  {y, "y", 5},
                                              });
    using cutlassIndex = cutlass::MatrixCoord::Index;
    using pytorchIndex = torch::IntArrayRef::value_type;

    auto M = static_cast<cutlassIndex>(A.size(0));
    auto N = static_cast<cutlassIndex>(B.size(0));
    auto K = static_cast<cutlassIndex>(
        B.size(1) * cutlass::sizeof_bits<Int4Storage>::value /
        cutlass::sizeof_bits<ElementComputing>::value);

    auto D = torch::empty(
        {M, N},
        torch::dtype(util::TorchDtypeDispatcher<typename Gemm::ElementC>::value)
            .device(A.device()));
    const auto extent =
        cutlass::make_Coord(M, K / kSparse / kElementsPerElementE);
    typename Gemm::Arguments arguments{
        {M, N, K},
        {(ElementComputing *)A.data_ptr<Storage>(), K / kSparse},
        {(ElementComputing *)B.data_ptr<Storage>(), K},
        {(cutlass::half_t *)y.template data_ptr<torch::Half>(), N},
        {(cutlass::half_t *)D.template data_ptr<torch::Half>(), N},
        {(ElementInputE *)E.data_ptr<ElementInputESigned>(),
         ReorderedLayoutInputE::packed(extent)},
        {(cutlass::half_t *)scale_col.data_ptr<torch::Half>(), N},
        {(cutlass::half_t *)scale_row.data_ptr<torch::Half>(), M},
        {(cutlass::half_t *)zero_row.data_ptr<torch::Half>(), M},
        {(cutlass::half_t *)w_reduced.data_ptr<torch::Half>(), N},
        typename Gemm::ElementC(shift_value),
    };

    auto workspaceSize = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    auto workspace =
        torch::empty(static_cast<pytorchIndex>(workspaceSize),
                     torch::dtype(torch::kUInt8).device(A.device()));

    Gemm gemmOp;
    cutlass::Status status;
    status = Gemm::can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                cutlassGetStatusString(status))

    status = gemmOp.initialize(arguments, workspace.data_ptr<uint8_t>());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                cutlassGetStatusString(status))

    status = gemmOp();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                cutlassGetStatusString(status))

    return D;
  }
};

using Int4GemmDequant = cutlass::gemm::device::asymmetric::SparseGemmDequant<
    cutlass::int4b_t,                         // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    cutlass::int4b_t,                         // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    cutlass::half_t,                          // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 256>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 256>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 128>      // ShapeMMAOp
    >;

using Int8GemmDequant = cutlass::gemm::device::asymmetric::SparseGemmDequant<
    int8_t,                                   // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    int8_t,                                   // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    cutlass::half_t,                          // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 128>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 128>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 64>       // ShapeMMAOp
    >;

template struct sparseFusedDequantize<Int4GemmDequant>;
template struct sparseFusedDequantize<Int8GemmDequant>;
}  // namespace

torch::Tensor int4SpFusedDequantizeCUDA(
    const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &E,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y) {
  return sparseFusedDequantize<Int4GemmDequant>::fusedDequantize(
      A, B, E, scale_row, scale_col, shift_value, zero_row, w_reduced, y);
}

torch::Tensor int8SpFusedDequantizeCUDA(
    const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &E,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y) {
  return sparseFusedDequantize<Int8GemmDequant>::fusedDequantize(
      A, B, E, scale_row, scale_col, shift_value, zero_row, w_reduced, y);
}

}  // namespace QUIK::asymmetric