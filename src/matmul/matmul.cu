#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include <c10/cuda/CUDAGuard.h>

#include "int4.h"
#include "symmetric/symmetric_internal.h"
#include "util.h"

namespace QUIK::matmul {

torch::Tensor int4MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("int4Matmul", {{A, "A", 0}, {B, "B", 1}});
  const at::cuda::CUDAGuard device_guard(A.device());
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t,                // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      cutlass::int4b_t,                // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int32_t,                         // ElementOutput
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
      {C.data_ptr<int32_t>(), N},
      {C.data_ptr<int32_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess);

  return C;
}

torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  const at::cuda::CUDAGuard device_guard(A.device());
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int32_t,                         // ElementOutput
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
      {C.data_ptr<int32_t>(), N},
      {C.data_ptr<int32_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))
  return C;
}

namespace sparseInt4 {
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    int32_t,                                     // ElementOutput,
    128 / cutlass::sizeof_bits<int32_t>::value,  // ElemsPerLoad,
    int32_t,                                     // ElementAccumulator,
    int32_t                                      // ElementComputeEpilogue
    >;
using Gemm = cutlass::gemm::device::SparseGemm<
    cutlass::int4b_t,                         // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    cutlass::int4b_t,                         // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    int32_t,                                  // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 256>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 256>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 128>,     // ShapeMMAOp,
    EpilogueOp                                // EpilogueOp,
    >;

constexpr int kSparse = Gemm::kSparse;
constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
using ElementInputE = Gemm::ElementE;
using ElementInputESigned = std::make_signed<ElementInputE>::type;
static_assert(std::is_same<ElementInputE, uint32_t>::value);
using ReorderedLayoutInputE = typename Gemm::LayoutE;
}  // namespace sparseInt4

torch::Tensor int4SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E) {
  torch::checkAllSameGPU("int4Matmul", {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
  using namespace sparseInt4;
  using cutlassIndex = cutlass::MatrixCoord::Index;
  using pytorchIndex = torch::IntArrayRef::value_type;

  auto M = static_cast<cutlassIndex>(A.size(0));
  auto N = static_cast<cutlassIndex>(B.size(0));
  auto K = static_cast<cutlassIndex>(B.size(1) * kElementsPerVector);

  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));
  const auto extent =
      cutlass::make_Coord(M, K / kSparse / kElementsPerElementE);
  typename Gemm::Arguments arguments{
      {M, N, K},
      {(cutlass::int4b_t *)A.data_ptr<uint8_t>(),
       K / kSparse},                                   // A, lda (sparse)
      {(cutlass::int4b_t *)B.data_ptr<uint8_t>(), K},  // B, ldb
      {C.data_ptr<int32_t>(), N},                      // C, ldc
      {C.data_ptr<int32_t>(), N},                      // D, ldd
      {(ElementInputE *)E.data_ptr<ElementInputESigned>(),
       ReorderedLayoutInputE::packed(extent)},  // E, lde (sparse metadata)
      {1, 0},                                   // alpha, beta
      1                                         // split_k_slices
  };

  auto workspaceSize = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  auto workspace = torch::empty(static_cast<pytorchIndex>(workspaceSize),
                                torch::dtype(torch::kUInt8).device(A.device()));

  Gemm gemmOp;
  cutlass::Status status;
  status = Gemm::can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess);

  status = gemmOp.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess);

  status = gemmOp();
  TORCH_CHECK(status == cutlass::Status::kSuccess);

  return C;
}

torch::Tensor reorderMeta(const torch::Tensor &E, const int M, const int N,
                          const int K) {
  using namespace sparseInt4;
  const auto extent =
      cutlass::make_Coord(M, K / kSparse / kElementsPerElementE);
  auto capacity = ReorderedLayoutInputE::packed(extent).capacity(extent);
  auto out = torch::empty(capacity, torch::dtype(torch::kInt32));
  cutlass::TensorRef<ElementInputE, cutlass::layout::RowMajor> tensor_e(
      (ElementInputE *)E.data_ptr(), K / kSparse / kElementsPerElementE);
  cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
      (ElementInputE *)out.data_ptr(), ReorderedLayoutInputE::packed(extent));
  cutlass::reorder_meta(tensor_e_reordered, tensor_e,
                        {M, N, K / kSparse / kElementsPerElementE});

  return out;
}

torch::Tensor reorderMeta(const torch::Tensor &E) {
  using namespace sparseInt4;
  torch::checkDeviceType("reorderMeta", {E}, torch::DeviceType::CPU);
  using cutlassIndex = cutlass::MatrixCoord::Index;
  auto M = static_cast<cutlassIndex>(E.size(0));
  auto K = static_cast<cutlassIndex>(E.size(1));
  const auto extent = cutlass::make_Coord(M, K);
  auto capacity = ReorderedLayoutInputE::packed(extent).capacity(extent);
  auto out = torch::empty(
      capacity,
      torch::dtype(util::TorchDtypeDispatcher<ElementInputESigned>::value));
  cutlass::TensorRef<ElementInputE, cutlass::layout::RowMajor> tensor_e(
      (ElementInputE *)E.data_ptr<ElementInputESigned>(), K);
  cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
      (ElementInputE *)out.data_ptr<ElementInputESigned>(),
      ReorderedLayoutInputE::packed(extent));
  cutlass::reorder_meta(tensor_e_reordered, tensor_e, {M, 0, K});

  return out;
}

torch::Tensor genRandomSparseMeta(const int M, const int K) {
  using namespace sparseInt4;
  constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  auto out = torch::empty({M, K / kSparse / kElementsPerElementE},
                          torch::dtype(torch::kInt32));
  const auto extent =
      cutlass::make_Coord(M, K / kSparse / kElementsPerElementE);

  cutlass::TensorView<ElementInputE, cutlass::layout::RowMajor> tensor_e_view{
      (ElementInputE *)out.data_ptr(),
      cutlass::layout::RowMajor::packed(extent), extent};
  cutlass::reference::host::TensorFillRandomSparseMeta(
      tensor_e_view, 1,
      kMetaSizeInBits);  // <- Fill matrix E on host with uniform-distribution
                         // random meta data
  return out;
}

torch::Tensor uncompress(const torch::Tensor &A, const torch::Tensor &E,
                         const int M, const int K) {
  using namespace sparseInt4;
  auto out = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8));
  cutlass::TensorRef<cutlass::int4b_t, cutlass::layout::RowMajor>
      uncompressed_tensor_a{(cutlass::int4b_t *)out.data_ptr(), K};
  cutlass::TensorRef<cutlass::int4b_t, cutlass::layout::RowMajor> tensor_a{
      (cutlass::int4b_t *)A.data_ptr(), K / kSparse};
  cutlass::TensorRef<ElementInputE, cutlass::layout::RowMajor> tensor_e{
      (ElementInputE *)E.data_ptr(), K / kSparse / kElementsPerElementE};
  cutlass::uncompress(uncompressed_tensor_a, tensor_a, tensor_e, M, K);
  return out;
}

}  // namespace QUIK::matmul