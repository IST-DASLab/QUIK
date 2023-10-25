#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include "int4.h"
#include "matmul/matmul_internal.h"
#include "util.h"

#ifdef QUIK_WITH_CUSPARSELT
#include <cusparseLt.h>
#endif

namespace QUIK::matmul {

torch::Tensor int4MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("int4Matmul", {{A, "A", 0}, {B, "B", 1}});
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

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

torch::Tensor int4OutputInt8MatmulCUDA(const torch::Tensor &A,
                                       const torch::Tensor &B) {
  torch::checkAllSameGPU("int4OutputInt8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt8).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t,                // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      cutlass::int4b_t,                // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int8_t,                          // ElementOutput
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
      {C.data_ptr<int8_t>(), N},
      {C.data_ptr<int8_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
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

torch::Tensor int8OutputInt8MatmulCUDA(const torch::Tensor &A,
                                       const torch::Tensor &B) {
  torch::checkAllSameGPU("int8OutputInt8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt8).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int8_t,                          // ElementOutput
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
      {C.data_ptr<int8_t>(), N},
      {C.data_ptr<int8_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

namespace {
template <typename Gemm>
struct sparseMatmul {
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

  static torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B,
                              const torch::Tensor &E) {
    torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}, {E, "E", 2}});
    using cutlassIndex = cutlass::MatrixCoord::Index;
    using pytorchIndex = torch::IntArrayRef::value_type;

    auto M = static_cast<cutlassIndex>(A.size(0));
    auto N = static_cast<cutlassIndex>(B.size(0));
    auto K = static_cast<cutlassIndex>(
        B.size(1) * cutlass::sizeof_bits<Int4Storage>::value /
        cutlass::sizeof_bits<ElementComputing>::value);

    auto C = torch::empty(
        {M, N},
        torch::dtype(util::TorchDtypeDispatcher<typename Gemm::ElementC>::value)
            .device(A.device()));
    const auto extent =
        cutlass::make_Coord(M, K / kSparse / kElementsPerElementE);
    typename Gemm::Arguments arguments{
        {M, N, K},
        {(ElementComputing *)A.data_ptr<Storage>(),
         K / kSparse},                                        // A, lda (sparse)
        {(ElementComputing *)B.data_ptr<Storage>(), K},       // B, ldb
        {C.template data_ptr<typename Gemm::ElementC>(), N},  // C, ldc
        {C.template data_ptr<typename Gemm::ElementC>(), N},  // D, ldd
        {(ElementInputE *)E.data_ptr<ElementInputESigned>(),
         ReorderedLayoutInputE::packed(extent)},  // E, lde (sparse metadata)
        {1, 0},                                   // alpha, beta
        1                                         // split_k_slices
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

    return C;
  }

  static torch::Tensor reorderMeta(const torch::Tensor &E) {
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
        (ElementInputE *)out.template data_ptr<ElementInputESigned>(),
        ReorderedLayoutInputE::packed(extent));
    cutlass::reorder_meta(tensor_e_reordered, tensor_e, {M, 0, K});

    return out;
  }

  static torch::Tensor genRandomSparseMeta(const int M, const int K) {
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

  static torch::Tensor uncompress(const torch::Tensor &A,
                                  const torch::Tensor &E, const int M,
                                  const int K) {
    auto out =
        torch::empty({M, K * cutlass::sizeof_bits<ElementComputing>::value /
                             cutlass::sizeof_bits<Storage>::value},
                     torch::dtype(util::TorchDtypeDispatcher<Storage>::value));
    cutlass::TensorRef<ElementComputing, cutlass::layout::RowMajor>
        uncompressed_tensor_a{(ElementComputing *)out.data_ptr(), K};
    cutlass::TensorRef<ElementComputing, cutlass::layout::RowMajor> tensor_a{
        (ElementComputing *)A.data_ptr(), K / kSparse};
    cutlass::TensorRef<ElementInputE, cutlass::layout::RowMajor> tensor_e{
        (ElementInputE *)E.data_ptr(), K / kSparse / kElementsPerElementE};
    cutlass::uncompress(uncompressed_tensor_a, tensor_a, tensor_e, M, K);
    return out;
  }
};
using Int4Gemm = cutlass::gemm::device::SparseGemm<
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
    cutlass::gemm::GemmShape<16, 8, 128>      // ShapeMMAOp
    >;

using Int8Gemm = cutlass::gemm::device::SparseGemm<
    int8_t,                                   // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    int8_t,                                   // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    int32_t,                                  // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 128>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 128>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 64>       // ShapeMMAOp
    >;

using Int4GemmOutputInt8 = cutlass::gemm::device::SparseGemm<
    cutlass::int4b_t,                         // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    cutlass::int4b_t,                         // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    int8_t,                                   // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 256>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 256>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 128>      // ShapeMMAOp
    >;

using Int8GemmOutputInt8 = cutlass::gemm::device::SparseGemm<
    int8_t,                                   // ElementInputA,
    cutlass::layout::RowMajor,                // LayoutInputA,
    int8_t,                                   // ElementInputB,
    cutlass::layout::ColumnMajor,             // LayoutInputB,
    int8_t,                                   // ElementOutput,
    cutlass::layout::RowMajor,                // LayoutOutput,
    int32_t,                                  // ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // MMAOp,
    cutlass::arch::Sm80,                      // SmArch,
    cutlass::gemm::GemmShape<128, 128, 128>,  // ShapeMMAThreadBlock,
    cutlass::gemm::GemmShape<64, 64, 128>,    // ShapeMMAWarp,
    cutlass::gemm::GemmShape<16, 8, 64>       // ShapeMMAOp
    >;

template struct sparseMatmul<Int4Gemm>;
template struct sparseMatmul<Int8Gemm>;
template struct sparseMatmul<Int4GemmOutputInt8>;
template struct sparseMatmul<Int8GemmOutputInt8>;
}  // namespace

torch::Tensor int4SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E) {
  return sparseMatmul<Int4Gemm>::matmul(A, B, E);
}

torch::Tensor int4OutputInt8SpMatmulCUDA(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &E) {
  return sparseMatmul<Int4GemmOutputInt8>::matmul(A, B, E);
}

torch::Tensor int4ReorderMeta(const torch::Tensor &E) {
  return sparseMatmul<Int4Gemm>::reorderMeta(E);
}

torch::Tensor int4GenRandomSparseMeta(const int M, const int K) {
  return sparseMatmul<Int4Gemm>::genRandomSparseMeta(M, K);
}

torch::Tensor int4Uncompress(const torch::Tensor &A, const torch::Tensor &E,
                             int M, int K) {
  return sparseMatmul<Int4Gemm>::uncompress(A, E, M, K);
}

torch::Tensor int8SpMatmulCUDA(const torch::Tensor &A, const torch::Tensor &B,
                               const torch::Tensor &E) {
  return sparseMatmul<Int8Gemm>::matmul(A, B, E);
}

torch::Tensor int8OutputInt8SpMatmulCUDA(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &E) {
  return sparseMatmul<Int8GemmOutputInt8>::matmul(A, B, E);
}

torch::Tensor int8ReorderMeta(const torch::Tensor &E) {
  return sparseMatmul<Int8Gemm>::reorderMeta(E);
}

torch::Tensor int8GenRandomSparseMeta(const int M, const int K) {
  return sparseMatmul<Int8Gemm>::genRandomSparseMeta(M, K);
}

torch::Tensor int8Uncompress(const torch::Tensor &A, const torch::Tensor &E,
                             int M, int K) {
  return sparseMatmul<Int8Gemm>::uncompress(A, E, M, K);
}

#ifdef QUIK_WITH_CUSPARSELT
#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
    }                                                                          \
  }

CusparseLtInt8SpMatmul::CusparseLtInt8SpMatmul(const torch::Tensor &A,
                                               const torch::Tensor &B,
                                               const int alg = 0)
    : A_(A),
      B_(B),
      handle_(new cusparseLtHandle_t),
      matA_(new cusparseLtMatDescriptor_t),
      matB_(new cusparseLtMatDescriptor_t),
      matC_(new cusparseLtMatDescriptor_t),
      matmul_(new cusparseLtMatmulDescriptor_t),
      alg_sel_(new cusparseLtMatmulAlgSelection_t),
      plan_(new cusparseLtMatmulPlan_t) {
  torch::checkAllContiguous("CusparseLtInt8SpMatmul",
                            {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("CusparseLtInt8SpMatmul", {A, B},
                         at::DeviceType::CUDA);
  torch::checkAllSameGPU("CusparseLtInt8SpMatmul", {{A, "A", 0}, {B, "B", 1}});
  M_ = A_.size(0);
  K_ = A_.size(1);
  N_ = B_.size(0);
  dA_ = A_.data_ptr<int8_t>();
  dB_ = B_.data_ptr<int8_t>();

  alpha_ = 1.0f;
  beta_ = 0.0f;

  cudaDataType_t input_type, output_type;
  cusparseComputeType compute_type;

  cusparseOrder_t order;
  bool is_rowmajor;
  bool isA_transposed, isB_transposed;
  int64_t num_A_rows, num_A_cols;
  int64_t num_B_rows, num_B_cols;
  int64_t num_C_rows, num_C_cols;
  unsigned alignment;
  int64_t lda, ldb, ldc;

  order = CUSPARSE_ORDER_ROW;
  opA_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  opB_ = CUSPARSE_OPERATION_TRANSPOSE;
  input_type = CUDA_R_8I;
  output_type = CUDA_R_16F;
  compute_type = CUSPARSE_COMPUTE_32I;

  is_rowmajor = (order == CUSPARSE_ORDER_ROW);
  isA_transposed = (opA_ != CUSPARSE_OPERATION_NON_TRANSPOSE);
  isB_transposed = (opB_ != CUSPARSE_OPERATION_NON_TRANSPOSE);
  num_A_rows = (isA_transposed) ? K_ : M_;
  num_A_cols = (isA_transposed) ? M_ : K_;
  num_B_rows = (isB_transposed) ? N_ : K_;
  num_B_cols = (isB_transposed) ? K_ : N_;
  num_C_rows = M_;
  num_C_cols = N_;
  alignment = 16;
  lda = (is_rowmajor) ? num_A_cols : num_A_rows;
  ldb = (is_rowmajor) ? num_B_cols : num_B_rows;
  ldc = (is_rowmajor) ? num_C_cols : num_C_rows;

  num_streams_ = 0;

  CHECK_CUSPARSE(cusparseLtInit(handle_))
  // matrix descriptor initialization
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      handle_, matA_, num_A_rows, num_A_cols, lda, alignment, input_type, order,
      CUSPARSELT_SPARSITY_50_PERCENT))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(handle_, matB_, num_B_rows,
                                               num_B_cols, ldb, alignment,
                                               input_type, order))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(handle_, matC_, num_C_rows,
                                               num_C_cols, ldc, alignment,
                                               output_type, order))
  // matmul, algorithm selection, and plan initialization

  CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
      handle_, matmul_, opA_, opB_, matA_, matB_, matC_, matC_, compute_type))

  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
      handle_, alg_sel_, matmul_, CUSPARSELT_MATMUL_ALG_DEFAULT))

  CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
      handle_, alg_sel_, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(handle_, plan_, matmul_, alg_sel_))
}

CusparseLtInt8SpMatmul::~CusparseLtInt8SpMatmul() {
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(matA_))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(matB_))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(matC_))
  CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(plan_))
  CHECK_CUSPARSE(cusparseLtDestroy(handle_))
}

void CusparseLtInt8SpMatmul::compress() {
  size_t compressed_size, compressed_buffer_size;
  CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(
      handle_, matA_, &compressed_size, &compressed_buffer_size))

  A_compressed_ = torch::empty(static_cast<pytorchIndex>(compressed_size),
                               torch::dtype(torch::kInt8).device(torch::kCUDA));
  dA_compressed_ = A_compressed_.data_ptr<int8_t>();

  auto A_compressedBuffer =
      torch::empty(static_cast<pytorchIndex>(compressed_buffer_size),
                   torch::dtype(torch::kUInt8).device(torch::kCUDA));
  void *dA_compressedBuffer = A_compressedBuffer.data_ptr<uint8_t>();

  CHECK_CUSPARSE(cusparseLtSpMMACompress2(handle_, matA_, true, opA_, dA_,
                                          dA_compressed_, dA_compressedBuffer,
                                          stream_))
}

torch::Tensor CusparseLtInt8SpMatmul::matmulDefault() {
  auto C = torch::empty({M_, N_},
                        torch::dtype(torch::kFloat16).device(torch::kCUDA));
  half *dC = (half *)C.data_ptr<torch::Half>();
  half *dD = dC;

  size_t workspace_size;

  CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(handle_, plan_, &workspace_size))

  auto workspace =
      torch::empty(static_cast<pytorchIndex>(workspace_size),
                   torch::dtype(torch::kUInt8).device(torch::kCUDA));
  void *d_workspace = workspace.data_ptr<uint8_t>();

  cusparseLtMatmul(handle_, plan_, &alpha_, dA_compressed_, dB_, &beta_, dC, dD,
                   d_workspace, streams_, num_streams_);
  return C;
}

torch::Tensor CusparseLtInt8SpMatmul::matmulBy(const torch::Tensor &B) {
  TORCH_CHECK(B.device() == A_compressed_.device())
  TORCH_CHECK(B.size(1) == K_)
  int8_t *dB = B.data_ptr<int8_t>();

  auto C = torch::empty({M_, N_},
                        torch::dtype(torch::kFloat16).device(torch::kCUDA));
  half *dC = (half *)C.data_ptr<torch::Half>();
  half *dD = dC;

  size_t workspace_size;

  CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(handle_, plan_, &workspace_size))

  auto workspace =
      torch::empty(static_cast<pytorchIndex>(workspace_size),
                   torch::dtype(torch::kUInt8).device(torch::kCUDA));
  void *d_workspace = workspace.data_ptr<uint8_t>();

  cusparseLtMatmul(handle_, plan_, &alpha_, dA_compressed_, dB, &beta_, dC, dD,
                   d_workspace, streams_, num_streams_);
  return C;
}

#endif
}  // namespace QUIK::matmul
