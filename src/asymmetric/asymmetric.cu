#include <c10/cuda/CUDAGuard.h>

#include "asymmetric/asymmetric_internal.h"
#include "int4.h"
#include "util.h"

namespace QUIK::asymmetric {
const unsigned MAX_NUMTHREADS = 1024;
const unsigned MAX_NUMBER_BLOCKS = 65535;
unsigned NUM_STRIDES_PER_THREAD_QUANTIZE = 0;
unsigned NUM_STRIDES_PER_THREAD_DEQUANTIZE = 0;

__global__ void quantizeCUDAKernel4Bits(Int4Storage *__restrict__ dst,
                                        const torch::Half *__restrict__ src,
                                        const torch::Half *__restrict__ meta,
                                        const unsigned rows,
                                        const unsigned colsSrc,
                                        unsigned colsDst) {
  const unsigned thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = blockDim.x * gridDim.x;
  const unsigned num_elems_src = rows * colsSrc;
  const unsigned num_elems_dst = rows * colsDst;
  Int4Storage storage;

  for (unsigned idx = thread_id; idx < num_elems_dst; idx += stride) {
    const unsigned row = idx / colsDst;
    memset(&storage, 0, sizeof(storage));

#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i) {
      bool safe = (idx * kElementsPerVector + i) < num_elems_src;
      if (safe) {
        __half scale = meta[2 * row];
        __half zero = meta[2 * row + 1];
        __half data =
            __hdiv(__hsub(src[idx * kElementsPerVector + i], zero), scale);
        int val = __half2int_rn(data);
        // needs to be shifted by 8 to fit int4b_t
        val = min(max(val, 0), 15) - 8;
        Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(val);
      }
    }
    dst[idx] = storage;
  }
}

__global__ void quantizeCUDAKernel8Bits(int8_t *__restrict__ dst,
                                        const torch::Half *__restrict__ src,
                                        const torch::Half *__restrict__ meta,
                                        const unsigned rows,
                                        const unsigned cols) {
  const unsigned thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = blockDim.x * gridDim.x;
  const unsigned num_elems = rows * cols;

  for (unsigned idx = thread_id; idx < num_elems; idx += stride) {
    const unsigned row = idx / cols;
    __half scale = meta[2 * row];
    __half zero = meta[2 * row + 1];
    __half data = __hdiv(__hsub(src[idx], zero), scale);
    int val = __half2int_rn(data);
    // needs to be shifted by 128 to fit int8_t
    val = min(max(val, 0), 255) - 128;
    dst[idx] = static_cast<int8_t>(val);
  }
}

torch::Tensor quantizeCUDAOld(const torch::Tensor &src,
                              const torch::Tensor &meta, int bits) {
  torch::checkAllSameGPU("quantize", {{src, "src", 0}, {meta, "meta", 1}});
  torch::checkSize("quantize", torch::TensorArg{meta, "meta", 1}, 0,
                   src.size(0) * 2);
  const at::cuda::CUDAGuard device_guard(src.device());

  if (NUM_STRIDES_PER_THREAD_QUANTIZE == 0) {
    char const *temp = getenv("NUM_STRIDES_PER_THREAD_QUANTIZE");
    if (temp)
      NUM_STRIDES_PER_THREAD_QUANTIZE = std::atoi(temp);
    else
      NUM_STRIDES_PER_THREAD_QUANTIZE = 1;
    TORCH_CHECK(NUM_STRIDES_PER_THREAD_QUANTIZE > 0 and
                    NUM_STRIDES_PER_THREAD_QUANTIZE < 64,
                "Quantize: invalid value of NUM_STRIDES_PER_THREAD_QUANTIZE");
  }

  unsigned rows = src.size(0);
  unsigned colsSrc = src.size(1);
  torch::Tensor dst;
  const unsigned num_elems = src.numel();
  const unsigned num_threads = min(num_elems, MAX_NUMTHREADS);
  const unsigned num_blocks =
      max((num_elems + num_threads - 1) /
              (num_threads * NUM_STRIDES_PER_THREAD_QUANTIZE),
          16);

  if (bits == 4) {
    unsigned colsDst = (colsSrc + kElementsPerVector - 1) / kElementsPerVector;
    dst = torch::empty(
        {rows, colsDst},
        torch::dtype(util::TorchDtypeDispatcher<Int4Storage>::value)
            .device(src.device()));
    quantizeCUDAKernel4Bits<<<num_blocks, num_threads>>>(
        dst.data_ptr<Int4Storage>(), src.data_ptr<torch::Half>(),
        meta.data_ptr<torch::Half>(), rows, colsSrc, colsDst);
  } else {
    dst = torch::empty({rows, colsSrc},
                       torch::dtype(util::TorchDtypeDispatcher<int8_t>::value)
                           .device(src.device()));
    quantizeCUDAKernel8Bits<<<num_blocks, num_threads>>>(
        dst.data_ptr<int8_t>(), src.data_ptr<torch::Half>(),
        meta.data_ptr<torch::Half>(), rows, colsSrc);
  }
  auto status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Failed quantize: " + std::string(cudaGetErrorString(status)));
  return dst;
}

template <typename KTorch>
__global__ void dequantizationKernel(
    KTorch *__restrict__ out, const int *__restrict__ x,
    const KTorch *__restrict__ meta, const KTorch *__restrict__ scaleCol,
    const KTorch *__restrict__ wReduced, const KTorch *__restrict__ y,
    const unsigned rows, const unsigned cols, const float shift_value) {
  const unsigned thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = blockDim.x * gridDim.x;
  const unsigned num_elems = rows * cols;
  using K = typename util::DtypeTorchDispatcher<KTorch>::value;
  using K2 = typename util::DtypeDtype2Dispatcher<K>::value;
  K2 *out2 = reinterpret_cast<K2 *>(out);
  // Casting all the variables to float32 before the dequantization
  // improves accuracy without significant effect on the performance.
  //  for (int idx = thread_id; idx < num_elems; idx += stride) {
  for (int i = thread_id; 2 * i < num_elems; i += stride) {
    unsigned idx = 2 * i;
    unsigned row = idx / cols;
    unsigned col = idx % cols;
    float xElement = static_cast<float>(x[idx]);
    float srow_f = util::type2float(meta[2 * row]);
    float scol_f = util::type2float(scaleCol[col]);
    float z_f = util::type2float(meta[2 * row + 1]);
    float shift =
        (z_f + shift_value * srow_f) * util::type2float(wReduced[col]);
    xElement *= srow_f * scol_f;
    xElement += shift + util::type2float(y[idx]);
    K res1 = util::float2type<K>(xElement);
    idx++;
    if (idx >= num_elems) {
      out2[i] = util::type2type2(res1, util::float2type<K>(0.0));
      break;
    }

    xElement = static_cast<float>(x[idx]);
    row = idx / cols;
    col = idx % cols;
    srow_f = util::type2float(meta[2 * row]);
    scol_f = util::type2float(scaleCol[col]);
    z_f = util::type2float(meta[2 * row + 1]);
    shift = (z_f + shift_value * srow_f) * util::type2float(wReduced[col]);
    xElement *= srow_f * scol_f;
    xElement += shift + util::type2float(y[idx]);
    K res2 = util::float2type<K>(xElement);
    out2[i] = util::type2type2(res1, res2);

    //    __half xElement = __int2half_rn(x[idx]);
    //    xElement = __hadd(__hmul(xElement, scaleCol[col]), meta[2 * row]);
    //    __half shift = __hmul(
    //        __hfma(__float2half(shift_value), meta[2 * row], meta[2 * row +
    //        1]), wReduced[col]);
    //    out[idx] = static_cast<torch::Half>(__hadd(__hadd(xElement, shift),
    //    y[idx]));
  }
}

torch::Tensor dequantizeCUDA(const torch::Tensor &x, const torch::Tensor &meta,
                             const torch::Tensor &scaleCol,
                             const torch::Tensor &wReduced,
                             const torch::Tensor &y, const int bits) {
  const at::cuda::CUDAGuard device_guard(x.device());
  torch::checkAllSameGPU("dequantize", {{x, "x", 0},
                                        {meta, "meta", 1},
                                        {scaleCol, "scaleCol", 2},
                                        {wReduced, "wReduced", 3},
                                        {y, "y", 4}});
  TORCH_CHECK(x.numel() == y.numel(), "Expected tensor for x",
              " to have same number of elements as tensor for y; but ",
              x.numel(), " does not equal ", y.numel(),
              " (while checking arguments for dequantize)");
  unsigned rows = x.size(0);
  unsigned cols = x.size(1);
  torch::checkSize("dequantize", torch::TensorArg{meta, "meta", 1}, 0,
                   2 * rows);
  torch::checkSize("dequantize", torch::TensorArg{scaleCol, "scaleCol", 2}, 0,
                   cols);
  torch::checkSize("dequantize", torch::TensorArg{wReduced, "wReduced", 3}, 1,
                   cols);
  auto out = torch::empty_like(y);
  float shift_value = (bits == 4) ? 8.0f : 128.0f;
  const unsigned num_elems = x.numel();
  const unsigned num_threads = min(num_elems, MAX_NUMTHREADS);
  if (NUM_STRIDES_PER_THREAD_DEQUANTIZE == 0) {
    char const *temp = getenv("NUM_STRIDES_PER_THREAD_DEQUANTIZE");
    if (temp)
      NUM_STRIDES_PER_THREAD_DEQUANTIZE = std::atoi(temp);
    else
      NUM_STRIDES_PER_THREAD_DEQUANTIZE = 1;
    TORCH_CHECK(
        NUM_STRIDES_PER_THREAD_DEQUANTIZE > 0 and
            NUM_STRIDES_PER_THREAD_DEQUANTIZE < 64,
        "Dequantize: invalid value of NUM_STRIDES_PER_THREAD_DEQUANTIZE");
  }
  const unsigned num_blocks =
      max((num_elems + num_threads * NUM_STRIDES_PER_THREAD_DEQUANTIZE - 1) /
              (num_threads * NUM_STRIDES_PER_THREAD_DEQUANTIZE),
          16);
  if (out.dtype() == torch::kHalf) {
    dequantizationKernel<<<num_blocks, num_threads>>>(
        out.data_ptr<torch::Half>(), x.data_ptr<int>(),
        meta.data_ptr<torch::Half>(), scaleCol.data_ptr<torch::Half>(),
        wReduced.data_ptr<torch::Half>(), y.data_ptr<torch::Half>(), rows, cols,
        shift_value);
  } else if (out.dtype() == torch::kBFloat16) {
    dequantizationKernel<<<num_blocks, num_threads>>>(
        out.data_ptr<torch::BFloat16>(), x.data_ptr<int>(),
        meta.data_ptr<torch::BFloat16>(), scaleCol.data_ptr<torch::BFloat16>(),
        wReduced.data_ptr<torch::BFloat16>(), y.data_ptr<torch::BFloat16>(),
        rows, cols, shift_value);
  }
  auto status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Failed dequantize: " + std::string(cudaGetErrorString(status)));
  return out;
}

template <typename K>
__device__ void warpReduce(volatile K *sdata, unsigned tid,
                           unsigned block_size) {
  sdata[tid] = __hmax(sdata[tid + 32], sdata[tid]);
  sdata[block_size + tid] =
      __hmin(sdata[block_size + tid + 32], sdata[block_size + tid]);

  sdata[tid] = __hmax(sdata[tid + 16], sdata[tid]);
  sdata[block_size + tid] =
      __hmin(sdata[block_size + tid + 16], sdata[block_size + tid]);

  sdata[tid] = __hmax(sdata[tid + 8], sdata[tid]);
  sdata[block_size + tid] =
      __hmin(sdata[block_size + tid + 8], sdata[block_size + tid]);

  sdata[tid] = __hmax(sdata[tid + 4], sdata[tid]);
  sdata[block_size + tid] =
      __hmin(sdata[block_size + tid + 4], sdata[block_size + tid]);

  sdata[tid] = __hmax(sdata[tid + 2], sdata[tid]);
  sdata[block_size + tid] =
      __hmin(sdata[block_size + tid + 2], sdata[block_size + tid]);

  sdata[tid] = __hmax(sdata[tid + 1], sdata[tid]);
  sdata[block_size + tid] =
      __hmin(sdata[block_size + tid + 1], sdata[block_size + tid]);
}

template <typename KTorch>
__device__ void FindMetaParallel(const KTorch *input, KTorch *meta,
                                 const unsigned num_elems,
                                 const unsigned divisor) {
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  using K = typename util::DtypeTorchDispatcher<KTorch>::value;

  extern __shared__ __align__(sizeof(K)) unsigned char my_smem[];
  K *sdata = reinterpret_cast<K *>(my_smem);
  sdata[tid] = input[0];
  sdata[block_size + tid] = input[0];
  unsigned int num_iters_per_bucket = (num_elems + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    unsigned int idx = i * block_size + tid;
    if (idx < num_elems) {
      sdata[tid] = __hmax(sdata[tid], input[idx]);
      sdata[block_size + tid] = __hmin(sdata[block_size + tid], input[idx]);
    }
  }
  __syncthreads();
  for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = __hmax(sdata[tid + s], sdata[tid]);
      sdata[block_size + tid] =
          __hmin(sdata[block_size + tid + s], sdata[block_size + tid]);
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduce(sdata, tid, block_size);
  }
  if (tid == 0) {
    float max_f = util::type2float(sdata[0]);
    float min_f = util::type2float(sdata[block_size]);
    meta[0] = util::float2type<K>(fmaxf((max_f - min_f) / divisor, 1e-6));
    meta[1] = sdata[block_size];
  }
}

template <typename K>
__global__ void FindMetaKernel(const K *input, K *meta,
                               const unsigned num_elems,
                               const unsigned bucket_size,
                               const unsigned divisor) {
  unsigned num_blocks = gridDim.x;
  unsigned int bid = blockIdx.x;
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned int cur_bucket_size;
  for (unsigned int bucket_id = bid; bucket_id < num_buckets;
       bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    FindMetaParallel(input + bucket_size * bucket_id, meta + 2 * bucket_id,
                     cur_bucket_size, divisor);
  }
}

torch::Tensor findMaxMinMetaCUDA(const torch::Tensor &src,
                                 const unsigned bits) {
  const at::cuda::CUDAGuard device_guard(src.device());
  unsigned rows = src.size(0);
  unsigned colsSrc = src.size(1);
  auto dst =
      torch::empty({2 * rows}, torch::dtype(torch::kHalf).device(src.device()));
  int num_blocks = min(rows, MAX_NUMBER_BLOCKS);
  int num_threads = max(min(MAX_NUMTHREADS, colsSrc), 64);
  int shared_memory_block_size = 2 * num_threads * sizeof(__half);
  FindMetaKernel<<<num_blocks, num_threads, shared_memory_block_size>>>(
      src.data_ptr<torch::Half>(), dst.data_ptr<torch::Half>(), src.numel(),
      colsSrc, (1 << bits) - 1);
  auto status = cudaGetLastError();
  TORCH_CHECK(
      status == cudaSuccess,
      "Failed findMaxMinMetaCUDA: " + std::string(cudaGetErrorString(status)));
  return dst;
}

template <typename KTorch>
__device__ void FindMetaParallelIndexed(const KTorch *input, KTorch *meta,
                                        const long *int_indices,
                                        const unsigned num_int,
                                        const unsigned divisor) {
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  using K = typename util::DtypeTorchDispatcher<KTorch>::value;

  extern __shared__ __align__(sizeof(K)) unsigned char my_smem[];
  K *sdata = reinterpret_cast<K *>(my_smem);
  sdata[tid] = static_cast<K>(input[int_indices[0]]);
  sdata[block_size + tid] = static_cast<K>(input[int_indices[0]]);
  unsigned int num_iters_per_bucket = (num_int + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    unsigned int idx = i * block_size + tid;
    if (idx < num_int) {
      sdata[tid] = __hmax(sdata[tid], static_cast<K>(input[int_indices[idx]]));
      sdata[block_size + tid] = __hmin(sdata[block_size + tid],
                                       static_cast<K>(input[int_indices[idx]]));
    }
  }
  __syncthreads();
  for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = __hmax(sdata[tid + s], sdata[tid]);
      sdata[block_size + tid] =
          __hmin(sdata[block_size + tid + s], sdata[block_size + tid]);
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduce(sdata, tid, block_size);
  }
  if (tid == 0) {
    float max_f = util::type2float(sdata[0]);
    float min_f = util::type2float(sdata[block_size]);
    meta[0] = util::float2type<K>(fmaxf((max_f - min_f) / divisor, 1e-6));
    meta[1] = sdata[block_size];
  }
}

template <typename KTorch>
__device__ void quantizeCUDABucketKernel8bitsIndexed(
    int8_t *__restrict__ dst, const KTorch *__restrict__ src,
    const KTorch *__restrict__ meta, const long *int_indices,
    const unsigned num_int) {
  const unsigned thread_id = threadIdx.x;
  const unsigned stride = blockDim.x;
  using K = typename util::DtypeTorchDispatcher<KTorch>::value;

  for (unsigned idx = thread_id; idx < num_int; idx += stride) {
    K scale = static_cast<K>(meta[0]);
    K zero = static_cast<K>(meta[1]);
    K src_value = static_cast<K>(src[int_indices[idx]]);
    K data = __hdiv(__hsub(src_value, zero), scale);
    int val = util::type2int_rn(data);
    // needs to be shifted by 128 to fit int8_t
    val = min(max(val, 0), 255) - 128;
    dst[idx] = static_cast<int8_t>(val);
  }
}
template <typename KTorch>
__device__ void quantizeCUDABucketKernel4bitsIndexed(
    Int4Storage *__restrict__ dst, const KTorch *__restrict__ src,
    const KTorch *__restrict__ meta, const long *int_indices,
    const unsigned num_int) {
  using K = typename util::DtypeTorchDispatcher<KTorch>::value;

  int8_t storage;
  const unsigned thread_id = threadIdx.x;
  const unsigned stride = blockDim.x;
  const unsigned num_elems_dst = num_int / 2;
  const unsigned num_elems_src = num_int;
  K scale = static_cast<K>(meta[0]);
  K zero = static_cast<K>(meta[1]);
  K data;
  int val;
  for (unsigned idx = thread_id; idx < num_elems_dst; idx += stride) {
    memset(&storage, 0, sizeof(storage));
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i) {
      //      bool safe = (idx * kElementsPerVector + i) < num_elems_src;
      //      if (safe) {
      K src_value =
          static_cast<K>(src[int_indices[idx * kElementsPerVector + i]]);
      data = __hdiv(__hsub(src_value, zero), scale);
      val = util::type2int_rn(data);
      // needs to be shifted by 8 to fit int4b_t
      val = min(max(val, 0), 15) - 8;
      Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(val);
      //      }
    }
    dst[idx] = storage;
  }
}

template <typename K>
__device__ void memcpyCUDAIndexed(const K *input, K *fp_x,
                                  const long *fp_indices,
                                  const unsigned num_fp) {
  const unsigned thread_id = threadIdx.x;
  const unsigned stride = blockDim.x;
  for (unsigned idx = thread_id; idx < num_fp; idx += stride) {
    fp_x[idx] = input[fp_indices[idx]];
  }
}

template <typename T, typename K, int BITS>
__global__ void quantizeCUDAKernel(T *dst, K *meta, K *fp_x, const K *src,
                                   const long *int_indices,
                                   const long *fp_indices, const unsigned rows,
                                   const unsigned cols, const unsigned num_int,
                                   const unsigned num_fp) {
  unsigned num_blocks = gridDim.x;
  const unsigned bid = blockIdx.x;
  const unsigned bucket_size = cols;
  // we quantize num_int values in row
  // we move num_fp values per row
  const unsigned compressed_bucket_size = (num_int * BITS + 7) / 8;
  //    unsigned cur_bucket_size;

  for (unsigned int bucket_id = bid; bucket_id < rows;
       bucket_id += num_blocks) {
    //      cur_bucket_size = umin(bucket_size, num_elems - bucket_id *
    //      bucket_size);

    // Find zero, scale for the values to quantize in int_indices positions
    FindMetaParallelIndexed(src + bucket_size * bucket_id, meta + 2 * bucket_id,
                            int_indices, num_int, (1 << BITS) - 1);
    __syncthreads();

    // Quantize values in int_indices positions
    if constexpr (BITS == 4) {
      quantizeCUDABucketKernel4bitsIndexed(
          reinterpret_cast<Int4Storage *>(dst +
                                          compressed_bucket_size * bucket_id),
          src + bucket_size * bucket_id, meta + 2 * bucket_id, int_indices,
          num_int);
    } else {
      quantizeCUDABucketKernel8bitsIndexed(
          reinterpret_cast<int8_t *>(dst + compressed_bucket_size * bucket_id),
          src + bucket_size * bucket_id, meta + 2 * bucket_id, int_indices,
          num_int);
    }

    // Move full precision values in fp_indices positions
    memcpyCUDAIndexed(src + bucket_size * bucket_id, fp_x + num_fp * bucket_id,
                      fp_indices, num_fp);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantizeCUDA(
    const torch::Tensor &src, const torch::Tensor &int_indices,
    const torch::Tensor &fp_indices, int bits) {
  const at::cuda::CUDAGuard device_guard(src.device());

  if (NUM_STRIDES_PER_THREAD_QUANTIZE == 0) {
    char const *temp = getenv("NUM_STRIDES_PER_THREAD_QUANTIZE");
    if (temp)
      NUM_STRIDES_PER_THREAD_QUANTIZE = std::atoi(temp);
    else
      NUM_STRIDES_PER_THREAD_QUANTIZE = 1;
    TORCH_CHECK(NUM_STRIDES_PER_THREAD_QUANTIZE > 0 and
                    NUM_STRIDES_PER_THREAD_QUANTIZE < 64,
                "Quantize: invalid value of NUM_STRIDES_PER_THREAD_QUANTIZE");
  }

  unsigned rows = src.size(0);
  unsigned cols = src.size(1);
  torch::Tensor dst;
  const unsigned num_elems = src.numel();
  const unsigned num_threads = min(num_elems, MAX_NUMTHREADS);
  const unsigned num_blocks =
      max((num_elems + num_threads * NUM_STRIDES_PER_THREAD_QUANTIZE - 1) /
              (num_threads * NUM_STRIDES_PER_THREAD_QUANTIZE),
          16);
  unsigned num_fp = fp_indices.numel();
  unsigned num_int = int_indices.numel();
  TORCH_CHECK(num_fp + num_int == cols,
              "Quantize: number of fp and int columns is not equal to total "
              "number of columns");
  TORCH_CHECK(
      num_int % 2 == 0,
      "Quantize: number of int columns has to be even (implementation detail)");
  auto meta =
      torch::empty({2 * rows}, torch::dtype(src.dtype()).device(src.device()));
  auto fp_x = torch::empty({rows, num_fp},
                           torch::dtype(src.dtype()).device(src.device()));
  int shared_memory_block_size = 2 * num_threads * src.element_size();
  //  std::cout << "Handle " << src.dtype << std::endl;
  if (src.dtype() == torch::kHalf) {
    //    printf("Process half\n");
    if (bits == 4) {
      dst = torch::empty(
          {rows, num_int / 2},
          torch::dtype(util::TorchDtypeDispatcher<Int4Storage>::value)
              .device(src.device()));
      quantizeCUDAKernel<Int4Storage, torch::Half, 4>
          <<<num_blocks, num_threads, shared_memory_block_size>>>(
              dst.data_ptr<Int4Storage>(), meta.data_ptr<torch::Half>(),
              fp_x.data_ptr<torch::Half>(), src.data_ptr<torch::Half>(),
              int_indices.data_ptr<long>(), fp_indices.data_ptr<long>(), rows,
              cols, num_int, num_fp);
    } else {
      dst = torch::empty({rows, num_int},
                         torch::dtype(util::TorchDtypeDispatcher<int8_t>::value)
                             .device(src.device()));
      quantizeCUDAKernel<int8_t, torch::Half, 8>
          <<<num_blocks, num_threads, shared_memory_block_size>>>(
              dst.data_ptr<int8_t>(), meta.data_ptr<torch::Half>(),
              fp_x.data_ptr<torch::Half>(), src.data_ptr<torch::Half>(),
              int_indices.data_ptr<long>(), fp_indices.data_ptr<long>(), rows,
              cols, num_int, num_fp);
    }
  } else if (src.dtype() == torch::kBFloat16) {
    //    printf("Process bfloat16\n");
    if (bits == 4) {
      dst = torch::empty(
          {rows, num_int / 2},
          torch::dtype(util::TorchDtypeDispatcher<Int4Storage>::value)
              .device(src.device()));
      quantizeCUDAKernel<Int4Storage, torch::BFloat16, 4>
          <<<num_blocks, num_threads, shared_memory_block_size>>>(
              dst.data_ptr<Int4Storage>(), meta.data_ptr<torch::BFloat16>(),
              fp_x.data_ptr<torch::BFloat16>(), src.data_ptr<torch::BFloat16>(),
              int_indices.data_ptr<long>(), fp_indices.data_ptr<long>(), rows,
              cols, num_int, num_fp);
    } else {
      dst = torch::empty({rows, num_int},
                         torch::dtype(util::TorchDtypeDispatcher<int8_t>::value)
                             .device(src.device()));
      quantizeCUDAKernel<int8_t, torch::BFloat16, 8>
          <<<num_blocks, num_threads, shared_memory_block_size>>>(
              dst.data_ptr<int8_t>(), meta.data_ptr<torch::BFloat16>(),
              fp_x.data_ptr<torch::BFloat16>(), src.data_ptr<torch::BFloat16>(),
              int_indices.data_ptr<long>(), fp_indices.data_ptr<long>(), rows,
              cols, num_int, num_fp);
    }
  }
  auto status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Failed quantize: " + std::string(cudaGetErrorString(status)));
  return {dst, meta, fp_x};
}

}  // namespace QUIK::asymmetric
