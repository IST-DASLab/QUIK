import gc
import numpy as np
import torch
from abc import ABC, abstractmethod

import quik


def pack_to_i4(X):
    def two_compl(x, bits):
        return torch.where(x < 0, 2 ** bits + x, x)

    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4


def pack_to_i8(X_i4):
    M, K_half = X_i4.shape

    flattened = X_i4.view(-1)

    high_nibbles = (flattened >> 4).to(torch.int8)
    low_nibbles = (flattened & 0x0F).to(torch.int8)

    high_nibbles[high_nibbles > 7] -= 16
    low_nibbles[low_nibbles > 7] -= 16

    X_i8 = torch.empty((M, K_half * 2), dtype=torch.int8, device=X_i4.device)
    X_i8[:, 0::2] = low_nibbles.view(M, K_half)
    X_i8[:, 1::2] = high_nibbles.view(M, K_half)

    return X_i8


class Kernel(ABC):
    @classmethod
    def __init__(cls, M, N, K):
        cls._M, cls._N, cls._K = M, N, K
        cls._a, cls._b = None, None
        cls._c, cls._c_ref = None, None
        cls._src, cls._scale = None, None
        cls._scale_row, cls._scale_col, cls._y = None, None, None

    @classmethod
    def quantize_randomizer(cls):
        torch.manual_seed(1)
        np.random.seed(1)
        cls._scale = torch.randn(cls._M, 1).to(torch.float16).cuda()

        cls._scale_row = torch.randn(cls._M, 1).to(torch.float16).cuda()
        cls._scale_col = torch.randn(1, cls._N).to(torch.float16).cuda()
        cls._y = torch.randn(cls._M, cls._N).to(torch.float16).cuda()

    @classmethod
    @abstractmethod
    def matmul_randomizer(cls):
        pass

    @classmethod
    @abstractmethod
    def quantization(cls):
        pass

    @classmethod
    @abstractmethod
    def calculation(cls):
        pass

    @classmethod
    def verification(cls):
        assert torch.equal(cls._c, cls._c_ref)

    @classmethod
    def dequantization(cls):
        cls._c_deq = quik.symmetric.dequantize(cls._c, cls._scale_row, cls._scale_col, cls._y, 32)

    @classmethod
    def cleaning(cls):
        del cls._a, cls._b, cls._c, cls._c_ref
        del cls._src, cls._scale, cls._scale_row, cls._scale_col, cls._y

    @staticmethod
    def empty_cache():
        torch.cuda.empty_cache()


class Int4MatmulInt32Out(Kernel):
    @classmethod
    def quantize_randomizer(cls):
        super().quantize_randomizer()
        cls._src = torch.randn(cls._M, cls._K).to(torch.float16).cuda()

    @classmethod
    def matmul_randomizer(cls):
        B = 1
        a = torch.empty((B, cls._M, cls._K), dtype=torch.int8).cuda()
        a[0, :, :] = 1
        a[0] = pack_to_i8(cls._a)
        assert (torch.equal(pack_to_i4(pack_to_i8(cls._a)), cls._a))
        b = torch.randint(-8, 7, (B, cls._N, cls._K), dtype=torch.int8).cuda()
        c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)
                         ).round().to(torch.int32)
        cls._c_ref = c_gt[0]
        cls._b = pack_to_i4(b[0])

    @classmethod
    def quantization(cls):
        cls._a = quik.symmetric.quantize(cls._src, cls._scale, 4)

    @classmethod
    def calculation(cls):
        cls._c = quik.matmul.int4Matmul(cls._a, cls._b)


class Int4FusionFp16Out(Int4MatmulInt32Out):
    @classmethod
    def calculation(cls):
        cls._c = quik.matmul.int4FusedDequantize(cls._a, cls._b, cls._scale_row, cls._scale_col, cls._y)

    @classmethod
    def matmul_randomizer(cls):
        super().matmul_randomizer()
        cls._c_ref = cls._c_ref * cls._scale_row * cls._scale_col + cls._y

    @classmethod
    def dequantization(cls):
        pass


class Int4SpMatmulInt32Out(Kernel):
    @classmethod
    def __init__(cls, M, N, K):
        super().__init__(M, N, K)
        cls._e = None

    @classmethod
    def quantize_randomizer(cls):
        super().quantize_randomizer()
        cls._src = torch.randn(cls._M, cls._K // 2).to(torch.float16).cuda()

    @classmethod
    def matmul_randomizer(cls):
        assert cls._K % 32 == 0
        b = torch.randint(-3, 3, (cls._N, cls._K), dtype=torch.int8).cuda()

        metadata = quik.matmul.int4GenRandomSparseMeta(cls._M, cls._K)
        rmeta = quik.matmul.int4ReorderMeta(metadata)
        cls._e = rmeta.cuda()

        cls._b = pack_to_i4(b)

        qa_uncompressed = quik.matmul.int4Uncompress(cls._a.cpu(), metadata, cls._M, cls._K)
        cls._c_ref = quik.matmul.int4Matmul(qa_uncompressed.cuda(), cls._b)

        del metadata, rmeta, qa_uncompressed

    @classmethod
    def quantization(cls):
        cls._a = quik.symmetric.quantize(cls._src, cls._scale, 4)

    @classmethod
    def calculation(cls):
        cls._c = quik.matmul.int4SpMatmul(cls._a, cls._b, cls._e)

    @classmethod
    def cleaning(cls):
        super().cleaning()
        del cls._e


class Int8MatmulInt32Out(Kernel):
    @classmethod
    def quantize_randomizer(cls):
        super().quantize_randomizer()
        cls._src = torch.randn(cls._M, cls._K).to(torch.float16).cuda()

    @classmethod
    def matmul_randomizer(cls):
        B = 1
        qa = torch.empty((B, cls._M, cls._K), dtype=torch.int8).cuda()
        qa[0, :, :] = 1
        qa[0] = cls._a
        qb = torch.randint(-8, 7, (B, cls._N, cls._K), dtype=torch.int8).cuda()
        cls._b = qb[0]
        c_gt = torch.bmm(qa.float(), qb.float().transpose(1, 2)
                         ).round().to(torch.int32)
        cls._c_ref = c_gt[0]

    @classmethod
    def calculation(cls):
        cls._c = quik.matmul.int8Matmul(cls._a, cls._b)

    @classmethod
    def quantization(cls):
        cls._a = quik.symmetric.quantize(cls._src, cls._scale, 8)


class Int8SpMatmulInt32Out(Kernel):
    @classmethod
    def __init__(cls, M, N, K):
        super().__init__(M, N, K)
        cls._e = None

    @classmethod
    def quantize_randomizer(cls):
        super().quantize_randomizer()
        cls._src = torch.randn(cls._M, cls._K // 2).to(torch.float16).cuda()

    @classmethod
    def matmul_randomizer(cls):
        assert cls._K % 32 == 0
        qb = torch.randint(-5, 5, (cls._N, cls._K), dtype=torch.int8).cuda()

        cls._b = qb

        metadata = quik.matmul.int8GenRandomSparseMeta(cls._M, cls._K)
        rmeta = quik.matmul.int8ReorderMeta(metadata)
        cls._e = rmeta.cuda()

        qa_uncompressed = quik.matmul.int8Uncompress(cls._a.cpu(), metadata, cls._M, cls._K)
        cls._c_ref = quik.matmul.int8Matmul(qa_uncompressed.cuda(), qb)
        del metadata, rmeta, qa_uncompressed

    @classmethod
    def calculation(cls):
        cls._c = quik.matmul.int8SpMatmul(cls._a, cls._b, cls._e)

    @classmethod
    def quantization(cls):
        cls._a = quik.symmetric.quantize(cls._src, cls._scale, 8)

    @classmethod
    def cleaning(cls):
        super().cleaning()
        del cls._e


class Int8FusionFp16Out(Int8MatmulInt32Out):
    @classmethod
    def calculation(cls):
        cls._c = quik.matmul.int8FusedDequantize(cls._a, cls._b, cls._scale_row, cls._scale_col, cls._y)

    @classmethod
    def matmul_randomizer(cls):
        super().matmul_randomizer()
        cls._c_ref = cls._c_ref * cls._scale_row * cls._scale_col + cls._y

    @classmethod
    def dequantization(cls):
        pass


class Int8SpmmCuspLtFp16Out(Kernel):
    @classmethod
    def quantize_randomizer(cls):
        super().quantize_randomizer()
        cls._src = torch.randn(cls._M, cls._K // 2).to(torch.float16).cuda()

    @classmethod
    def matmul_randomizer(cls):
        b = torch.randint(-1, 2, (cls._N, cls._K), dtype=torch.int8).cuda()
        metadata = quik.matmul.int8GenRandomSparseMeta(cls._M, cls._K)
        a_uncompressed = quik.matmul.int8Uncompress(cls._qa.cpu(), metadata, cls._M, cls._K)
        cls._a = a_uncompressed.cuda()
        cls._b = b
        cls.__instance = quik.matmul.CusparseLtInt8SpMatmul(cls._a, cls._b)
        cls.__instance.compress()
        # a_compressed = cls.__instance.getACompressed()
        cls._c_ref = torch.matmul(cls._a.float(),
                                  cls._b.float().transpose(1, 0)).to(torch.float16)

    @classmethod
    def quantization(cls):
        cls._qa = quik.symmetric.quantize(cls._src, cls._scale, 8)

    @classmethod
    def calculation(cls):
        cls._c = cls.__instance.matmul_by(cls._b)

    @classmethod
    def dequantization(cls):
        cls._c_deq = quik.symmetric.dequantize(cls._c, cls._scale_row, cls._scale_col, cls._y, 16)

    @classmethod
    def cleaning(cls):
        super().cleaning()
        gc.collect()


class FP16Matmul(Kernel):
    @classmethod
    def quantize_randomizer(cls):
        pass

    @classmethod
    def matmul_randomizer(cls):
        B = 1
        cls._a = torch.rand((B, cls._M, cls._K), dtype=torch.float16).cuda()
        cls._b = torch.rand((B, cls._K, cls._N), dtype=torch.float16).cuda()

    @classmethod
    def calculation(cls):
        cls._c = torch.bmm(cls._a, cls._b)

    @classmethod
    def quantization(cls):
        pass

    @classmethod
    def dequantization(cls):
        pass

    @classmethod
    def verification(cls):
        pass


class FP32Matmul(Kernel):
    @classmethod
    def quantize_randomizer(cls):
        pass

    @classmethod
    def matmul_randomizer(cls):
        B = 1
        cls._a = torch.rand((B, cls._M, cls._K), dtype=torch.float32).cuda()
        cls._b = torch.rand((B, cls._K, cls._N), dtype=torch.float32).cuda()

    @classmethod
    def calculation(cls):
        cls._c = torch.bmm(cls._a, cls._b)

    @classmethod
    def quantization(cls):
        pass

    @classmethod
    def dequantization(cls):
        pass

    @classmethod
    def verification(cls):
        pass
