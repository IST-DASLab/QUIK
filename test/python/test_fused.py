import random
import torch

import quik


def pack_to_i4(X):
    def two_compl(x, bits):
        return torch.where(x < 0, 2 ** bits + x, x)

    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4


def int4_fused_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)

    a_ref = torch.randint(-3, 3, (M, K), dtype=torch.int8).cuda()
    a = pack_to_i4(a_ref)
    b_ref = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()
    b = pack_to_i4(b_ref)
    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()

    y = torch.rand(M, N).to(torch.float16).cuda()
    c = quik.symmetric.int4FusedDequantize(a, b, scale_row, scale_col, y)

    c_ref = torch.matmul(a_ref.float(),
                         b_ref.float().transpose(1, 0)
                         ).round().to(torch.int32)

    c_ref = c_ref * scale_row * scale_col + y

    assert (torch.equal(c, c_ref))


def int4_asy_fused_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(2)

    a_ref = torch.randint(-3, 3, (M, K), dtype=torch.int8).cuda()
    a = pack_to_i4(a_ref)
    b_ref = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()
    b = pack_to_i4(b_ref)
    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()
    shift_value = random.random()
    zero_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    w_reduce = torch.rand(1, N, dtype=torch.float16).cuda()
    y = torch.rand(M, N, dtype=torch.float16).cuda()

    c_ref = torch.matmul(a_ref.float(),
                         b_ref.float().transpose(1, 0)
                         ).round().to(torch.int32)

    c_ref = c_ref * scale_col * scale_row + \
            (zero_row + torch.tensor([shift_value], dtype=torch.float16).item() * scale_row) * w_reduce + y

    c = quik.asymmetric.int4FusedDequantize(a, b, scale_row, scale_col, shift_value, zero_row, w_reduce, y)

    assert (torch.equal(c, c_ref))


def int4_sparse_fused_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()
    y = torch.rand(M, N, dtype=torch.float16).cuda()

    a = torch.randint(-3, 3, (M, K // 2), dtype=torch.int8).cuda()
    a = pack_to_i4(a)

    metadata = quik.matmul.int4GenRandomSparseMeta(M, K)
    rmeta = quik.matmul.int4ReorderMeta(metadata)
    e = rmeta.cuda()

    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()
    b = pack_to_i4(b)

    c = quik.symmetric.int4SpFusedDequantize(a, b, e, scale_row, scale_col, y)

    a_uncompressed = quik.matmul.int4Uncompress(a.cpu(), metadata, M, K)
    c_ref = quik.symmetric.int4FusedDequantize(a_uncompressed.cuda(), b, scale_row, scale_col, y)

    assert (torch.equal(c, c_ref))


def int4_asy_sparse_fused_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)

    a = torch.randint(-3, 3, (M, K // 2), dtype=torch.int8).cuda()
    a = pack_to_i4(a)

    metadata = quik.matmul.int4GenRandomSparseMeta(M, K)
    rmeta = quik.matmul.int4ReorderMeta(metadata)
    e = rmeta.cuda()

    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()
    b = pack_to_i4(b)

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()
    shift_value = random.random()
    zero_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    w_reduce = torch.rand(1, N, dtype=torch.float16).cuda()
    y = torch.rand(M, N, dtype=torch.float16).cuda()

    c = quik.asymmetric.int4SpFusedDequantize(a, b, e, scale_row, scale_col, shift_value, zero_row, w_reduce, y)

    a_uncompressed = quik.matmul.int4Uncompress(a.cpu(), metadata, M, K)
    c_ref = quik.asymmetric.int4FusedDequantize(a_uncompressed.cuda(), b, scale_row, scale_col, shift_value,
                                                zero_row, w_reduce, y)

    assert (torch.equal(c, c_ref))


def int8_fused_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)

    a = torch.randint(-3, 3, (M, K), dtype=torch.int8).cuda()
    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()

    y = torch.rand(M, N).to(torch.float16).cuda()
    c = quik.symmetric.int8FusedDequantize(a, b, scale_row, scale_col, y)

    c_ref = torch.matmul(a.float(),
                         b.float().transpose(1, 0)
                         ).round().to(torch.int32)
    c_ref = c_ref * scale_row * scale_col + y

    assert (torch.equal(c, c_ref))


def int8_asy_fused_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(2)

    a = torch.randint(-3, 3, (M, K), dtype=torch.int8).cuda()
    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()
    shift_value = random.random()
    zero_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    w_reduce = torch.rand(1, N, dtype=torch.float16).cuda()
    y = torch.rand(M, N, dtype=torch.float16).cuda()

    c_ref = torch.matmul(a.float(),
                         b.float().transpose(1, 0)
                         ).round().to(torch.int32)

    c_ref = c_ref * scale_col * scale_row + \
            (zero_row + torch.tensor([shift_value], dtype=torch.float16).item() * scale_row) * w_reduce + y

    c = quik.asymmetric.int8FusedDequantize(a, b, scale_row, scale_col, shift_value, zero_row, w_reduce, y)
    assert (torch.equal(c, c_ref))


def int8_sparse_fused_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()
    y = torch.rand(M, N, dtype=torch.float16).cuda()

    a = torch.randint(-3, 3, (M, K // 2), dtype=torch.int8).cuda()

    metadata = quik.matmul.int8GenRandomSparseMeta(M, K)
    rmeta = quik.matmul.int8ReorderMeta(metadata)
    e = rmeta.cuda()

    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()

    c = quik.symmetric.int8SpFusedDequantize(a, b, e, scale_row, scale_col, y)

    a_uncompressed = quik.matmul.int8Uncompress(a.cpu(), metadata, M, K)
    c_ref = quik.symmetric.int8FusedDequantize(a_uncompressed.cuda(), b, scale_row, scale_col, y)

    assert (torch.equal(c, c_ref))


def int8_asy_sparse_fused_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)

    a = torch.randint(-3, 3, (M, K // 2), dtype=torch.int8).cuda()

    metadata = quik.matmul.int8GenRandomSparseMeta(M, K)
    rmeta = quik.matmul.int8ReorderMeta(metadata)
    e = rmeta.cuda()

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()
    shift_value = random.random()
    zero_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    w_reduce = torch.rand(1, N, dtype=torch.float16).cuda()
    y = torch.rand(M, N, dtype=torch.float16).cuda()

    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()

    c = quik.asymmetric.int8SpFusedDequantize(a, b, e, scale_row, scale_col, shift_value, zero_row, w_reduce, y)

    a_uncompressed = quik.matmul.int8Uncompress(a.cpu(), metadata, M, K)
    c_ref = quik.asymmetric.int8FusedDequantize(a_uncompressed.cuda(), b, scale_row, scale_col, shift_value,
                                                zero_row, w_reduce, y)

    assert (torch.equal(c, c_ref))


if __name__ == "__main__":
    int4_fused_test()
    int4_asy_fused_test()
    int4_sparse_fused_test()
    int4_asy_sparse_fused_test()
    int8_fused_test()
    int8_asy_fused_test()
    int8_sparse_fused_test()
    int8_asy_sparse_fused_test()
    print("Verification passed")
