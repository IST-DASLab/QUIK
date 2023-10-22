import torch

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


def int4_fusion_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)

    src = torch.rand(M, K, dtype=torch.float16).cuda()
    scale = torch.rand(M, 1, dtype=torch.float16).cuda()
    a = quik.symmetric.quantize(src, scale, 4)
    a_ref = pack_to_i8(a)
    b_ref = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()
    b = pack_to_i4(b_ref)
    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()

    y = torch.rand(M, N).to(torch.float16).cuda()
    c = quik.matmul.int4FusedDequantize(a, b, scale_row, scale_col, y)

    c_ref = torch.matmul(a_ref.float(),
                         b_ref.float().transpose(1, 0)
                         ).round().to(torch.int32)

    c_ref = c_ref * scale_row * scale_col + y

    assert (torch.equal(c, c_ref))


def int4_sparse_fusion_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)
    src = torch.rand(M, K // 2, dtype=torch.float16).cuda()
    scale = torch.rand(M, 1, dtype=torch.float16).cuda()

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()
    y = torch.rand(M, N, dtype=torch.float16).cuda()

    a = quik.symmetric.quantize(src, scale, 4)

    metadata = quik.matmul.int4GenRandomSparseMeta(M, K)
    rmeta = quik.matmul.int4ReorderMeta(metadata)
    e = rmeta.cuda()

    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()
    b = pack_to_i4(b)

    c = quik.matmul.int4SpFusedDequantize(a, b, e, scale_row, scale_col, y)

    qa_uncompressed = quik.matmul.int4Uncompress(a.cpu(), metadata, M, K)
    c_ref = quik.matmul.int4FusedDequantize(qa_uncompressed.cuda(), b, scale_row, scale_col, y)

    assert (torch.equal(c, c_ref))


def int8_fusion_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)

    src = torch.rand(M, K, dtype=torch.float16).cuda()
    scale = torch.rand(M, 1, dtype=torch.float16).cuda()
    a = quik.symmetric.quantize(src, scale, 8)
    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()

    y = torch.rand(M, N).to(torch.float16).cuda()
    c = quik.matmul.int8FusedDequantize(a, b, scale_row, scale_col, y)

    c_ref = torch.matmul(a.float(),
                         b.float().transpose(1, 0)
                         ).round().to(torch.int32)
    c_ref = c_ref * scale_row * scale_col + y

    assert (torch.equal(c, c_ref))


def int8_sparse_fusion_test():
    M = 1024
    N = 5120
    K = 2048
    torch.manual_seed(1)
    src = torch.rand(M, K // 2, dtype=torch.float16).cuda()
    scale = torch.rand(M, 1, dtype=torch.float16).cuda()

    scale_row = torch.rand(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.rand(1, N, dtype=torch.float16).cuda()
    y = torch.rand(M, N, dtype=torch.float16).cuda()

    a = quik.symmetric.quantize(src, scale, 8)

    metadata = quik.matmul.int8GenRandomSparseMeta(M, K)
    rmeta = quik.matmul.int8ReorderMeta(metadata)
    e = rmeta.cuda()

    b = torch.randint(-3, 3, (N, K), dtype=torch.int8).cuda()

    c = quik.matmul.int8SpFusedDequantize(a, b, e, scale_row, scale_col, y)

    qa_uncompressed = quik.matmul.int8Uncompress(a.cpu(), metadata, M, K)
    c_ref = quik.matmul.int8FusedDequantize(qa_uncompressed.cuda(), b, scale_row, scale_col, y)

    assert (torch.equal(c, c_ref))


if __name__ == "__main__":
    int4_fusion_test()
    int4_sparse_fusion_test()
    int8_fusion_test()
    int8_sparse_fusion_test()
    print("Verification passed")
