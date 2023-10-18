import numpy as np
import torch

import quik


def pack_to_i4(X):
    def two_compl(x, bits):
        return torch.where(x < 0, 2 ** bits + x, x)

    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4


def int4_kernel_test():
    B, M, K, N = 1, 512, 512, 32
    a = torch.randint(-8, 7, (B, M, K), dtype=torch.int8).cuda()
    b = torch.randint(-8, 7, (B, N, K), dtype=torch.int8).cuda()
    c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)
                     ).round().to(torch.int32)
    qa = pack_to_i4(a[0])
    qb = pack_to_i4(b[0])
    c = quik.matmul.int4Matmul(qa, qb)

    assert torch.equal(c, c_gt[0])


def int4_spmm_test():
    torch.manual_seed(1)
    np.random.seed(1)
    M, K, N = 256, 512, 128
    assert K % 16 == 0
    a = torch.randint(-3, 3, (M, K // 2), dtype=torch.int32).cuda()
    metadata = quik.matmul.int4GenRandomSparseMeta(M, K)

    rmeta = quik.matmul.int4ReorderMeta(metadata)

    e = rmeta.cuda()

    b = torch.randint(-3, 3, (N, K), dtype=torch.int32).cuda()

    qa = pack_to_i4(a)
    qb = pack_to_i4(b)
    c = quik.matmul.int4SpMatmul(qa, qb, e)

    qa_uncompressed = quik.matmul.int4Uncompress(qa.cpu(), metadata, M, K)
    c_ref = quik.matmul.int4Matmul(qa_uncompressed.cuda(), qb)

    assert torch.equal(c, c_ref)


def int8_spmm_test():
    torch.manual_seed(1)
    np.random.seed(1)
    M, K, N = 256, 512, 128
    assert K % 16 == 0
    qa = torch.randint(-5, 5, (M, K // 2), dtype=torch.int8).cuda()
    metadata = quik.matmul.int8GenRandomSparseMeta(M, K)

    rmeta = quik.matmul.int8ReorderMeta(metadata)

    e = rmeta.cuda()

    qb = torch.randint(-5, 5, (N, K), dtype=torch.int8).cuda()

    c = quik.matmul.int8SpMatmul(qa, qb, e)

    qa_uncompressed = quik.matmul.int8Uncompress(qa.cpu(), metadata, M, K)
    c_ref = quik.matmul.int8Matmul(qa_uncompressed.cuda(), qb)

    assert torch.equal(c, c_ref)


if __name__ == '__main__':
    int4_kernel_test()
    int4_spmm_test()
    int8_spmm_test()
