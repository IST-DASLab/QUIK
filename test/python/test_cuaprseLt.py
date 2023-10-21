import gc
import numpy as np
import torch

import quik


def cusparseLt_test():
    M, N, K = 1024, 512, 2048
    ### Quantiza randomizer
    torch.manual_seed(1)
    np.random.seed(1)
    src = torch.randn(N, K, dtype=torch.float16).cuda()
    scale = torch.randn(N, 1, dtype=torch.float16).cuda()

    scale_row = torch.randn(M, 1, dtype=torch.float16).cuda()
    scale_col = torch.randn(1, N, dtype=torch.float16).cuda()
    y = torch.randn(M, N, dtype=torch.float16).cuda()

    ### Quantization
    qb = quik.symmetric.quantize(src, scale, 8)
    qb_ref = (src / scale).round().to(torch.int8)
    assert (torch.equal(qb, qb_ref))

    ### Matmul randomizer
    a_compressed = torch.randint(-3, 3, (M, K // 2), dtype=torch.int8).cuda()
    metadata = quik.matmul.int8GenRandomSparseMeta(M, K)
    a_uncompressed = quik.matmul.int8Uncompress(a_compressed.cpu(), metadata, M, K)
    a = a_uncompressed.cuda()
    b = qb

    c_ref = torch.matmul(a.float(),
                         b.float().transpose(1, 0)).to(torch.float16)

    ### Init cusparseLt context
    instance = quik.matmul.CusparseLtInt8SpMatmul(a, b)
    # instance = quik.matmul.CusparseLtInt8SpMatmul(a, b, 0) # The final parameter for algorithm selection is set to 0 by default.
    instance.compress()
    ### Calculate
    c = instance.matmul_by(b)  # You can input other B's that have the same shape as B

    assert (torch.equal(c, c_ref))

    ### Dequantization
    c_deq = quik.symmetric.dequantize(c, scale_row, scale_col, y, 16)
    c_deq_ref = c * scale_row * scale_col + y
    assert (torch.equal(c_deq, c_deq_ref))

    ### Destruct the instance
    gc.collect()


if __name__ == "__main__":
    cusparseLt_test()
    print("Verification passed.")
