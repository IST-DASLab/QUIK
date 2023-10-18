import torch
import unittest

import quik

NUM_ROWS = 4096
NUM_COLS = 11088


class TestQuan(unittest.TestCase):
    def test_meta(self):
        for BITS in [4, 8]:
            t = torch.rand((NUM_ROWS, NUM_COLS), dtype=torch.float16, device='cuda')
            xmin = t.min(1)[0]
            xmax = t.max(1)[0]
            unit = torch.maximum(((xmax - xmin) / ((1 << BITS) - 1)), torch.tensor(1e-6, device=xmax.device))

            meta = quik.asymmetric.find_meta(t, BITS)
            zeros = meta[1::2]
            scale = meta[::2]
            self.assertAlmostEqual(torch.max(torch.abs(xmin - zeros)).item(), 0.0, 3, f"Wrong min for {BITS}")
            self.assertAlmostEqual(torch.max(torch.abs(scale - unit)).item(), 0.0, 3, f"Wrong scale for {BITS}")

    def test_quantize(self):
        fp_features_num = 256
        int_indices = torch.arange(NUM_COLS)
        t = torch.rand((NUM_ROWS, NUM_COLS), dtype=torch.float16, device='cuda')
        fp_indices = torch.randperm(NUM_COLS)[:fp_features_num]
        int_indices = int_indices[~torch.isin(int_indices, fp_indices)]
        fp_t_base = t[:, fp_indices]
        int_t = t[:, int_indices]

        for BITS in [4, 8]:
            meta_base = quik.asymmetric.find_meta(int_t, BITS)
            q_base = quik.asymmetric.quantizeOld(int_t, meta_base, BITS)
            q, meta, fp_t = quik.asymmetric.quantize(t, int_indices.cuda(), fp_indices.cuda(), BITS)
            self.assertAlmostEqual(torch.max(torch.abs(meta - meta_base)).item(), 0.0, 3, f"Wrong meta")
            self.assertAlmostEqual(torch.max(torch.abs(fp_t - fp_t_base)).item(), 0.0, 3, f"Wrong full precision")
            self.assertAlmostEqual(torch.max(torch.abs(q - q_base)).item(), 0.0, 3, f"Wrong quantized")


if __name__ == '__main__':
    unittest.main()
