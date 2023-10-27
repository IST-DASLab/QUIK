import torch
from qlinear import MixedQLinear, Linear8bit, Linear4bit
import time
import argparse
import numpy as np


fp_features_num = 256
model_sizes = [(4096, 4096), (8192, 1024), (5120, 5120), (8192, 8192), (11008, 4096), (28672, 8192)]

def benchmark(args):
    global model_sizes
    input_size = args.input_size
    for (feature_dim_in, feature_dim_out) in model_sizes:
        for dtype in [torch.float16]:
            x = torch.rand((input_size, feature_dim_in)).cuda().to(dtype)
            def run_benchmark(module):
                num_bench_steps = 100
                for i in range(10):
                    out = module(x)
                start_time = time.perf_counter()
                torch.cuda.synchronize()
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
                for i in range(num_bench_steps):
                    out = module(x)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStop()
                return (end_time - start_time) * 1000 / num_bench_steps
            baseline_mod = torch.nn.Linear(feature_dim_in, feature_dim_out, bias=False).cuda().to(dtype)
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data, low=-8, high=7).to(dtype)
            fp_indices = torch.randperm(feature_dim_in)[:fp_features_num]
            s_w = torch.ones((feature_dim_out, 1), dtype=dtype, device='cuda')
            int4_mod = MixedQLinear.from_float(baseline_mod,
                                               baseline_mod.weight.data,
                                               s_w, shared_input=None,
                                               fp_indices=fp_indices, bits=4).cuda()
            int8_mod = MixedQLinear.from_float(baseline_mod,
                                               baseline_mod.weight.data,
                                               s_w, shared_input=None,
                                               fp_indices=None, bits=8).cuda()


            # int4_mod.fused_quantization = False
            # int4_mod.fused_dequantization = False
            # times = []
            # for i in range(10):
            #     times.append(run_benchmark(int4_mod))
            # print(f"Int4 v1 time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms")
            #
            # int4_mod.fused_quantization = True
            # int4_mod.fused_dequantization = False
            # times = []
            # for i in range(10):
            #     times.append(run_benchmark(int4_mod))
            # print(f"Int4 v2 time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms")

            # int4_mod.fused_quantization = True
            # int4_mod.fused_dequantization = True

            times = []
            for i in range(10):
                times.append(run_benchmark(int4_mod))
            print(f"Int4 v3 time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms")

            times = []
            for i in range(10):
                times.append(run_benchmark(int8_mod))
            print(f"Int8 time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms")
            times = []
            for i in range(10):
                times.append(run_benchmark(baseline_mod))
            print(f"FP16 time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-size', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--profile', help='Do profile',
        action='store_true',
    )
    args = parser.parse_args()
    benchmark(args)
