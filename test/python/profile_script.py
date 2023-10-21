import numpy as np
import torch

import samples

quant_time_all_type_s = []
matmul_time_all_type_s = []
dequant_time_all_type_s = []
total_time_all_type_s = []
performance_all_type_TFLOPS = []


def timer(func):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    func()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event)


def profiler(shape_list, profile_type_list, dont_need_quant_and_dequant, dont_need_dequant, repetitions=300,
             preheats=5):
    for instance in profile_type_list:
        quant_time_per_type_s = []
        matmul_time_per_type_s = []
        dequant_time_per_type_s = []
        total_time_per_type_s = []
        TFLOPS_per_type = []
        for val in shape_list:
            M, N, K = val, val, val
            # M, N = 10240, 10240
            # K = val
            instance(M, N, K)
            instance.quantize_randomizer()
            multiple_time_quant_ms = 0.0
            multiple_time_matmul_ms = 0.0
            multiple_time_dequant_ms = 0.0
            for n in range(repetitions):

                single_time_quant_ms = timer(instance.quantization)

                if (n == 0):
                    instance.matmul_randomizer()

                single_time_matmul_ms = timer(instance.calculation)

                if (n == 0):
                    instance.verification()
                    print(instance.__name__, " verification passed.")

                single_time_dequant_ms = timer(instance.dequantization)

                if (n >= preheats):
                    multiple_time_quant_ms += single_time_quant_ms
                    multiple_time_matmul_ms += single_time_matmul_ms
                    multiple_time_dequant_ms += single_time_dequant_ms

                if (instance in dont_need_quant_and_dequant):
                    multiple_time_quant_ms = 0.0
                    multiple_time_dequant_ms = 0.0

                if (instance in dont_need_dequant):
                    multiple_time_dequant_ms = 0.0

            instance.cleaning()
            instance.empty_cache()

            average_time_quant_per_shape_s = multiple_time_quant_ms / 1e3 / (repetitions - preheats)
            average_time_matmul_per_shape_s = multiple_time_matmul_ms / 1e3 / (repetitions - preheats)
            average_time_dequant_per_shape_s = multiple_time_dequant_ms / 1e3 / (repetitions - preheats)

            total_time_per_shape_s = average_time_quant_per_shape_s + \
                                     average_time_matmul_per_shape_s + \
                                     average_time_dequant_per_shape_s

            FLOPs_per_shape = M * N * (2 * K - 1)
            TFLOPS_per_shape = round(FLOPs_per_shape / average_time_matmul_per_shape_s / 1e12)

            print("label: ", instance.__name__, " shape: ", val)
            print(f"average quantization time: {average_time_quant_per_shape_s:.2e}(s)")
            print(f"average matmul time: {average_time_matmul_per_shape_s:.2e}(s)")
            print(f"average dequantization time: {average_time_dequant_per_shape_s:.2e}(s)")
            print(f"total time: {total_time_per_shape_s:.2e}(s)")
            print("TFLOPS: ", TFLOPS_per_shape)
            quant_time_per_type_s.append(average_time_quant_per_shape_s)
            matmul_time_per_type_s.append(average_time_matmul_per_shape_s)
            dequant_time_per_type_s.append(average_time_dequant_per_shape_s)
            total_time_per_type_s.append(total_time_per_shape_s)
            TFLOPS_per_type.append(TFLOPS_per_shape)

        quant_time_all_type_s.append(quant_time_per_type_s)
        matmul_time_all_type_s.append(matmul_time_per_type_s)
        dequant_time_all_type_s.append(dequant_time_per_type_s)
        total_time_all_type_s.append(total_time_per_type_s)
        performance_all_type_TFLOPS.append(TFLOPS_per_type)


import os


def writer(data_dict, path="test/data/"):
    data_directory_path = path

    if not os.path.exists(data_directory_path):
        os.makedirs(data_directory_path)

    for label, data in data_dict.items():
        with open(data_directory_path + label + ".txt", "w") as f:
            np.savetxt(f, data, fmt="%.3e")


shape_list = [256, 512] + \
             [x * 1024 for x in range(1, 15)]

profile_type_list = [
    samples.Int4MatmulInt32Out,
    samples.Int4SpMatmulInt32Out,
    samples.Int8MatmulInt32Out,
    samples.Int8SpMatmulInt32Out,
    samples.FP16Matmul,
    samples.FP32Matmul,
    # samples.Int8SpmmCuspLtFp16Out,
    samples.Int4FusionFp16Out,
    samples.Int8FusionFp16Out,
]

dont_need_quant_and_dequant = [
    samples.FP16Matmul,
    samples.FP32Matmul,
]

dont_need_dequant = [
    samples.Int4FusionFp16Out,
    samples.Int8FusionFp16Out,
]

written_data_directory = "test/python/data/"

written_data_dict = {
    "quantization": quant_time_all_type_s,
    "matmul": matmul_time_all_type_s,
    "dequantization": dequant_time_all_type_s,
    "total": total_time_all_type_s,
    "performance": performance_all_type_TFLOPS,
}

written_data_label = list(written_data_dict.keys())

if __name__ == "__main__":
    repetitions = 300
    preheats = 10
    profiler(shape_list, profile_type_list,
             dont_need_quant_and_dequant,
             dont_need_dequant, repetitions=repetitions, preheats=preheats)

    # writer(written_data_dict, path=written_data_directory)
