# QUIK
This repository contains the code for QUIK, weights and activations post-training quantization.

## Install

### Dependencies

- cmake
- C++ compiler (GCC/clang/...)
- nvcc

### Instructions

```bash
git clone git@github.com:IST-DASLab/QUIK.git
cd QUIK
pip install -e .  # or pip install .
```

## Example

### LLama example
```bash
cd experiments
pip install -r requirements.txt
python llama.py --fp_features_num 256 --model meta-llama/Llama-2-7b-hf --hf_token <your_hf_token> --dataset c4 \ 
--w_bits 4 --w_clip --a_bits 4 --save_qmodel_path save_gptq_model_path --int8_down_proj --sim_eval --benchmark 
```

Benchmark will be run on all available GPUs.
### Linear layer benchmarks
Linear layer benchmarks can be run with ``python layer_benchmark.py``. One can vary input size with command line parameters.


### Model adapt to QUIK
First, one has to quantize the model weights using GPTQ algorithm. In `llama.py` it is done with `llama_sequential` function.
From that we get quantized weights (that are still stored in `torch.float16`).
Then ones needs create QUIK Linear layers using `qlinear.MixedQLinear.from_float` that must replace original Linear layers. See `llama_replace_with_kernels` in `llama.py`.
Now the quantized model is ready for use.


### Fake Quantization examples

To run the fake quantization example, check [`fake_quant`](https://github.com/IST-DASLab/QUIK/tree/master/experiments/fake_quant) directory.
