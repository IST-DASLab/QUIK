# QUIK
This repository contains the code for QUIK, weights and activations post-training quantization.

## Install

### Dependencies

- cmake
- C++ compiler (GCC/clang/...)
- CUDA (tested with 11.8)


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
python llama.py --fp_features_num 256 --model meta-llama/Llama-2-7b-hf --hf-token $HF_TOKEN --dataset c4 \ 
--w_bits 4 --a_bits 4 --load_qmodel_path gptq_model_path --int8_down_proj --benchmark 
```

Benchmark will be run on all available GPUs.
### Linear layer benchmarks
Linear layer benchmarks can be run with ``python layer_benchmark.py``. One can vary input size with command line parameters.


### Model adapt to QUIK
First, one has to quantize the model weights using GPTQ algorithm. In `llama.py` it is done with `llama_sequential` function.
From that we get quantized weights (that are still stored in `torch.float16`).
Then ones needs create QUIK Linear layers using `qlinear.MixedQLinear.from_float` that must replace original Linear layers. See `llama_replace_with_kernels` in `llama.py`.
Now the quantized model is ready for use.

