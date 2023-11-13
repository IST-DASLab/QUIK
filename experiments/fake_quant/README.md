# Fake Quantization using QUIK


In this directory, we provide the torch scripts for the experiments in QUIK. We mainly focus on the language generation task. We provide the scripts for OPT, LLaMA-2, and Falcon models. 

### Dependencies

- `torch`: tested on v2.0.0
- `transformers`: tested on v4.31.0
- `datasets`: tested on v1.17.0


## Language Generation

Currently, we have the implementation of the following models:
- OPT in `opt.py`
- LLaMA-2 in `llama.py`
- Falcon in `falcon.py`

You can simply run the above files to reproduce the results in the paper. The main arguments are:

- `--model`: the model name (or path to the weights)
- `--dataset`: the calibration dataset for GPTQ quantization
- `--fp_features`: the number of outliers (should be saved in `act_scales` directory)
- `--fp_relative`: Whether we want to use more outliers (proportional to the number of input features) in the `down_proj` or `fc2` in LLaMA-2 and Falcon models.
- `--int8_down_proj`: Whether we want to use int8 quantization for `down_proj`  in LLaMA-2 models.
- `--int8_fc2`: Whether we want to use int8 quantization for `fc2`  in Falcon models.
- `--hf_token`: HuggingFace token for accessing to the LLaMA-2 and Falcon models.
- `--a_bits`: the number of bits for activation quantization
- `--w_bits`: the number of bits for weight quantization
- `--w_clip`: Whether we want to clip the weights
- `sparsity`: Sparse weights using SparseGPT (for Falcon models)
- `prunen`: N for N:M using SparseGPT (for Falcon models)
- `prunem`: M for N:M using SparseGPT (for Falcon models)

For example, to run LLaMA2-70B model with `256` outliers, you can run the following command:

```bash
python llama.py --model meta-llama/Llama-2-70b-hf  --fp_features 256 --fp_relative --a_bits 4 --w_bits 4 --w_clip --hf_token <your_hf_token>
```
