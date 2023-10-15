import torch
import quant_sim
import transformers
from qlinear import MixedQLinear
from quant_sim import ActQuantWrapper
import os


def get_fp_features_num(module: torch.nn.Linear, args, model=None):
    fp_features_num = args.fp_features_num
    if args.fp_features_frac is not None:
        fp_features_num = max(int(module.in_features * args.fp_features_frac), fp_features_num)
    if getattr(args, "fp_relative", False) and model is not None:
        fp_features_num = int(module.in_features / model.config.hidden_size) * fp_features_num
    return fp_features_num


def skip(*args, **kwargs):
    pass


def find_layers(module, layers=[torch.nn.Linear, quant_sim.ActQuantWrapper,
                                transformers.models.falcon.modeling_falcon.FalconLinear], name=''):
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    # if type(module) in layers:
    #     return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_opt(model):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    print('Loading {} Model...'.format(model))
    model = transformers.OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model



def get_llama(model_name, seqlen, hf_token):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    config_path = f"configs/{model_name.split('/')[1]}.json"
    if os.path.exists(config_path):
        config = transformers.LlamaConfig.from_json_file(config_path)
        config._flash_attn_2_enabled = True
    else:
        config = None
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto', use_auth_token=hf_token,
                                                          config=config)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if not os.path.exists(config_path):
        config = model.config
        config._flash_attn_2_enabled = True
        # reload with new config options applied
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto', use_auth_token=hf_token,
                                                              config=config)
        config.to_json_file(config_path, use_diff=False)
    model.seqlen = seqlen
    return model



def replace_single_mod_opt(module, name, layer_to_replace):
    if isinstance(module, MixedQLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in [torch.nn.Linear, ActQuantWrapper]:
            if attr in name:
                setattr(module, attr, layer_to_replace)

    for name1, child in module.named_children():
        replace_single_mod_opt(child, name + '.' + name1 if name != '' else name1, layer_to_replace)
