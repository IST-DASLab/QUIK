
'''
    This is the code for SmoothQuant from different files in their repo.
'''

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconAttention, FalconMLP
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTAttention
import torch
from torch import nn
from functools import partial

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w_dev = w.device
    w = w.cuda()
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    w = w.to(w_dev)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w_dev = w.device
    w = w.cuda()
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    w = w.to(w_dev)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant='per_token', quantize_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, weight_quant='per_channel', act_quant='per_token', quantize_output=False):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output)
        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'
    

@torch.no_grad()
def smooth_opt_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.float().cuda()
    weight_scales = torch.cat([fc.weight.float().abs().cuda().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales).to(device).to(dtype)
    ln.bias.div_(scales).to(device).to(dtype)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1)).to(device).to(dtype)


@torch.no_grad()
def smooth_opt(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_opt_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']
            smooth_opt_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

@torch.no_grad()
def quantize_opt(model, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=False, act_scales=None):
    alpha = 0.5
    from tqdm import tqdm
    model = model.cpu()
    if act_scales is not None:
        smooth_opt(model, act_scales, alpha)
        print('SmoothQuant: Smoothing Done with alpha = {}!'.format(alpha))
    else:
        print('SmoothQuant: No smoothing!')
        raise ValueError
    for name, m in tqdm(model.model.named_modules(), desc='SmoothQuant', total=len(list(model.model.named_modules()))):
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant).cpu()
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant).cpu()
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input).cpu()
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input).cpu()
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input).cpu()
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant).cpu()
    return model

@torch.no_grad()
def smooth_llama_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, LlamaRMSNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.float().cuda()
    weight_scales = torch.cat([fc.weight.float().abs().cuda().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    ln = ln.to(device).to(dtype)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1)).to(device).to(dtype)

        
@torch.no_grad()
def smooth_llama(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_llama_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            ffn_inputs = [
                module.mlp.up_proj,                                
                module.mlp.gate_proj,
            ]
            fc1_input_scales = scales[name + '.mlp.up_proj']
            smooth_llama_fcs(ffn_ln, ffn_inputs, fc1_input_scales, alpha)


@torch.no_grad()
def quantize_llama(model, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=False, act_scales=None):
    alpha = 0.8
    from tqdm import tqdm
    model = model.cpu()
    if act_scales is not None:
        smooth_opt(model, act_scales, alpha)
        print('SmoothQuant: Smoothing Done with alpha = {}!'.format(alpha))
    for name, m in tqdm(model.model.named_modules(), desc='SmoothQuant', total=len(list(model.model.named_modules()))):
        if isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(m.gate_proj, weight_quant=weight_quant, act_quant=act_quant).cpu()
            m.up_proj = W8A8Linear.from_float(m.up_proj, weight_quant=weight_quant, act_quant=act_quant).cpu()
            m.down_proj = W8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant).cpu()
        elif isinstance(m, LlamaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input).cpu()
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input).cpu()
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input).cpu()
            m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant).cpu()
    return model



@torch.no_grad()
def smooth_falcon(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, FalconDecoderLayer):
            attn_ln = module.ln_attn
            qkv = [module.self_attention.query_key_value]
            qkv_input_scales = scales[name + '.self_attention.query_key_value']
            smooth_opt_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.ln_mlp
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']
            smooth_opt_fcs(ffn_ln, fc1, fc1_input_scales, alpha)



@torch.no_grad()
def quantize_falcon(model, weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=False, act_scales=None):
    assert model.config.new_decoder_architecture == True, 'We only support new decoder architecture for now (as they have two LN)!'
    alpha = 0.5
    from tqdm import tqdm
    assert model.device == torch.device('cpu'), 'We only support CPU for now!'
    if act_scales is not None:
        smooth_falcon(model, act_scales, alpha)
        print('SmoothQuant: Smoothing Done with alpha = {}!'.format(alpha))
    for name, m in tqdm(model.transformer.named_modules(), desc='SmoothQuant', total=len(list(model.transformer.named_modules()))):
        if isinstance(m, FalconMLP):
            new_mod = W8A8Linear.from_float(m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant)
            setattr(model, name, new_mod)
            new_mod = W8A8Linear.from_float(m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant)
            setattr(model, name, new_mod)
            del new_mod
        elif isinstance(m, FalconAttention):
            new_mod = W8A8Linear.from_float(m.query_key_value, weight_quant=weight_quant, act_quant=act_quant)
            setattr(model, name, new_mod)
            new_mod = W8A8Linear.from_float(m.dense, weight_quant=weight_quant, act_quant=act_quant)
            setattr(model, name, new_mod)
            del new_mod
    return model
