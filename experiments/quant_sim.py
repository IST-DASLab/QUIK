import numpy as np
import torch


def asymmetric_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def symmetric_quantize(x, scale, bits):
    if bits == 16:
        return x
    elif bits == 4:
        q = torch.clamp(torch.round(x / scale), -8, 7)
    elif bits == 8:
        q = torch.clamp(torch.round(x / scale), -128, 127)
    return scale * q

class WeightQuantizer(torch.nn.Module):
    '''
        A class for quantizing the weights. 
        We support both symmetric and asymmetric quantization for both perchannel and pertensor.
    '''
    
    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
            self,
            bits, perchannel=False, sym=True, mse=False,
        ):
        self.bits = bits
        if sym:
            if self.bits == 4:
                self.maxq = torch.tensor(7)
            elif self.bits == 8:
                self.maxq = torch.tensor(127)
            else:
                raise ValueError('Only 4/8-bit is supported!')
        else:
            if self.bits == 4:
                self.maxq = torch.tensor(15)
            elif self.bits == 8:
                self.maxq = torch.tensor(255)
            else:
                raise ValueError('Only 4/8-bit is supported!')
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.grid = 100
        self.maxshrink = 0.8
        self.norm = 2.4

    def find_params(self, x):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        
        if self.sym:
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmax1 = p * xmax
                xmin1 = p * xmin

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = symmetric_quantize(x, scale1.unsqueeze(1), self.bits)
                else:
                    
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asymmetric_quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)           
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
            
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        if torch.any(torch.isnan(self.scale)):
            print('nan scale WARNING')
            print(self.scale, self.zero)
            print(xmax)
            print(self.maxq)
        return

    def quantize(self, x):
        if self.bits == 16:
            return x
        
        if self.ready():
            if self.sym:
                return symmetric_quantize(x, self.scale, self.bits)
            return asymmetric_quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

class ActQuantizer(torch.nn.Module):
    
    '''
        A class for quantizing the activations. We only support the asymmetric/pertoken quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16
        self.configured = False

    def forward(self, x):
        return asymmetric_quantize(x, self.scale, self.zero, self.maxq)  

    def configure(self, bits):    
        self.maxq = torch.tensor(2 ** bits - 1)          
        self.bits = bits
        self.configured = True

    def find_params(self, x):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape
        
        if len(init_shape) == 3:
                assert init_shape[0] == 1, 'Only batch size of 1 is supported!'
        
        reshaped_x = x.reshape((-1, x.shape[-1]))
             
        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp)
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)
                
        self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
        self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

            
        

class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
    '''

    def __init__(self, module):
        super(ActQuantWrapper, self).__init__()
        self.module = module
        self.quantizer = ActQuantizer() 
        self.fp_features_num = 0
        self.act_scales = None
    
    def extra_repr(self) -> str:
         str_ = f'Act. Bits: {self.quantizer.bits}, FP features: {self.fp_features_num}'
         return str_

    def fp_features_configure(self, scales, fp_features):
        
        '''
            This function extract the indices of the features that will be quantized a
            nd the ones that will be kept in FP16.
        '''
        self.act_scales = scales
        self.fp_features_num = fp_features

        if fp_features == 0:
            self.quantized_feature_idx = torch.arange(scales.shape[0])
            self.fp_feature_idx = None
        else:
            self.fp_feature_idx = torch.sort(scales)[1][-fp_features:]
            self.quantized_feature_idx = torch.sort(scales)[1][:-fp_features]

    def forward(self, x):
        x_dtype = x.dtype
        
        if self.quantizer.bits == 16:
            return self.module(x).to(x_dtype)
        
        if self.fp_features_num == 0: #corner case: Quantize all features
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            return self.module(x).to(x_dtype)
        
        mixed_precition_x = torch.zeros_like(x)
        
        if len(x.shape) == 3:
            q_features = x[:, :, self.quantized_feature_idx]
            fp_features = x[:, :, self.fp_feature_idx]
        elif len(x.shape) == 2:
            q_features = x[:, self.quantized_feature_idx]
            fp_features = x[:, self.fp_feature_idx]

        self.quantizer.find_params(q_features)
        q_features = self.quantizer(q_features).to(x_dtype)
        
        if len(x.shape) == 3:
            mixed_precition_x[:, :, self.quantized_feature_idx] = q_features
            mixed_precition_x[:, :, self.fp_feature_idx] = fp_features
        elif len(x.shape) == 2:
            mixed_precition_x[:, self.quantized_feature_idx] = q_features
            mixed_precition_x[:, self.fp_feature_idx] = fp_features
        return self.module(mixed_precition_x).to(x_dtype)

def add_actquant(module, name='', layers=[torch.nn.Linear, ActQuantWrapper]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)
    