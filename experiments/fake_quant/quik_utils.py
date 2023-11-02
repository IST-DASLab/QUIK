import quant
import torch
import math


class QUIK:
    def __init__(self, layer,
                 act_scales=None,
                 fp_features=0):
        
        if isinstance(layer, quant.ActQuantWrapper):
            layer = layer.module
            
        assert isinstance(layer, torch.nn.Linear), 'QUIK only supports torch.nn.Linear layers!'
                  
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.act_scales = act_scales
        self.fp_features = fp_features
        
        self.int_indices = None
        self.fp_indices = None
        self.col_perm = None
        self.inv_col_perm = None
        
        if fp_features > 0:    
            self.fp_indices = torch.sort(act_scales)[1][-fp_features:]
            self.int_indices = torch.sort(act_scales)[1][:-fp_features]
            self.col_perm = act_scales.sort()[1]
            self.inv_col_perm = torch.zeros_like(self.col_perm)
            self.inv_col_perm[self.col_perm] = torch.arange(self.col_perm.numel())
         
    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, torch.nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1
    ):
        W = self.layer.weight.data.clone()
        
        W = W.float()

        if not self.quantizer.ready():
            if self.fp_features > 0:
                # We calculate the quantization parameters only on the integer part of the weights
                # Then, we permute the columns so the integer part is on the left
                self.quantizer.find_params(W[:, self.int_indices])
                W = W[:, self.col_perm]
                self.H = self.H[self.col_perm, :][:, self.col_perm]
            else:
                self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            if i1 >= self.columns - self.fp_features and self.fp_features > 0:
                Q[:, i1:i2] = W[:, i1:i2].clone()
                continue
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                if i + i1 >= self.columns - self.fp_features and self.fp_features > 0:
                    Q1[:, i] = W1[:, i]
                    continue
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        # Permut Back the weights
        if self.fp_features > 0:
            self.layer.weight.data = self.layer.weight.data[:, self.inv_col_perm]
            
    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()