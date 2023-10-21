import torch


def quantize(src: torch.Tensor, scale: torch.Tensor,  bits: int) -> torch.Tensor: ...


def dequantize(x: torch.Tensor, scale_row: torch.Tensor, scale_col: torch.Tensor,
                   y: torch.Tensor, bits: int) -> torch.Tensor: ...

def int4FusedDequantize(A: torch.Tensor, B: torch.Tensor,
                        scale_row: torch.Tensor, scale_col: torch.Tensor,
                        y: torch.Tensor) -> torch.Tensor: ...

def int8FusedDequantize(A: torch.Tensor, B: torch.Tensor,
                        scale_row: torch.Tensor, scale_col: torch.Tensor,
                        y: torch.Tensor) -> torch.Tensor: ...

def int4SpFusedDequantize(A: torch.Tensor, B: torch.Tensor, E: torch.Tensor,
                          scale_row: torch.Tensor, scale_col: torch.Tensor,
                          y: torch.Tensor) -> torch.Tensor: ...

def int8SpFusedDequantize(A: torch.Tensor, B: torch.Tensor, E: torch.Tensor,
                          scale_row: torch.Tensor, scale_col: torch.Tensor,
                          y: torch.Tensor) -> torch.Tensor: ...
