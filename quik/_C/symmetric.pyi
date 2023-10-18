import torch


def quantize(src: torch.Tensor, scale: torch.Tensor,  bits: int) -> torch.Tensor: ...


def dequantize(x: torch.Tensor, scale_row: torch.Tensor, scale_col: torch.Tensor,
                   y: torch.Tensor, bits: int) -> torch.Tensor: ...
