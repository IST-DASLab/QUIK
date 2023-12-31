import torch
from typing import Tuple

def quantize(src: torch.Tensor, int_indices: torch.Tensor, fp_indices: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def dequantize(x: torch.Tensor, meta: torch.Tensor, scale_col: torch.Tensor,
               wReduced: torch.Tensor, y: torch.Tensor, bits: int) -> torch.Tensor: ...


def find_meta(x: torch.Tensor, bits: int) -> torch.Tensor: ...

def quantizeOld(src: torch.Tensor, meta: torch.Tensor, bits: int) -> torch.Tensor: ...
