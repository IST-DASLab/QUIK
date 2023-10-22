import torch


def int4Matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor: ...

def int8Matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor: ...

def int4SpMatmul(A: torch.Tensor, B: torch.Tensor, E: torch.Tensor) -> torch.Tensor: ...


def reorderMeta(E: torch.Tensor) -> torch.Tensor: ...

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