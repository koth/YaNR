"""Small numerical-mode helpers shared by the student and its router."""
from contextlib import contextmanager

import torch


@contextmanager
def tf32_matmul(enabled=True):
    """Temporarily enable/disable TF32 for FP32 CUDA matmuls."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
