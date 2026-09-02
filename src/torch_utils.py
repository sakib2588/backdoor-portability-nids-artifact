"""Shared batched-inference helper for torch victims (vision CNN + tabular MLP).

Both `src/vision_control.py` and `src/models.py` need to run a trained torch model over more rows
than fit comfortably in one forward pass, moving each chunk to `device` first. This is the one place
that batching loop lives; callers only supply what differs (the per-batch op, and how to concatenate
the result -- torch.cat for tensor outputs, np.concatenate for numpy outputs).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def batched_apply(fn, x: torch.Tensor, device: str, batch_size: int = 1024) -> list:
    """Run `fn` over `x` in chunks of `batch_size`, moving each chunk to `device` first.

    Returns the list of per-batch outputs (whatever type/device `fn` returns) -- the caller
    concatenates with `torch.cat` or `np.concatenate` depending on what it needs out of the batch.
    """
    return [fn(x[i:i + batch_size].to(device)) for i in range(0, len(x), batch_size)]
