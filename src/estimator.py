from typing import Any, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

from utils import MAX_BATCH_SIZE, get_gpu_memory


def estimate_TransMIL_memory_usage(input_size: Tuple[int, int, int]) -> float:
    """
    Estimate the VRAM memory usage (in megabytes) during the forward pass of TransMIL.

    This includes intermediate activations from:
        - Input projection (fc1)
        - Padding and cls token
        - Transformer layers (x2)
        - Positional encoding (PPEG)
        - LayerNorm and final classifier

    Assumes activations are stored as float32 (4 bytes per value).

    Args:
        input_size (Tuple[int, int, int]): A tuple (B, N, C) representing:
            - B: batch size (number of bags)
            - N: number of instances (patches) per bag
            - C: feature dimension of each instance (e.g., 1024)

    Returns:
        float: Estimated VRAM usage in megabytes (MB)
    """
    B, n, _ = input_size
    total_elements = 0

    # fc1: [B, n, 512]
    total_elements += B * n * 512

    # Pad to square
    _H = _W = int(np.ceil(np.sqrt(n)))
    padded_N = _H * _W
    added_tokens = padded_N - n
    total_elements += B * added_tokens * 512  # padding

    # cls_token: [B, 1, 512]
    total_elements += B * 1 * 512

    total_tokens = padded_N + 1

    # layer1: [B, total_tokens, 512]
    total_elements += B * total_tokens * 512
    # pos_layer: [B, total_tokens, 512]
    total_elements += B * total_tokens * 512
    # layer2: [B, total_tokens, 512]
    total_elements += B * total_tokens * 512
    # norm: [B, total_tokens, 512]
    total_elements += B * total_tokens * 512
    # CLS token after norm: [B, 512]
    total_elements += B * 512
    # final classifier output: [B, 2]
    total_elements += B * 2

    # Convert element count to megabytes (float32 → 4 bytes)
    activation_memory_mb = total_elements * 4 / (1024 ** 2)
    return activation_memory_mb

def estimate_dynamic_vram_usage(
    model_cls: Type[nn.Module],
    input_dim: int = 1024,
    tiles_per_bag: int = 100,
    batch_size: int = 4,
    num_classes: int = 2,
    return_rounded: bool = True
) -> float:
    """
    Estimate the approximate VRAM usage dynamically (in MB) for a given torch module.

    Args:
        model_cls (Type[nn.Module]): The MIL model class (e.g., TransMIL).
        input_dim (int): Feature vector size per tile (default 1024).
        tiles_per_bag (int): Number of tiles per bag/sample (default 100).
        batch_size (int): Number of samples per batch (default 4).
        num_classes (int): Number of output classes for the model. This should be 2.
        return_rounded (bool): Whether to round the result to 2 decimal places (default True).

    Returns:
        float: Estimated VRAM usage in megabytes (MB).
    """
    # Initiate Model and set it to evaluation mode
    try:
        model: nn.Module = model_cls(n_feats=input_dim, n_out=num_classes).cuda()
    except TypeError:
        model: nn.Module = model_cls()
    model.eval()

    # Dummy input tensor for dry run
    dummy_input: torch.Tensor = torch.randn(batch_size, tiles_per_bag, input_dim).cuda()
    lens = torch.tensor([tiles_per_bag] * batch_size).cuda()

    output_sizes: list[int] = []

    def output_size_hook(module: nn.Module, input: Any, output: Any) -> None:
        """Forward hook to count the number of elements in a feature map

        Args:
            module (nn.Module): Module to count the number of elements of intermediate feature maps of
            input (Any): input
            output (Any): output (should be a tensor or a iterable)
        """
        if isinstance(output, torch.Tensor):
            output_sizes.append(output.numel())
        elif isinstance(output, (list, tuple)):
            output_sizes.extend(o.numel() for o in output if isinstance(o, torch.Tensor))

    # Register hooks to all modules
    hooks = [m.register_forward_hook(output_size_hook)
             for m in model.modules() if not isinstance(m, nn.Sequential)]

    # Dry run
    # TODO | Check time requirements
    with torch.no_grad():
        model(dummy_input, lens)

    # Removing hooks
    # TODO | Check if necessary
    for h in hooks:
        h.remove()

    # sum of output sizes in Mb
    output_sum = sum(output_sizes) * 4 / (1024 ** 2)
    param_mem = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    buffer_mem = sum(b.numel() for b in model.buffers()) * 4 / (1024**2)
    input_mem = dummy_input.numel() * 4 / (1024**2) + lens.numel() * 4 / (1024**2)
    sum_mem = output_sum + param_mem + buffer_mem + input_mem
    return round(sum_mem, 2) if return_rounded else sum_mem

def adjust_batch_size(
    model_cls: Type[nn.Module],
    initial_batch_size: int,
    num_slides: int,
    input_dim: int,
    tiles_per_bag: int
) -> int:
    """Adjusts batch size based on the available memory

    Args:
        model_cls (Type[nn.Module]): The MIL model class (e.g., TransMIL).
        initial_batch_size (int): Initial batch size
        num_slides (int): Number of slides in the dataset (used to compute upper bound for batch size)
        input_dim (int): Feature vector size (usually 1024)
        tiles_per_bag (int): Number of tiles per bag

    Returns:
        int: Batch size adjusted to available memory
    """
    global MAX_BATCH_SIZE
    
    # Get estimated memory usage of model and free memory (both in Mb)
    estimated_mem_usage = estimate_dynamic_vram_usage(
        model_cls,
        input_dim,
        tiles_per_bag,
        initial_batch_size
    )
    free_mem = get_gpu_memory()["free_MB"]
    batch_size_limit = min(
        num_slides // 2,
        MAX_BATCH_SIZE
    )

    # Adjust batch size according to free memory (with upper bound in mind)
    adjusted_batch_size = min(
        round(free_mem / estimated_mem_usage) * initial_batch_size, 
        batch_size_limit
    )

    return int(adjusted_batch_size)