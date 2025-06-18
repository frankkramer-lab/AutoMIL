import numpy as np


def estimate_TransMIL_memory_usage(input_size: tuple[int, int, int]) -> float:
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
        input_size (tuple[int, int, int]): A tuple (B, N, C) representing:
            - B: batch size (number of bags)
            - N: number of instances (patches) per bag
            - C: feature dimension of each instance (e.g., 1024)

    Returns:
        float: Estimated VRAM usage in megabytes (MB)
    """
    B, N, C = input_size
    total_elements = 0

    # fc1: [B, N, 512]
    total_elements += B * N * 512

    # Pad to square
    _H = _W = int(np.ceil(np.sqrt(N)))
    padded_N = _H * _W
    added_tokens = padded_N - N
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

if __name__ == "__main__":
    print(estimate_TransMIL_memory_usage((4, 256, 1024)))