"""
Utility functions for extracting information from feature bags.
"""
from pathlib import Path

import torch


def get_bag_avg_and_num_features(bags_dir: Path) -> tuple[int, int]:
    """Compute the average number of tiles per bag and the number of features per tile.

    Args:
        bags_dir: Path to the directory containing the feature bags

    Returns:
        A tuple containing the average number of tiles per bag and the number of features per tile.

    Raises:
        ValueError: If no valid .pt feature bags are found in the specified directory.
    """
    num_tiles = []
    num_features = 0
    for bag_path in bags_dir.glob("*.pt"):
        try:
            tensor = torch.load(bag_path, map_location="cpu")
            # Shape: (tiles_per_bag, num_features)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                num_tiles.append(tensor.shape[0])
                num_features = tensor.shape[1]
        except Exception:
            continue  # skip corrupt or unexpected bags

    if not num_tiles:
        raise ValueError(f"No valid .pt feature bags found in {bags_dir}")
    
    return (int(sum(num_tiles) / len(num_tiles)), num_features)