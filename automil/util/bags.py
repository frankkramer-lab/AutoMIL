#==============================================================================#
#  AutoMIL - Automated Machine Learning for Image Classification in            #
#  Whole-Slide Imaging with Multiple Instance Learning                         #
#                                                                              #
#  Copyright (C) 2026 Jonas Waibel                                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.      #
#==============================================================================#
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