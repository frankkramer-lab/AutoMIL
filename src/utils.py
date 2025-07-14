from enum import Enum
from pathlib import Path
from typing import Optional

import openslide
import slideflow as sf
import torch

"""
Variables and utility methods used throughout the project.
"""

# ------------------------- #
# --- Project Variables --- #
# ------------------------- #

# --- General ---

# Random seed for reproducing results
RANDOM_SEED: int = 42

# --- Paths ---



# --- Hyperparameters ---

# Resolution Presets for extracting dataset tiles (specifies tile size and magnification level)
class RESOLUTION_PRESETS(Enum):
    Low   = (1_000, "10x")
    High  = (299, "20x")
    Ultra = (224, "40x")

# --- Bags ---

# Feature Extractor to use
FEATURE_EXTRACTOR: str = "ctranspath"

# --- Training ---

# Learning Rate
LEARNING_RATE: float = 1e-4

# Batch Size
BATCH_SIZE: int = 32

# Number of Epochs
EPOCHS: int = 40

# ----------------------- #
# --- Utility methods --- #
# ----------------------- #

def get_gpu_memory() -> dict:
    """Utility Method that returns the free and total GPU memory

    Returns:
        dict: dictionary with free memory (key: free_MB), the total memory (key: total_MB) and the memory in usage (key: used_MB)
    """
    free_mem, total_mem = torch.cuda.mem_get_info()
    return {
        'free_MB': free_mem // (1024**2),
        'total_MB': total_mem // (1024**2),
        'used_MB': (total_mem - free_mem) // (1024**2)
    }

def get_num_slides(dataset: sf.Dataset) -> int:
    """Returns the number of slides in a given dataset source

    Args:
        dataset (sf.Dataset): Dataset source

    Returns:
        int: number of slides
    """
    return len(dataset.slides())

def get_slide_magnification(slide_path: str) -> Optional[str]:
    """Retrieves magnification from slide properties or estimates it using microns per pixel (mpp)

    Args:
        slide_path (str): path to slide as string

    Returns:
        str | None: magnification as string (e.g. '20') or None if unable
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        # Typical property for objective magnification
        if (mag := slide.properties.get("openslide.objective-power")):
            return mag
        # Estimate magnification from mpp_x
        if (mpp_x := slide.properties.get("openslide.mpp-x")):
            estimated_mag = 10 / float(mpp_x)
            return str(estimated_mag)
    # Estimation (or opening slide) failed
    except Exception:
        return None
    
def get_slide_properties(slide_path: str) -> Optional[dict]:
    try:
        slide = openslide.OpenSlide(slide_path)
        return {key: val for key, val in slide.properties.items()}
    # Estimation (or opening slide) failed
    except Exception:
        return None


def remove_pip_version_specs(required_packages_file: Path, out_file: Optional[Path] = None) -> None:
    """Removes the version specifiers from a file with pip package requirements as produced by pip freeze.

    Example:
        beautifulsoup4==4.13.4
        click==8.1.8
        ***************
        beautifulsoup4
        click==8.1.8

    Args:
        required_packages_file (Path): Path to file with required packages list
        out_file (Optional[Path], optional): Path to write modified package list to. If not specified, takes 'required_packages_file'.

    Raises:
        ValueError: If 'required_packages_file'
    """

    if not required_packages_file.is_file():
        raise ValueError(f"{required_packages_file} is not a file")
    
    with open(required_packages_file, "r") as file_stream:
        lines = file_stream.readlines()

    version_removed_lines = [
        line.split("==")[0] if "==" in line else line
        for line in lines
    ]

    if not out_file: out_file = required_packages_file
    with open(out_file, "w") as file_stream:
        file_stream.write("\n".join(version_removed_lines))