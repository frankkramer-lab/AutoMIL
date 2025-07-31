from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, Optional, TypeVar

import openslide
import pandas as pd
import pyvips
import slideflow as sf
import torch
from slideflow.mil.models import Attention_MIL, TransMIL
from slideflow.mil.models.bistro.transformer import \
    Attention as BistroTransformer
from slideflow.util import log as slideflow_log

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


# --- Model Selection ---

class ModelType(Enum):
    Attention_MIL     = Attention_MIL
    TransMIL          = TransMIL
    BistroTransformer = BistroTransformer

# --- Hyperparameters ---

# Resolution Presets for extracting dataset tiles (specifies tile size and magnification level)
class RESOLUTION_PRESETS(Enum):
    Ultra_Low = (2_000, "5x")
    Low   = (1_000, "10x")
    High  = (299, "20x")
    Ultra = (224, "40x")

COMMON_MPP_VALUES = {
    "20x": 0.5,
    "40x": 0.25,
    "10x": 1.0,
    "5x": 2.0,
}

# --- Bags ---

# Feature Extractor to use
FEATURE_EXTRACTOR: str = "ctranspath"

# --- Training ---

# Learning Rate
LEARNING_RATE: float = 1e-4

# Batch Size
BATCH_SIZE: int = 32

MAX_BATCH_SIZE: int = 200  # Maximum batch size for training, used for estimating VRAM usage

# Number of Epochs
EPOCHS: int = 40

# --- Rich Formatting Colors ---

# Colors for variables in log messages
INFO_CLR: str = "cyan"        # For general variables and parameters
SUCCESS_CLR: str = "green"    # For success messages and completed operations
ERROR_CLR: str = "red"        # For error messages and warnings
HIGHLIGHT_CLR: str = "yellow" # For highlighting important information

# ----------------------- #
# --- Utility methods --- #
# ----------------------- #

# --- Logging ---

def get_vlog(verbose: bool) -> Callable:
    """Returns a logging function that only logs messages if verbose is True.

    Args:
        verbose (bool): Verbosity flag

    Returns:
        Callable[[str, Any], None]: Logging function
    """
    def _vlog(message: str) -> None:
        if verbose:
            slideflow_log.info(message)
    return _vlog

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
        dataset (sf.Dataset): Dataset object

    Returns:
        int: number of slides
    """
    return len(dataset.slide_paths())

def get_slide_magnification(slide_path: str | Path) -> Optional[str]:
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

def get_lowest_magnification(slide_dir: Path) -> Optional[str]:
    """Returns the lowest magnification from a list of slide paths.

    Args:
        slide_paths (list[str]): List of slide paths

    Returns:
        Optional[str]: Lowest magnification as string or None if no valid magnification found
    """
    magnifications = []
    for slide_path in slide_dir.iterdir():
        if (mag := get_slide_magnification(slide_path)):
            try:
                magnifications.append(float(mag))
            except ValueError:
                continue  # Skip invalid magnifications
    return str(min(magnifications)) if magnifications else None

def get_mpp_from_slide(slide_path: Path) -> Optional[float]:
    """
    Extract microns per pixel (MPP) from a slide file using OpenSlide.

    Args:
        slide_path (str): Path to the slide file (.svs, .tiff, etc.)

    Returns:
        Tuple[float, float]: (mpp_x, mpp_y) if available, else None
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        if (mpp_x := slide.properties.get("openslide.mpp-x")) is None:
            return None
        else:
            return float(mpp_x)
    except (openslide.OpenSlideError, ValueError, TypeError, KeyError):
        return None
    
def calculate_average_mpp(slide_dir: Path, return_rounded: bool = True) -> Optional[float]:
    """Calculates the average microns per pixel (MPP) from all slides in a directory.

    Args:
        slide_dir (Path): Path to the directory containing slide files.

    Returns:
        Optional[float]: Average MPP if available, else None.
    """
    mpp_values = []
    for slide_path in slide_dir.iterdir():
        if (mpp := get_mpp_from_slide(slide_path)):
            mpp_values.append(mpp)
    average_mpp = sum(mpp_values) / len(mpp_values)
    if return_rounded and average_mpp is not None:
        return round(average_mpp, 2)
    return average_mpp

def get_slide_properties(slide_path: str) -> Optional[dict]:
    """Retrieves slide properties as a dictionary.

    Args:
        slide_path (str): path to slide as string

    Returns:
        Optional[dict]: dictionary with slide properties or None if unable to retrieve
    """
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

def get_bag_avg_and_num_features(bags_dir: Path) -> tuple[int, int]:
    """Computes the average number of tiles per bag (for estimation) and the number of features per tile.

    Args:
        bags_dir (Path): Path to the directory containing the feature bags.

    Raises:
        ValueError: If no valid .pt feature bags are found in the specified directory.

    Returns:
        int: Tuple containing the average number of tiles per bag and the number of features per tile.
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

def get_unique_labels(annotations_file: Path, label_column: str) -> list[str]:
    """Extracts list of unique labels from the specified column (should be the labels) in the annotations file.

    Args:
        annotations_file (Path): Path to the annotations CSV file.
        label_column (str):      Name of the column containing labels.

    Returns:
        set: List of unique labels found in the specified column.
    """
    annotations = pd.read_csv(annotations_file)
    return [str(label) for label in annotations[label_column].dropna().unique()]

# --- Batch Generator ---

# Generic Type for Annotations
T = TypeVar('T')

def batch_generator(input_list: list[T], batch_size: int) -> Generator[list[T], None, None]: 
    """
    Generator that yields batches of a given size from the input list.
    
    Args:
        input_list (list): List of items to be batched.
        batch_size (int):  Size of each batch.
    
    Yields:
        batch (list):      A batch of items from the input list.
    """
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

# --- PNG -> TIFF Conversion ---

def convert_img_to_tiff(in_path: Path, out_path: Path) -> str | bool:
    """Converts a given image into a .tiff file

    Args:
        in_path (Path):     Input path to image (.png, .jpg, ...)
        out_path (Path):    Output path to where to save the converted .tiff to
    
    Returns:
        str: error message in case of failure or an empty string in case of success
    """
    try:
        image = pyvips.Image.new_from_file(in_path)
        if not isinstance(image, pyvips.Image):
            return f"Failed to read image from {in_path}"
        # Save as .tiff with pyramid and tile options
        image.write_to_file(out_path, tile=True, pyramid=True, bigtiff=True)
    except Exception as err:
        return str(err)
    else:
        return ""

def convert_worker(in_path: Path, out_folder: Path, verbose: bool = True) -> Path | None:
    """Worker function for converting a single image to .tiff format.

    Args:
        in_path (Path): Path to input image file to convert
        out_folder (Path): Path to output folder in which to save the tiff
        verbose (bool): Verbose flag to log progress messages

    Returns:
        Path | None: Path to the converted .tiff file if successful, None if conversion failed
    """
    out_path = out_folder / f"{in_path.stem}.tiff"
    vlog = get_vlog(verbose)

    # Skip already converted images
    if out_path.exists():
        if verbose:
            print(f"{out_path} already exists. Skipping")
        return out_path

    if err := convert_img_to_tiff(in_path, out_path):
        vlog(f"Error while converting {in_path}: {err}")
        return None
    else:
        vlog(f"Successfully converted {in_path.name}")
        return out_path

def batch_conversion(file_list: list[Path], out_folder: Path, verbose: bool = True) -> list[Path]:
    """Converts a list of image files to .tiff and saves them in the given output folder.

    Args:
        file_list (list[Path]):     List of file paths to convert
        out_folder (Path):          Path to output folder in which to save the tiffs.
                                    The tiff file will inherit the original files stem (i.e /input/image.png => /output/image.tiff)
        verbose (bool, optional):   Whether to log progress messages. Defaults to False.
    
    Returns:
        file_list (list[Path]):     List of paths to .tiff files
    """
    out_list = []   # List of output files
    vlog = get_vlog(verbose)

    for in_path in file_list:
        # Path to output file
        out_path = out_folder / f"{in_path.stem}.tiff"

        # Skip already converted images
        if out_path.exists():
            vlog(f"{out_path} already exists. Skipping")
            out_list.append(out_path)
            continue
        
        # Convert image to tiff
        if err := convert_img_to_tiff(in_path, out_path):
            vlog(f"Error while converting {in_path}: {err}")
        else:
            vlog(f"Successfully converted {in_path.name}")
            out_list.append(out_path)
        
    return out_list

def batch_conversion_concurrent(file_list: list[Path], out_folder: Path, verbose: bool = True) -> list[Path]:
    """Multithreaded version of batch_conversion. Uses convert_worker.

    Args:
        file_list (list[Path]): List of file paths to convert
        out_folder (Path): Path to output folder in which to save the tiffs.
        verbose (bool, optional): Verbose flag. Defaults to True.

    Returns:
        list[Path]: List of paths to converted .tiff files.
    """
    out_list = []
    max_workers = min(32, len(file_list))  # Reasonable upper limit on threads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_worker, f, out_folder, verbose) for f in file_list]

        for future in as_completed(futures):
            result = future.result()
            if result:
                out_list.append(result)

    return out_list