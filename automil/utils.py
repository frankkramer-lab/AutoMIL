import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Optional, TypeVar

import cv2
import numpy as np
import openslide
import pandas as pd
import pyvips
import slideflow as sf
import torch
import torch.nn as nn
from PIL import Image
from slideflow.io import TFRecordWriter, write_tfrecords_multi
from slideflow.io.torch import serialized_record
from slideflow.mil.models import Attention_MIL, TransMIL
from slideflow.mil.models.bistro.transformer import \
    Attention as BistroTransformer
from slideflow.util import log as slideflow_log
from slideflow.util.tfrecord2idx import create_index

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

    @property
    def model_name(self) -> str:
        """The associated string name to pass to slideflow"""
        name_mapping = {
            Attention_MIL: "attention_mil",
            TransMIL: "transmil", 
            BistroTransformer: "bistro.transformer"
        }
        return name_mapping[self.value]
    
    @property
    def model_class(self):
        """The associated torch module"""
        return self.value

# --- Hyperparameters ---

# Resolution Presets for extracting dataset tiles (specifies tile size and magnification level)
class RESOLUTION_PRESETS(Enum):
    Ultra_Low = (2_000, "5x")
    Low   = (1_000, "10x")
    High  = (299, "20x")
    Ultra = (224, "40x")

    @property
    def tile_px(self) -> int:
        """Tile size in pixels"""
        return self.value[0]
    
    @property
    def magnification(self) -> str:
        """Tile magnification level"""
        return self.value[1]

# Based on commonly cited microns per pixel (mpp) values for different default magnifications
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

# Maximum Batch size for training, used for estimating VRAM usage
# Choice of 100 is inspired by https://arxiv.org/abs/2503.10510v1
MAX_BATCH_SIZE: int = 100

# Number of Epochs
EPOCHS: int = 40

# --- Rich Formatting Colors ---

# Colors for variables in log messages
INFO_CLR:       str = "cyan"       # General purpose  | Variable names and parameters
SUCCESS_CLR:    str = "green"      # Success Messages | Completed operations
ERROR_CLR:      str = "red"        # Error Messages   | Warnings
HIGHLIGHT_CLR:  str = "yellow"     # Highlighting     | Important information

# --- Logging Flags ---

class LogLevel(Enum):
    INFO    = 20
    DEBUG   = 10
    WARNING = 30
    ERROR   = 40

# ----------------------- #
# --- Utility methods --- #
# ----------------------- #

# === Input Validation === #

def is_input_pretiled(slide_dir: Path, slide_ids: list[str] | None = None) -> bool:
    """Check if the input slide directory contains pretiled slides.

    `slide_dir` is considered pretiled, if the following conditions are met:
        1. `slide_dir` contains a subdirectory for each slide (Ideally named after the slide ID)
        2. Each slide subdirectory contains loose image tiles (.tif, .tiff, .png, .svs)
    
    Example structure:
    ```
        slide_dir/
        |-- slide1/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
        |-- slide2/
        |    |-- tile_0_0.png
        |    |-- tile_0_1.png
        |    |-- ...
    ```

    Args:
        slide_dir: Path to the slide directory
        slide_ids: Optional list of expected slide IDs (will check if subdirectory names match these IDs)
    Returns:
        True if the slide directory contains pretiled slides, False otherwise.
    """
    tile_formats = [".png", ".svs", ".tiff", ".tif"]
    if not slide_dir.is_dir() or not slide_dir.exists():
        return False
    
    # Collect all subdirectories
    slide_subdirs = [entry for entry in slide_dir.iterdir() if entry.is_dir()]
    
    # Check if there even are subdirectories
    if not slide_subdirs:
        return False

    # If slide IDs are provided, check subdirectory names against them
    if slide_ids:
        if not all(
            slide_subdir.name in slide_ids
            for slide_subdir in slide_subdirs
        ):
            return False

    # Check that subdirectories contain loose image files
    if not all(
        file.suffix in tile_formats for slide_subdir in slide_subdirs
        for file in slide_subdir.iterdir() if file.is_file()
    ):
        return False

    # If all checks were passed, this is (likely) a pretiled input
    return True


# === Model Instantiation === #

def create_model_instance(
    model_type: ModelType,
    input_dim: int,
    n_out: int = 2
) -> nn.Module:
    """Safely create a model instance with the correct parameters.
    
    Args:
        model_type: The ModelType enum
        input_dim: Input feature dimension
        n_out: Number of output classes
        
    Raise:
        Exception: If model instantiation fails

    Returns:
        Instantiated model
    """
    try:
        match model_type:

            case ModelType.Attention_MIL:
                model_cls = Attention_MIL
                return model_cls(n_feats=input_dim, n_out=n_out)
            
            case ModelType.TransMIL:
                model_cls = TransMIL
                return model_cls(n_feats=input_dim, n_out=n_out)
            
            case ModelType.BistroTransformer:
                model_cls = BistroTransformer
                return model_cls(dim=input_dim)

            case _:
                return model_cls()
    except Exception as e:
        slideflow_log.error(f"Error while creating model instance: {e}")
        raise e

# --- Logging ---

def get_vlog(verbose: bool) -> Callable:
    """Return a logging function that only logs messages if verbose is True.

    Args:
        verbose: Verbosity flag

    Returns:
        Logging function that conditionally logs messages.
    """
    def _vlog(message: str, log_level: LogLevel = LogLevel.INFO) -> None:
        if verbose:
            match log_level:
                case LogLevel.INFO:
                    slideflow_log.info(message)
                case LogLevel.DEBUG:
                    slideflow_log.debug(message)
                case LogLevel.WARNING:
                    slideflow_log.warning(message)
                case LogLevel.ERROR:
                    slideflow_log.error(message)
    return _vlog

def format_ensemble_summary(
    num_models: int,
    confusion_matrix: np.ndarray,
    auc: float,
    ap: float,
    acc: float,
    f1: float,
) -> str:
    """
    Format ensemble evaluation metrics into a readable summary.
    
    Args:
        num_models: Number of models in the ensemble
        confusion_matrix: Confusion matrix (can be binary 2x2 or multi-class NxN)
        auc: Area under curve score
        ap: Average precision score
        acc: Accuracy score
        f1: F1 score (macro for multiclass, regular for binary)
        f1_macro: Macro F1 score (multiclass only)
        f1_weighted: Weighted F1 score (multiclass only)
    
    Returns:
        Formatted summary string
    """
    n_classes = confusion_matrix.shape[0]
    
    if n_classes == 2:
        # Binary classification - original format
        tn, fp, fn, tp = confusion_matrix.ravel()
        cm_formatted = f"""
                 Predicted
                 0     1
    Actual 0  {tn:4d}  {fp:4d}
           1  {fn:4d}  {tp:4d}"""
    else:
        # Multi-class classification - matrix format
        cm_formatted = "\n                 Predicted\n"
        cm_formatted += "              " + "".join([f"{i:>6}" for i in range(n_classes)]) + "\n"
        for i, row in enumerate(confusion_matrix):
            cm_formatted += f"    Actual {i:>1}  " + "".join([f"{val:>6}" for val in row]) + "\n"

    # Build metrics section
    metrics_text = f"""-- AUC: {auc:.3f}
-- Average Precision: {ap:.3f}  
-- Accuracy: {acc:.3%}
-- F1 Score: {f1:.3f}"""
    summary = f"""
Ensemble Evaluation Metrics | Models: {num_models} | Classes: {n_classes}
{metrics_text}
-- Confusion Matrix:{cm_formatted}
"""
    return summary

# --- Slide / Dataset Info ---

def get_num_slides(dataset: sf.Dataset) -> int:
    """Return the number of slides in a given dataset source.

    Args:
        dataset: Dataset object

    Returns:
        Number of slides in the dataset.
    """
    return len(dataset.slides())

def get_slide_magnification(slide_path: str | Path) -> Optional[str]:
    """Retrieve magnification from slide properties or estimate using microns per pixel.

    Args:
        slide_path: Path to slide as string or Path object

    Returns:
        Magnification as string (e.g. '20') or None if unable to determine.
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
    """Return the lowest magnification from slides in a directory.

    Args:
        slide_dir: Path to directory containing slide files

    Returns:
        Lowest magnification as string or None if no valid magnification found.
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
    """Extract microns per pixel (MPP) from a slide file using OpenSlide.

    Args:
        slide_path: Path to the slide file (.svs, .tiff, etc.)

    Returns:
        MPP value if available, else None.
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
    """Calculate the average microns per pixel (MPP) from all slides in a directory.

    Args:
        slide_dir: Path to the directory containing slide files
        return_rounded: Whether to round the result to 2 decimal places

    Returns:
        Average MPP if available, else None.
    """
    mpp_values = []
    for slide_path in slide_dir.iterdir():
        if (mpp := get_mpp_from_slide(slide_path)):
            mpp_values.append(mpp)
    if not mpp_values: return None
    average_mpp = sum(mpp_values) / len(mpp_values)
    if return_rounded and average_mpp is not None:
        return round(average_mpp, 2)
    return average_mpp

def get_slide_properties(slide_path: str) -> Optional[dict]:
    """Retrieve slide properties as a dictionary.

    Args:
        slide_path: Path to slide as string

    Returns:
        Dictionary with slide properties or None if unable to retrieve.
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        return {key: val for key, val in slide.properties.items()}
    # Estimation (or opening slide) failed
    except Exception:
        return None

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

def get_unique_labels(annotations_file: Path, label_column: str) -> list[str]:
    """Extract list of unique labels from the specified column in the annotations file.

    Args:
        annotations_file: Path to the annotations CSV file
        label_column: Name of the column containing labels

    Returns:
        List of unique labels found in the specified column.
    """
    annotations = pd.read_csv(annotations_file)
    return [str(label) for label in annotations[label_column].dropna().unique()]

# --- Batch Generator ---

# Generic Type for Annotations
T = TypeVar('T')

def batch_generator(input_list: list[T], batch_size: int) -> Generator[list[T], None, None]: 
    """Generate batches of a given size from the input list.
    
    Args:
        input_list: List of items to be batched
        batch_size: Size of each batch
    
    Yields:
        A batch of items from the input list.
    """
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

# --- PNG -> TIFF Conversion ---

def convert_img_to_tiff(in_path: Path, out_path: Path) -> str:
    """Convert a given image into a TIFF file.

    Args:
        in_path: Input path to image (.png, .jpg, ...)
        out_path: Output path where to save the converted TIFF
    
    Returns:
        Error message in case of failure or an empty string in case of success.
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
    """Worker function for converting a single image to TIFF format.

    Args:
        in_path: Path to input image file to convert
        out_folder: Path to output folder in which to save the TIFF
        verbose: Whether to log progress messages

    Returns:
        Path to the converted TIFF file if successful, None if conversion failed.
    """
    out_path = out_folder / f"{in_path.stem}.tiff"
    vlog = get_vlog(verbose)

    # Skip already converted images
    if out_path.exists():
        vlog(f"{out_path} already exists. Skipping")
        return out_path

    if err := convert_img_to_tiff(in_path, out_path):
        vlog(f"Error while converting [{INFO_CLR}]{in_path}[/]: {err}")
        return None
    else:
        vlog(f"Successfully converted [{INFO_CLR}]{in_path.name}[/]")
        return out_path

def batch_conversion(file_list: list[Path], out_folder: Path, verbose: bool = True) -> list[Path]:
    """Convert a list of image files to TIFF and save them in the output folder.

    Args:
        file_list: List of file paths to convert
        out_folder: Path to output folder in which to save the TIFFs
        verbose: Whether to log progress messages
    
    Returns:
        List of paths to converted TIFF files.

    Note:
        The TIFF file will inherit the original file's stem (e.g. /input/image.png -> /output/image.tiff).
    """
    out_list = []   # List of output files
    vlog = get_vlog(verbose)

    for in_path in file_list:
        # Path to output file
        out_path = out_folder / f"{in_path.stem}.tiff"

        # Skip already converted images
        if out_path.exists():
            vlog(f"[{INFO_CLR}]{out_path}[/] already exists. Skipping")
            out_list.append(out_path)
            continue
        
        # Convert image to tiff
        if err := convert_img_to_tiff(in_path, out_path):
            vlog(f"Error while converting [{INFO_CLR}]{in_path}[/]: {err}")
        else:
            vlog(f"Successfully converted [{INFO_CLR}]{in_path.name}[/]")
            out_list.append(out_path)
        
    return out_list

def batch_conversion_concurrent(file_list: list[Path], out_folder: Path, verbose: bool = True) -> list[Path]:
    """Multithreaded version of batch_conversion using convert_worker.

    Args:
        file_list: List of file paths to convert
        out_folder: Path to output folder in which to save the TIFFs
        verbose: Whether to log progress messages

    Returns:
        List of paths to converted TIFF files.
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

# --- GPU Memory Info ---

def get_free_memory() -> float:
    """Return the amount of free memory on the current GPU in MB."""
    free_mem, _ = torch.cuda.mem_get_info()
    return free_mem / (1024 ** 2)  # Convert to MB

def get_cuda_gpu_memory_used() -> int:
    """Retrieves the total memory the cuda driver has reserved using nvidia-smi.

    Returns:
        int: Memory in MB
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
    )
    return int(result.decode().strip().split('\n')[0])  # memory in MB of GPU 0

def reserve_tensor_memory() -> float:
    """Gets the amount of memory overhead reserved when allocating a minimal tensor (small as possible).

    Returns:
        float: Memory overhead for tensor allocation in MB
    """
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()

    # We can get the memory overhead by measuring
    # the total memory reserved before and after allocating a minimal tensor
    before = get_cuda_gpu_memory_used()
    a = torch.FloatTensor(1).cuda()
    torch.cuda.synchronize() # Ensure the allocation is complete
    after = get_cuda_gpu_memory_used()
    return after - before

# --- Miscalleneous ---

def remove_pip_version_specs(required_packages_file: Path, out_file: Optional[Path] = None) -> None:
    """Remove version specifiers from a file with pip package requirements.

    Processes a file produced by pip freeze and removes version specifiers,
    converting "beautifulsoup4==4.13.4" to "beautifulsoup4".

    Args:
        required_packages_file: Path to file with required packages list
        out_file: Path to write modified package list to. If not specified, overwrites the input file.

    Raises:
        ValueError: If required_packages_file is not a valid file.

    Example:
        >>> remove_pip_version_specs(Path("requirements.txt"))
        # Converts: beautifulsoup4==4.13.4 -> beautifulsoup4
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

# --- TFRecords Conversion ---

def pretiled_to_tfrecords(
    pretiled_root: Path,
    outdir: Path,
    overwrite: bool = False,
    verbose: bool = True
):
    """
    Convert loose tiles organized by slide subfolders into TFRecords,
    assigning sequential indices compatible with Slideflow.

    Expected structure:
        pretiled_root/
            slideA/
                tile_00001.tif
                tile_00002.tif
                ...
            slideB/
                tile_00001.tif
                ...
    or:
        pretiled_root/
            slideA/
                tile_00001.tiff
                tile_00002.tiff
                ...
            slideB/
                tile_00001.tiff
                ...

    Args:
        pretiled_root: Root folder with one subfolder per slide.
        outdir: Destination directory for .tfrecord files.
        overwrite: Whether to overwrite existing TFRecord files.
        verbose: Whether to print progress information.
    """
    vlog = get_vlog(verbose)
    outdir.mkdir(parents=True, exist_ok=True)

    slide_dirs = sorted([entry for entry in pretiled_root.iterdir() if entry.is_dir()])
    if not slide_dirs:
        raise ValueError(f"No slide subdirectories found in {pretiled_root}")

    for slide_dir in slide_dirs:
        slide_id = slide_dir.name
        tfrecord_path = outdir / f"{slide_id}.tfrecords"
        if tfrecord_path.exists() and not overwrite:
            vlog(f"[{INFO_CLR}]{tfrecord_path}[/] already exists. Skipping.")
            continue
        
        extensions = [".png", ".svs", ".tif", ".tiff"]
        tile_paths = sorted([tile for tile in slide_dir.iterdir() if tile.suffix in extensions])
        if not tile_paths:
            vlog(f"No tiles found in [{INFO_CLR}]{slide_dir}[/], skipping.")
            continue

        writer = TFRecordWriter(str(tfrecord_path))
        vlog(f"Writing [{INFO_CLR}]{len(tile_paths)}[/] tiles for [{INFO_CLR}]{slide_id}[/]")

        for idx, tile_path in enumerate(tile_paths):
            # Aritrary grid location
            grid_width = len(tile_paths) if len(tile_paths) < 32 else 32
            loc_x = int(idx % grid_width)
            loc_y = int(idx // grid_width)

            # Convert tile to bytes
            label = bytes(slide_id, 'utf-8')
            img = np.array(Image.open(tile_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_string = cv2.imencode(".png", img)[1].tobytes()

            # Serializze and write record
            record = serialized_record(label, image_string, loc_x=loc_x, loc_y=loc_y)
            writer.write(record)

        writer.close()

        # Create index file for the TFRecords
        create_index(str(tfrecord_path))

def pretiled_to_tfrecords_multi(
    pretiled_root: Path,
    outdir: Path,
    verbose: bool = True
):
    """
    Convert loose .tif/.tiff tiles organized by slide subfolders into TFRecords

    Expected structure:
        pretiled_root/
            slideA/
                tile_00001.tif
                tile_00002.tif
                ...
            slideB/
                tile_00001.tif
                ...
    or:
        pretiled_root/
            slideA/
                tile_00001.tiff
                tile_00002.tiff
                ...
            slideB/
                tile_00001.tiff
                ...

    Args:
        pretiled_root: Root folder with one subfolder per slide.
        outdir: Destination directory for .tfrecord files.
    """
    vlog = get_vlog(verbose)
    outdir.mkdir(parents=True, exist_ok=True)

    vlog(f"Converting pretiled images in [{INFO_CLR}]{pretiled_root}[/] to TFRecords in [{INFO_CLR}]{outdir}[/]")
    # Use write_tfrecords_multi to process multiple slide subdirectories
    # Each subdirectory name is taken as a slide name
    write_tfrecords_multi(
        str(pretiled_root),
        str(outdir)
    )
