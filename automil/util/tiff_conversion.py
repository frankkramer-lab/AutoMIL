from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, TypeVar

from .constants import INFO_CLR
from .logging import get_vlog


# === PNG -> TIFF Conversion Utilities === #
def convert_img_to_tiff(in_path: Path, out_path: Path) -> str:
    """Convert a given image into a TIFF file.

    Args:
        in_path: Input path to image (.png, .jpg, ...)
        out_path: Output path where to save the converted TIFF
    
    Returns:
        Error message in case of failure or an empty string in case of success.
    """
    try:
        import pyvips
    except ImportError as e:
        raise RuntimeError(
            "pyvips is required for TIFF conversion. "
            "Install with: pip install automil[vips]"
        ) from e

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

# === Pure batching === #

T = TypeVar("T")

def batch_generator(items: list[T], batch_size: int) -> Generator[list[T], None, None]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]